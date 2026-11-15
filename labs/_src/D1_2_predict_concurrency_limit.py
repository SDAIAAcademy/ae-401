# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 1.2 · Concurrency limit at three context lengths
#
# Day 1 · Session 2 (Key-value cache arithmetic) · 25 min.
#
# **AI off for tasks 1 to 3** (the hand sizing); the assistant may explain the engine log in task 4.
# You predict how many of Musamir's conversations fit in the L4's key-value cache at 2,000, 4,000 and 8,000
# tokens, then check against the block count the engine reports and a run that pushes it to saturation.
# The outcome's evidence is a prediction within 25% of the measured run.
#
# **Starts from** `data/model_cards.json`, `data/accelerators.json`, `data/deployments/baseline.json`.
#
# **Tasks**
# - [ ] 1 · Cache bytes per token for the base model (4 min)
# - [ ] 2 · Memory left for the cache after weights and reserve (5 min)
# - [ ] 3 · Predict the concurrency limit at three context lengths (5 min)
# - [ ] 4 · Read the engine's blocks and run it to saturation (6 min)
# - [ ] 5 · Compare, explain the gap, check the 25% band (5 min)
# - [ ] ✱ Stretch · Predict the limit again with 8-bit weights

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ae-401.git"
    if not Path("/content/ae-401").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ae-401"], check=True)
    os.chdir("/content/ae-401/labs")
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import pandas as pd
import servelab as S

m8, l4 = S.models()["qwen3-8b"], S.accelerators()["L4"]
cfg = S.Config.load("baseline").but(max_num_seqs=64, enable_chunked_prefill=True, max_num_batched_tokens=2048)
print({k: m8[k] for k in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")})
print("gpu_memory_utilization:", cfg.gpu_memory_utilization, "| the platform team's note: the engine keeps about 1.5 GiB "
      "for activations, graphs and its runtime outside the cache")

# %% [markdown]
# ## 1 · Cache bytes per token for the base model (4 min)
#
# For every token in a conversation the engine keeps a key and a value vector in every layer. With grouped-query
# attention there are only `num_key_value_heads` of them (8 here, against 32 query heads).
#
# ✏️ **Exercise** · `kv_per_token(m, bytes_per_value)`: `2 × layers × KV heads × head_dim × bytes`.

# %%
def kv_per_token(m, bytes_per_value=2):
    # BEGIN SOLUTION
    return 2 * m["num_hidden_layers"] * m["num_key_value_heads"] * m["head_dim"] * bytes_per_value
    # END SOLUTION

# %%
try:
    KVT = kv_per_token(m8)
except NotImplementedError:
    KVT = S.kv_bytes_per_token(m8)
print(f"{KVT:,} bytes = {KVT / 1024:.0f} KiB per token; a 2,000-token conversation holds {2000 * KVT / 1e9:.2f} GB")
S.check(KVT == 147456, "147,456 bytes of cache per token in 16 bits")

# %% [markdown]
# ## 2 · Memory left for the cache after weights and reserve (5 min)
#
# The engine may use `gpu_memory_utilization` of the 24 GB. From that it subtracts the weights and its reserve.
#
# ✏️ **Exercise** · `cache_memory(cfg)` returns the bytes left for the cache. Use 1.5 GiB for the reserve.

# %%
def cache_memory(cfg, reserve_gib=1.5):
    # BEGIN SOLUTION
    total = l4["memory_gb"] * 1e9 * cfg.gpu_memory_utilization
    return total - S.weight_bytes(m8, cfg.weight_dtype) - reserve_gib * 2**30
    # END SOLUTION

# %%
try:
    FREE = cache_memory(cfg)
except NotImplementedError:
    FREE = S.reference("d1_2")["cache_bytes"]
print(f"cache memory {FREE / 1e9:.2f} GB = {FREE / KVT:,.0f} tokens")
S.check(3.0e9 < FREE < 4.5e9, "between 3 and 4.5 GB is left for the cache")

# %% [markdown]
# ## 3 · Predict the concurrency limit at three context lengths (5 min)
#
# ✏️ **Exercise** · `predict_limits(free, per_token)` returns `{2000: n, 4000: n, 8000: n}`: how many sequences of
# that many tokens (prompt and answer together) fit at once.

# %%
def predict_limits(free, per_token, lengths=(2000, 4000, 8000)):
    # BEGIN SOLUTION
    return {L: free / (L * per_token) for L in lengths}
    # END SOLUTION

# %%
try:
    PRED = predict_limits(FREE, KVT)
except NotImplementedError:
    PRED = {int(k): v for k, v in S.reference("d1_2")["predicted"].items()}
print({L: round(v, 1) for L, v in PRED.items()})

# %% [markdown]
# ## 4 · Read the engine's blocks and run it to saturation (6 min)
#
# The engine reports its cache at start in blocks of 16 tokens. Then 40 users send requests of a fixed length
# (prompt `L - 200` tokens, answer 200 tokens) so that the cache, not the sequence cap of 64, is the limit.
# `S.max_running(run)` is the largest number of sequences the engine held at once.

# %%
print(S.start_log(cfg))
rows = []
for L in (2000, 4000, 8000):
    pool = S.synthetic_requests(200, L - 200, 200)
    run = S.run_closed(cfg, 40, pool=pool, n_requests=S.pick(160, 90), seed=1)
    s = run.summary()
    rows.append({"tokens": L, "predicted": round(PRED[L], 1), "measured_max_running": S.max_running(run),
                 "preemptions": s["preemptions"], "tpot_p90_ms": round(s["tpot_p90"] * 1000, 1)})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
S.save_rows(rows, "D1_2", "sizing_check")

# %% [markdown]
# ## 5 · Compare, explain the gap, check the 25% band (5 min)
#
# ✏️ **Exercise** · Add a column `error` = (predicted - measured) / measured, and say in a comment why the engine
# can differ from the arithmetic: blocks are whole, and every running sequence keeps growing.

# %%
try:
    # BEGIN SOLUTION
    table["error"] = (table.predicted - table.measured_max_running) / table.measured_max_running
    # the engine counts whole blocks of 16 tokens and admits a sequence at its prompt length (L - 200), so it
    # can hold one more than the arithmetic for a moment; as the answers grow, sequences need new blocks and
    # the newest is preempted (the preemptions column), so the number it keeps settles at the prediction
    # END SOLUTION
except NotImplementedError:
    table["error"] = (table.predicted - table.measured_max_running) / table.measured_max_running
print(table.to_string(index=False))
S.check(all(abs(e) <= 0.25 for e in table.error), "every prediction within 25% of the measured run")

# %% [markdown]
# 🔍 **Look at** the 8,000-token row: three conversations at once. Musamir's peak brings about two new requests
# a second. What does that say about the 16-bit deployment, before any queueing theory?
#
# ✱ **Stretch** · Predict the three limits again with 8-bit weights (`weight_dtype="fp8"`), then check with
# `S.start_log(cfg.but(weight_dtype="fp8"))`.

# %%
print(S.start_log(cfg.but(weight_dtype="fp8")).splitlines()[2])

# %% [markdown]
# ## Check yourself
#
# 1. Why does grouped-query attention matter for serving?
#    <details><summary>Answer</summary>The cache stores only the key and value heads: 8 instead of 32 here, a quarter of the memory per token, so four times the conversations.</details>
# 2. You forgot the reserve. By how much is your prediction off?
#    <details><summary>Answer</summary>About 1.6 GB of 3.6 GB: the prediction is some 45% too high, outside the band. The engine log says what it reserved.</details>
# 3. Which grows the cache of a running conversation: the prompt or the answer?
#    <details><summary>Answer</summary>Both: every token of the answer adds its keys and values, which is why a full cache preempts during decoding, not only at admission.</details>
#
# **Carry forward** · Your sizing sheet now has the concurrency limit; the capstone's cost model starts from it.
