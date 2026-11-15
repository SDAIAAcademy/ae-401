# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 3.3 · Acceptance rate at two loads
#
# Day 3 · Session 3 (Speculative decoding) · 25 min.
#
# Kenan read that speculative decoding makes answers two to three times faster. A small draft model (Qwen3-0.6B)
# proposes a few tokens, Musamir's base model checks them all in one step, and keeps the longest correct run plus
# one token of its own. You predict the tokens per step from the draft's acceptance rate, measure it per
# language, and measure the time per output token with and without speculation at a quiet hour and at the peak.
#
# **AI tutor allowed**: it may explain the algorithm; it does not decide when to switch it on.
#
# **Starts from** the engine chosen in Lab 3.2 (8-bit weights and cache) and `data/spec_acceptance.json`
# (synthetic per-token acceptance of the draft, by language).
#
# **Tasks**
# - [ ] 1 · Expected tokens per step from acceptance (5 min)
# - [ ] 2 · Acceptance rate by language (5 min)
# - [ ] 3 · TPOT at low load, on and off (5 min)
# - [ ] 4 · TPOT at high load, on and off (5 min)
# - [ ] 5 · Check the outputs are identical and decide (5 min)
# - [ ] ✱ Stretch · Find the best draft length at each load

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

OBJ = S.objectives()
ACC = S.load_json("spec_acceptance.json")["qwen3-0.6b"]
K = 3
off = S.Config.load("tuned_day2").but(weight_dtype="fp8", kv_cache_dtype="fp8")
on = off.but(speculative_model="qwen3-0.6b", num_speculative_tokens=K)
pool = S.peak_requests(seed=0)
LOW, HIGH = 4, 64
print("per-token acceptance:", {k: v for k, v in ACC.items() if k != "note"})

# %% [markdown]
# ## 1 · Expected tokens per step from acceptance (5 min)
#
# If each drafted token is accepted with probability `a`, independently, and the draft proposes `k`, the base
# model emits on average `(1 - a**(k + 1)) / (1 - a)` tokens per step: the accepted run plus its own token.
#
# ✏️ **Exercise** · `tokens_per_step(a, k)`, then the value for Arabic and English at `k = 3`.

# %%
def tokens_per_step(a, k):
    # BEGIN SOLUTION
    return (1 - a ** (k + 1)) / (1 - a)
    # END SOLUTION


try:
    predicted = {lang: round(tokens_per_step(ACC[lang], K), 2) for lang in ("ar", "en")}
except NotImplementedError:
    predicted = {lang: round((1 - ACC[lang] ** (K + 1)) / (1 - ACC[lang]), 2) for lang in ("ar", "en")}
print("predicted tokens per step:", predicted)
S.check(predicted["en"] > predicted["ar"] > 1.5, "English gains more than Arabic; both more than one token a step")

# %% [markdown]
# ## 2 · Acceptance rate by language (5 min)
#
# Each finished request carries `spec_accepted` (drafted tokens kept) and `spec_proposed` (drafted tokens
# checked). One step proposes `k`, so the steps are `spec_proposed / k`.
#
# ✏️ **Exercise** · From a run at low load, the measured tokens per step per language:
# `1 + accepted / steps`.

# %%
run_on_low = S.run_closed(on.but(max_num_seqs=8), LOW, pool=pool, n_requests=S.pick(120, 60), seed=1)
try:
    # BEGIN SOLUTION
    measured = {}
    for lang in ("ar", "en"):
        rs = [r for r in run_on_low.all if r.lang == lang]
        steps = sum(r.spec_proposed for r in rs) / K
        measured[lang] = round(1 + sum(r.spec_accepted for r in rs) / steps, 2)
    # END SOLUTION
except NotImplementedError:
    measured = S.reference("d3_3")["measured"]
print("measured tokens per step:", measured, "predicted:", predicted)
S.check(all(abs(measured[l] - predicted[l]) < 0.25 for l in ("ar", "en")), "the measurement agrees with the formula")

# %% [markdown]
# ## 3 · TPOT at low load, on and off (5 min)
#
# ✏️ **Exercise** · `compare(users)` runs the same load with speculation off and on and returns one row each:
# p50 and p90 TPOT, p90 TTFT, goodput, attainment and cache blocks.

# %%
def compare(users, k=K):
    rows = []
    for name, cfg in (("off", off), ("on", off.but(speculative_model="qwen3-0.6b", num_speculative_tokens=k))):
        # BEGIN SOLUTION
        cfg = cfg.but(max_num_seqs=max(8, users))
        s = S.run_closed(cfg, users, pool=pool, n_requests=S.pick(max(120, users * 5), max(60, users * 3)), seed=1).summary()
        rows.append({"users": users, "speculation": name, "tpot_p50": s["tpot_p50"], "tpot_p90": s["tpot_p90"],
                     "ttft_p90": s["ttft_p90"], "goodput_rps": s["goodput_rps"], "attainment": s["attainment"],
                     "cache_blocks": S.cache_blocks(cfg)})
        # END SOLUTION
    return rows


try:
    low = pd.DataFrame(compare(LOW))
except NotImplementedError:
    low = pd.DataFrame([r for r in S.reference("d3_3")["rows"] if r["users"] == LOW])
print(low.to_string(index=False))
S.check(low.tpot_p50.iloc[1] < 0.9 * low.tpot_p50.iloc[0], "at low load speculation cuts TPOT by more than 10%")

# %% [markdown]
# ## 4 · TPOT at high load, on and off (5 min)

# %%
try:
    high = pd.DataFrame(compare(HIGH))
except NotImplementedError:
    high = pd.DataFrame([r for r in S.reference("d3_3")["rows"] if r["users"] == HIGH])
both = pd.concat([low, high])
print(both.to_string(index=False))
S.save_rows(both.to_dict("records"), "D3_3", "speculation")
S.check(high.attainment.iloc[1] < high.attainment.iloc[0], "at the peak speculation lowers attainment")

# %% [markdown]
# 🔍 **Look at** `cache_blocks` and `ttft_p90` at 64 users. The draft model has weights and a cache of its own:
# half the blocks are gone, so requests wait to be admitted. Each verification step also does `k + 1` tokens of
# work per sequence, so the gain in TPOT shrinks as the batch grows.
#
# ## 5 · Check the outputs are identical and decide (5 min)
#
# With greedy decoding, speculation keeps only tokens the base model would have chosen, so the answers do not
# change. The recorded outcomes of `spec_draft` let you check that on the evaluation set.
#
# ✏️ **Exercise** · Count the items where `spec_draft` and `base_bf16` differ, then set `rule`: the condition under
# which Musamir's engine should run with speculation.

# %%
OUT = S.eval_outcomes()
try:
    # BEGIN SOLUTION
    differ = sum(OUT["spec_draft"][i] != OUT["base_bf16"][i] for i in OUT["base_bf16"])
    rule = ("Speculation on for the quiet hours (a few visitors at a time), where it cuts TPOT by about 15%; "
            "off at the evening peak, where the draft's memory costs more admissions than it saves in decode time.")
    # END SOLUTION
except NotImplementedError:
    differ, rule = sum(OUT["spec_draft"][i] != OUT["base_bf16"][i] for i in OUT["base_bf16"]), "(write your rule)"
print("items that differ:", differ)
print("rule:", rule)
S.check(differ == 0, "speculation does not change a single recorded answer")

# %% [markdown]
# ✱ **Stretch** · Try `k` from 1 to 5 at both loads. Which draft length gives the lowest TPOT at 4 users, and
# which the highest goodput at 64?

# %%
try:
    for k in (1, 2, 5):
        for users in (LOW, HIGH):
            r = compare(users, k)[1]
            print("k", k, "users", users, "tpot_p50", r["tpot_p50"], "goodput", r["goodput_rps"])
except NotImplementedError:
    print("write compare() in task 3 first")

# %% [markdown]
# ## Check yourself
#
# 1. The vendor promises "two to three times faster". What does the formula say for Arabic at `k = 3`?
#    <details><summary>Answer</summary>About 1.8 tokens per step before the draft's own cost; the measured TPOT gain at low load is about 15%, because the draft takes time too.</details>
# 2. Why does speculation help most when the engine is quiet?
#    <details><summary>Answer</summary>A small batch leaves the step waiting on memory; checking four tokens costs almost the same as one. A large batch already uses that time, and the draft's memory displaces sequences.</details>
# 3. How do you know speculation did not change the answers?
#    <details><summary>Answer</summary>With greedy decoding it is exact by construction; the recorded outcomes agree on all 600 items, which checks the deployment, not only the theory.</details>
#
# **Carry forward** · `artifacts/D3_3/speculation.csv` and your rule: the capstone's scaling policy may switch
# speculation by hour.
