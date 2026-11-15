# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 4.1 · When disaggregation pays
#
# Day 4 · Session 1 (Disaggregated prefill and decode) · 25 min.
#
# On festival evenings half of Musamir's requests may carry an event-terms document. Yamen has two L4 cards and
# reads that splitting them into a prefill card and a decode card stops long prompts from stalling answers. You
# predict, with Day 1's arithmetic, the load on each pool, the cost of moving the cache, and the step times
# together and split; then you check the prediction against two supplied runs and state when the split pays.
#
# **AI tutor allowed**: it may check your arithmetic; it does not state the condition for you.
#
# **Starts from** two supplied runs at 32 users with 50% documents, on 8-bit weights and cache:
# `data/reference/d4_1_together.csv` (two cards, each doing both phases, chunked prefill at 1,024 tokens) and
# `d4_1_split.csv` (one prefill card, one decode card), made with `S.simulate(..., disagg=(1, 1))`.
#
# **Tasks**
# - [ ] 1 · Predict prefill and decode pool load (6 min)
# - [ ] 2 · Cache transfer time per request (4 min)
# - [ ] 3 · Predicted TTFT and TPOT, together and split (5 min)
# - [ ] 4 · Compare with the supplied run (6 min)
# - [ ] 5 · State the condition when it pays (4 min)
# - [ ] ✱ Stretch · Change the share of long prompts and predict again

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
cfg = S.Config.load("tuned_day2").but(weight_dtype="fp8", kv_cache_dtype="fp8", max_num_seqs=32)
M, ACC = S.models()["qwen3-8b"], S.accelerators()["L4"]
together = pd.read_csv(S.DATA / "reference" / "d4_1_together.csv")
split = pd.read_csv(S.DATA / "reference" / "d4_1_split.csv")
rate = len(together) / (together.arrival.max() - together.arrival.min())
computed = (together.prompt_tokens - together.hit_tokens).mean()
print(f"arrival rate {rate:.2f} requests per second; prompt {together.prompt_tokens.mean():.0f} tokens, "
      f"{computed:.0f} computed after cache hits; answer {together.out_tokens.mean():.0f} tokens")

# %% [markdown]
# ## 1 · Predict prefill and decode pool load (6 min)
#
# A prefill card does `S.flops_per_token(M)` operations per prompt token, at `peak × 0.45` (the simulator's
# efficiency, `S.CAL["compute_eff"]`) with 8-bit weights. The decode pool holds, by Little's law, about
# `rate × answer tokens × TPOT` sequences at once.
#
# ✏️ **Exercise** · `prefill_busy`: the share of time one prefill card is busy. `decoding`: the sequences the decode
# pool holds at once, with a TPOT of 70 ms.

# %%
try:
    # BEGIN SOLUTION
    ops = S.peak_ops(ACC, "fp8") * S.CAL["compute_eff"]
    prefill_busy = rate * computed * S.flops_per_token(M) / ops
    decoding = rate * together.out_tokens.mean() * 0.070
    # END SOLUTION
except NotImplementedError:
    ops = S.peak_ops(ACC, "fp8") * S.CAL["compute_eff"]
    prefill_busy, decoding = rate * computed * S.flops_per_token(M) / ops, rate * together.out_tokens.mean() * 0.070
print(f"one prefill card busy {prefill_busy:.0%} of the time; {decoding:.0f} sequences decoding at once")
S.check(prefill_busy < 0.6, "one prefill card has room: the split's prefill pool is not the bottleneck")

# %% [markdown]
# ## 2 · Cache transfer time per request (4 min)
#
# When split, the prefill card sends the prompt's cache to the decode card over the node's link (effective
# `S.CAL["link_gbs"]` GB/s here).
#
# ✏️ **Exercise** · Transfer time for the mean prompt and for a 10,000-token document.

# %%
def transfer_s(tokens):
    # BEGIN SOLUTION
    return tokens * S.kv_bytes_per_token(M, cfg.kv_cache_dtype) / (S.CAL["link_gbs"] * 1e9)
    # END SOLUTION


try:
    t_mean, t_doc = transfer_s(together.prompt_tokens.mean()), transfer_s(10_000)
except NotImplementedError:
    kvb = S.kv_bytes_per_token(M, cfg.kv_cache_dtype)
    t_mean, t_doc = together.prompt_tokens.mean() * kvb / 20e9, 10_000 * kvb / 20e9
print(f"transfer: {t_mean * 1000:.0f} ms for the mean prompt, {t_doc * 1000:.0f} ms for 10,000 tokens")
S.check(t_doc < 0.1, "moving the cache costs tens of milliseconds, small next to the TTFT objective")

# %% [markdown]
# ## 3 · Predicted TTFT and TPOT, together and split (5 min)
#
# `S.step_time(cfg, prefill_tokens, prefill_attn, decode_ctx, n_decode)` is the simulator's step model (Day 1's
# arithmetic plus overheads). Together, each card decodes half the sequences and many steps also carry a chunk
# of 1,024 prompt tokens. Split, the decode card's steps never carry a chunk.
#
# ✏️ **Exercise** · Predict the step time with and without a chunk for each layout. The longest stall is about
# the longest step; TPOT lies between the two step times, closer to the decode-only one.

# %%
ctx = together.prompt_tokens.mean() + together.out_tokens.mean() / 2
try:
    # BEGIN SOLUTION
    n_card = decoding / 2
    pred = {"together": {"decode_step": S.step_time(cfg, 0, 0, n_card * ctx, int(n_card)),
                         "chunk_step": S.step_time(cfg, 1024, 1024 * ctx, n_card * ctx, int(n_card))},
            "split": {"decode_step": S.step_time(cfg, 0, 0, decoding * ctx, int(decoding)), "chunk_step": None}}
    # END SOLUTION
except NotImplementedError:
    n_card = decoding / 2
    pred = {"together": {"decode_step": S.step_time(cfg, 0, 0, n_card * ctx, int(n_card)),
                         "chunk_step": S.step_time(cfg, 1024, 1024 * ctx, n_card * ctx, int(n_card))},
            "split": {"decode_step": S.step_time(cfg, 0, 0, decoding * ctx, int(decoding)), "chunk_step": None}}
for k, v in pred.items():
    print(k, {a: (round(b, 3) if b else b) for a, b in v.items()})
S.check(pred["together"]["chunk_step"] > 2 * pred["split"]["decode_step"], "a chunk step is more than twice a decode step")

# %% [markdown]
# ## 4 · Compare with the supplied run (6 min)
#
# ✏️ **Exercise** · For both runs, the p50 TPOT, p90 stall and p90 TTFT by kind, and the attainment.

# %%
rows = []
for name, t in (("together", together), ("split", split)):
    try:
        # BEGIN SOLUTION
        for kind, d in t.groupby("kind"):
            rows.append({"layout": name, "kind": kind, "tpot_p50": round(d.tpot_s.median(), 3), "stall_p90": round(d.stall_s.quantile(.9), 3),
                         "ttft_p90": round(d.ttft_s.quantile(.9), 2), "attainment": round(d.met.mean(), 3)})
        # END SOLUTION
    except NotImplementedError:
        for kind, d in t.groupby("kind"):
            rows.append({"layout": name, "kind": kind, "tpot_p50": round(d.tpot_s.median(), 3), "stall_p90": round(d.stall_s.quantile(.9), 3),
                         "ttft_p90": round(d.ttft_s.quantile(.9), 2), "attainment": round(d.met.mean(), 3)})
cmp_ = pd.DataFrame(rows)
print(cmp_.to_string(index=False))
print("attainment overall:", round(together.met.mean(), 3), "together,", round(split.met.mean(), 3), "split")
S.check(together.met.mean() > split.met.mean(), "under Musamir's objective, two cards together beat the split")

# %% [markdown]
# 🔍 **Look at** the stall: split, it drops from about 0.18 s to 0.13 s, as predicted. But the objective allows
# 1.0 s, so the gain buys nothing, and the single prefill card makes documents wait longer for their first token.
#
# ## 5 · State the condition when it pays (4 min)
#
# Hanadi asks whether a stricter objective would change the answer: an answer that pauses for more than 150 ms
# reads as jerky on the new voice channel.
#
# ✏️ **Exercise** · Recompute the attainment of both runs with a stall limit of 0.15 s, then write `condition`: when
# disaggregation pays for Musamir.

# %%
def attainment(t, stall):
    return round(((t.ttft_s <= OBJ["ttft_s"]) & (t.tpot_s <= OBJ["tpot_s"]) & (t.stall_s <= stall)).mean(), 3)


strict = {name: attainment(t, 0.15) for name, t in (("together", together), ("split", split))}
print("with a 0.15 s stall limit:", strict)
try:
    # BEGIN SOLUTION
    condition = ("Disaggregation pays when the stall or TPOT limit is tighter than a chunked step (about 0.17 s here), "
                 "as for a voice channel; under the chat objective of 1.0 s, chunked prefill on both cards serves more.")
    # END SOLUTION
except NotImplementedError:
    condition = "(write the condition)"
print(condition)
S.check(strict["split"] >= OBJ["attainment_target"] > strict["together"], "with a 150 ms stall limit only the split meets the objective")
S.save_rows(cmp_.to_dict("records"), "D4_1", "together_split")

# %% [markdown]
# ✱ **Stretch** · Run both layouts yourself at 20% documents (`S.mixed_pool(share_docs=0.2)`) and predict first:
# does the split gain or lose against together? (About 20 seconds per run.)

# %%
pool = S.mixed_pool(share_docs=0.2, n=400, seed=1)
for name, kw in (("together", dict(replicas=2)), ("split", dict(disagg=(1, 1)))):
    s = S.simulate(cfg, users=32, pool=pool, n_requests=S.pick(256, 128), seed=1, **kw).summary()
    print(name, {k: s[k] for k in ("goodput_rps", "attainment", "ttft_p90", "stall_p90")})

# %% [markdown]
# ## Check yourself
#
# 1. Why did splitting two cards lower attainment under Musamir's objective?
#    <details><summary>Answer</summary>Chunked prefill already kept the stall under 1 s, so the split had nothing to fix; meanwhile all prefills queued on one card and the transfer added to every TTFT.</details>
# 2. When does a separate decode pool matter?
#    <details><summary>Answer</summary>When the pause between tokens must stay under the time of a step that carries a prompt chunk: a tight TPOT or stall objective, long prompts, or many cards where pools can be sized to each phase.</details>
# 3. The cache transfer is 12 ms on average. When would it matter?
#    <details><summary>Answer</summary>Between nodes on a slower network, or for very long prompts with a 16-bit cache: the time grows with tokens × bytes per token.</details>
#
# **Carry forward** · `artifacts/D4_1/together_split.csv` and your condition: the topology section of the capstone.
