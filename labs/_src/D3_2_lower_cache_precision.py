# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 3.2 · Lower cache precision
#
# Day 3 · Session 2 (Cache precision buys concurrency) · 25 min.
#
# With 8-bit weights Musamir serves most of the peak but still misses the objective. The cache is the next
# thing to shrink: an 8-bit cache stores each key and value in one byte instead of two. You measure the blocks,
# the concurrency and the goodput gained, then the accuracy lost on long contexts, where the cache matters
# most, as a paired difference with an interval in each language.
#
# **AI tutor allowed**: it may explain the bootstrap; it does not take the decision.
#
# **Starts from** the 8-bit-weight row of Lab 3.1 (`tuned_day2` with `weight_dtype="fp8"`), the first minutes of
# the evening peak, and the recorded outcomes of `w_fp8` (8-bit weights, 16-bit cache) and `kv_fp8` (8-bit
# weights, 8-bit cache).
#
# **Tasks**
# - [ ] 1 · Blocks at 16-bit and 8-bit cache (4 min)
# - [ ] 2 · Concurrency and goodput gained (6 min)
# - [ ] 3 · Long-context accuracy by language (7 min)
# - [ ] 4 · Paired difference with an interval (5 min)
# - [ ] 5 · Decide, with the floor (3 min)
# - [ ] ✱ Stretch · Keep the 16-bit cache only for long-context requests

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

OBJ, FLOORS = S.objectives(), S.floors()
M = S.models()["qwen3-8b"]
w8 = S.Config.load("tuned_day2").but(weight_dtype="fp8", max_num_seqs=32)
start, _ = S.peak_window()
peak = S.read_trace(start=start, end=start + S.pick(300, 120))
items = pd.DataFrame(S.eval_items()).set_index("item")
OUT = S.eval_outcomes()
print(items.groupby(["lang", "context"]).size().unstack())

# %% [markdown]
# ## 1 · Blocks at 16-bit and 8-bit cache (4 min)
#
# ✏️ **Exercise** · Compute the bytes per token of cache for each format by hand
# (2 × layers × key-value heads × head size × bytes), check it against `S.kv_bytes_per_token`, and read the blocks
# from `S.cache_blocks`.

# %%
rows = []
for kv in ("bf16", "fp8"):
    try:
        # BEGIN SOLUTION
        by_hand = 2 * M["num_hidden_layers"] * M["num_key_value_heads"] * M["head_dim"] * S.BYTES[kv]
        # END SOLUTION
    except NotImplementedError:
        by_hand = S.kv_bytes_per_token(M, kv)
    rows.append({"cache": kv, "bytes_per_token": int(by_hand), "engine": int(S.kv_bytes_per_token(M, kv)),
                 "blocks": S.cache_blocks(w8.but(kv_cache_dtype=kv))})
blocks = pd.DataFrame(rows).set_index("cache")
print(blocks.to_string())
S.check((blocks.bytes_per_token == blocks.engine).all(), "the hand count matches the engine")
S.check(blocks.loc["fp8", "blocks"] == 2 * blocks.loc["bf16", "blocks"], "an 8-bit cache doubles the blocks")

# %% [markdown]
# ## 2 · Concurrency and goodput gained (6 min)
#
# ✏️ **Exercise** · Run the peak with each cache format and keep goodput, attainment, p90 TTFT, and the engine's
# mean `running`, `waiting` and `kv_cache_usage` (`S.engine_signature(run)`).

# %%
gain = []
for kv in ("bf16", "fp8"):
    try:
        # BEGIN SOLUTION
        run = S.run_open(w8.but(kv_cache_dtype=kv), peak, seed=1)
        s, g = run.summary(), S.engine_signature(run)
        gain.append({"cache": kv, "running": g["running"], "waiting": g["waiting"], "kv_cache_usage": g["kv_cache_usage"],
                     "goodput_rps": s["goodput_rps"], "attainment": s["attainment"], "ttft_p90": s["ttft_p90"]})
        # END SOLUTION
    except NotImplementedError:
        gain = S.reference("d3_2")["gain"]
        break
gain = pd.DataFrame(gain).set_index("cache")
print(gain.to_string())
S.check(gain.loc["fp8", "attainment"] >= OBJ["attainment_target"] > gain.loc["bf16", "attainment"],
        "with an 8-bit cache the peak meets the objective")

# %% [markdown]
# 🔍 **Look at** `kv_cache_usage` with the 16-bit cache: a third of the blocks in use, yet requests wait. At this
# load room was not the limit: each decode step reads every running sequence's cache, about as many bytes as the
# weights. Halving those bytes shortens every step, and the queue drains. The doubled room matters for the
# longer contexts and the bursts.
#
# ## 3 · Long-context accuracy by language (7 min)
#
# The cache holds every earlier token; a coarser cache blurs what the model reads back from far away. Long
# items (`context == "long"`) carry an event-terms document of 6,000 to 10,000 tokens.
#
# ✏️ **Exercise** · Score `w_fp8` and `kv_fp8` per language and context (share correct).

# %%
items["w_fp8"] = [OUT["w_fp8"][i] for i in items.index]
items["kv_fp8"] = [OUT["kv_fp8"][i] for i in items.index]
try:
    # BEGIN SOLUTION
    by = items.groupby(["lang", "context"])[["w_fp8", "kv_fp8"]].mean().round(3)
    # END SOLUTION
except NotImplementedError:
    by = items.groupby(["lang", "context"])[["w_fp8", "kv_fp8"]].mean().round(3)
by["change"] = (by.kv_fp8 - by.w_fp8).round(3)
print(by.to_string())

# %% [markdown]
# ## 4 · Paired difference with an interval (5 min)
#
# The same items were answered with both caches, so compare item by item: `S.paired_diff(a, b)` returns the mean
# of `b - a` and a bootstrap 95% interval. Two separate intervals would overlap and hide a real loss.
#
# ✏️ **Exercise** · The paired difference on the long items of each language.

# %%
diffs = {}
for lang in ("ar", "en"):
    d = items[(items.lang == lang) & (items.context == "long")]
    try:
        # BEGIN SOLUTION
        diffs[lang] = S.paired_diff(list(d.w_fp8), list(d.kv_fp8))
        # END SOLUTION
    except NotImplementedError:
        diffs[lang] = S.paired_diff(list(d.w_fp8), list(d.kv_fp8))
    print(lang, "long items:", len(d), "change", diffs[lang][0], "95% interval", diffs[lang][1])
S.check(diffs["ar"][1][1] < 0, "the Arabic long-context loss is real: the whole interval is below zero")
for lang in ("ar", "en"):
    d = items[(items.lang == lang) & (items.context == "long")]
    print(lang, "separate Wilson intervals:", S.wilson(d.w_fp8.sum(), len(d)), S.wilson(d.kv_fp8.sum(), len(d)))

# %% [markdown]
# 🔍 **Look at** the separate intervals: they overlap, and a reader would call the loss noise. The paired
# interval does not include zero.
#
# ## 5 · Decide, with the floor (3 min)
#
# ✏️ **Exercise** · Compute each language's overall score with the 8-bit cache, compare with the floor, and write
# `decision` in two sentences: what you would serve and what the Director must accept or refuse.

# %%
overall = items.groupby("lang").kv_fp8.mean().round(3).to_dict()
print("overall with 8-bit cache:", overall, "floors:", {k: FLOORS[k] for k in ("ar", "en")})
try:
    # BEGIN SOLUTION
    decision = ("The 8-bit cache passes both floors overall (Arabic 0.740 against 0.73) and lets one card meet the objective at the peak. "
                "It loses about five points on long Arabic documents, with an interval below zero: Hanadi must accept that loss, "
                "or the long-document requests keep a 16-bit cache.")
    # END SOLUTION
except NotImplementedError:
    decision = "(write your decision)"
print(decision)
S.save_rows([{"variant": "weights fp8, cache fp8", "weights_gb": round(S.weight_bytes(M, "fp8") / 1e9, 1),
              "cache_blocks": int(blocks.loc["fp8", "blocks"]), "goodput_rps": gain.loc["fp8", "goodput_rps"],
              "attainment": gain.loc["fp8", "attainment"], "ar": overall["ar"], "en": overall["en"],
              "passes_floors": overall["ar"] >= FLOORS["ar"] and overall["en"] >= FLOORS["en"],
              "long_ar_change": diffs["ar"][0], "long_ar_low": diffs["ar"][1][0], "long_ar_high": diffs["ar"][1][1]}],
            "D3_2", "serving_row")

# %% [markdown]
# ✱ **Stretch** · Serve long-context requests from a pool with the 16-bit cache and the rest with the 8-bit one.
# What are the scores per language? (Use `w_fp8` outcomes for long items, `kv_fp8` for short ones.) What does
# the second pool cost in cards?

# %%
mix = [OUT["w_fp8"][i] if c == "long" else OUT["kv_fp8"][i] for i, c in zip(items.index, items.context)]
print(items.assign(mix=mix).groupby("lang").mix.mean().round(3).to_dict())

# %% [markdown]
# ## Check yourself
#
# 1. The 16-bit cache was only a third full, yet the 8-bit cache raised goodput. How?
#    <details><summary>Answer</summary>Decode steps read every running sequence's cache; at 20 sequences of a few thousand tokens that is as many bytes as the weights, so halving it shortens every step.</details>
# 2. Why compare the two caches item by item rather than with two intervals?
#    <details><summary>Answer</summary>Most items are answered the same way by both; the paired difference removes that shared variation and shows a loss that separate intervals hide.</details>
# 3. The 8-bit cache passes the floors. Why still report the long-context loss?
#    <details><summary>Answer</summary>The floor is an average over the set; long Arabic documents are a third of the Arabic items and the requests where a wrong answer about event terms costs most.</details>
#
# **Carry forward** · `artifacts/D3_2/serving_row.csv`: the fourth row of the serving table.
