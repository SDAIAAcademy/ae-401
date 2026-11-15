# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 3.1 · Two weight precisions
#
# Day 3 · Session 1 (Quantising weights) · 25 min.
#
# Yesterday's engine meets the objective at a few visitors and is still far from the evening peak. Fewer bytes per
# weight leave more of the card to the cache and make each decode step read less. You compare Musamir's base
# model in 16-bit, 8-bit (FP8) and 4-bit weights on memory, on goodput at the peak, and on the Arabic and English
# evaluation set with intervals, then check each against the quality floors Hanadi set.
#
# **AI tutor allowed**: it may explain number formats; it does not choose the variant.
#
# **Starts from** `data/deployments/tuned_day2.json`, the first minutes of the evening peak in `data/trace_day.csv`,
# and the recorded evaluation outcomes in `data/eval/` (synthetic; 300 items per language).
#
# **Tasks**
# - [ ] 1 · Memory and cache left for each precision (5 min)
# - [ ] 2 · Goodput at the peak for each precision (6 min)
# - [ ] 3 · Arabic and English scores with intervals (7 min)
# - [ ] 4 · Check each against the floors (4 min)
# - [ ] 5 · Write the rows of the serving table (3 min)
# - [ ] ✱ Stretch · Which formats the T4 of a notebook can run

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
base = S.Config.load("tuned_day2").but(max_num_seqs=32)
VARIANTS = {"bf16": "base_bf16", "fp8": "w_fp8", "int4": "w_int4"}   # weight format -> evaluation variant
start, _ = S.peak_window()
peak = S.read_trace(start=start, end=start + S.pick(300, 120))
print(f"{len(peak)} requests in the first {S.pick(300, 120)} s of the peak: {len(peak) / S.pick(300, 120):.2f} per second")
print("floors:", FLOORS)

# %% [markdown]
# ## 1 · Memory and cache left for each precision (5 min)
#
# `S.weight_bytes(model, dtype)` gives the bytes of the weights (4-bit counts 0.53 bytes per weight: the scales
# take the rest). `S.cache_blocks(cfg)` gives the blocks the engine reports at start.
#
# ✏️ **Exercise** · For each format, fill a row: weights in GB, cache blocks, and how many conversations of 2,000
# tokens the cache holds (16 tokens a block).

# %%
mem = []
for wd in VARIANTS:
    cfg = base.but(weight_dtype=wd)
    try:
        # BEGIN SOLUTION
        blocks = S.cache_blocks(cfg)
        mem.append({"weights": wd, "weights_gb": round(S.weight_bytes(M, wd) / 1e9, 1), "cache_blocks": blocks,
                    "conversations_2000": round(blocks * 16 / 2000, 1)})
        # END SOLUTION
    except NotImplementedError:
        blocks = S.cache_blocks(cfg)
        mem.append({"weights": wd, "weights_gb": round(S.weight_bytes(M, wd) / 1e9, 1), "cache_blocks": blocks,
                    "conversations_2000": round(blocks * 16 / 2000, 1)})
mem = pd.DataFrame(mem).set_index("weights")
print(mem.to_string())
S.check(mem.loc["fp8", "cache_blocks"] > 3 * mem.loc["bf16", "cache_blocks"], "8-bit weights more than triple the cache")

# %% [markdown]
# 🔍 **Look at** the step from 16 to 8 bits: the weights shrink by 8.2 GB, and almost all of it goes to the cache.
#
# ## 2 · Goodput at the peak for each precision (6 min)
#
# Open loop: the peak's requests arrive at their own times, whatever the engine does. Sequence cap 32, the rest
# as tuned on Day 2.
#
# ✏️ **Exercise** · For each format, run `S.run_open(cfg, peak, seed=1)` and keep throughput, goodput, attainment,
# p90 TTFT and p90 TPOT.

# %%
runs = []
for wd in VARIANTS:
    try:
        # BEGIN SOLUTION
        s = S.run_open(base.but(weight_dtype=wd), peak, seed=1).summary()
        runs.append({"weights": wd, **{k: s[k] for k in ("throughput_rps", "goodput_rps", "attainment", "ttft_p90", "tpot_p90")}})
        # END SOLUTION
    except NotImplementedError:
        runs = S.reference("d3_1")["runs"]
        break
runs = pd.DataFrame(runs).set_index("weights")
print(runs.to_string())
S.check(runs.loc["bf16", "attainment"] < 0.1 and runs.loc["fp8", "goodput_rps"] > 10 * max(0.01, runs.loc["bf16", "goodput_rps"]),
        "16-bit weights drown at the peak; 8-bit weights serve most of it")

# %% [markdown]
# ## 3 · Arabic and English scores with intervals (7 min)
#
# `S.eval_outcomes()[variant][item]` is 1 when the variant answered the item correctly. Every item was answered
# by every variant, so the comparison is paired.
#
# ✏️ **Exercise** · `score(variant, lang)` returns the share correct and its 95% Wilson interval (`S.wilson(k, n)`).

# %%
items = pd.DataFrame(S.eval_items()).set_index("item")
OUT = S.eval_outcomes()


def score(variant, lang):
    # BEGIN SOLUTION
    ids = items.index[items.lang == lang]
    k = sum(OUT[variant][i] for i in ids)
    return round(k / len(ids), 3), S.wilson(k, len(ids))
    # END SOLUTION


def _score_ref(variant, lang):
    ids = items.index[items.lang == lang]
    k = sum(OUT[variant][i] for i in ids)
    return round(k / len(ids), 3), S.wilson(k, len(ids))


q = []
for wd, v in VARIANTS.items():
    for lang in ("ar", "en"):
        try:
            p, ci = score(v, lang)
        except NotImplementedError:
            p, ci = _score_ref(v, lang)
        q.append({"weights": wd, "lang": lang, "score": p, "low": ci[0], "high": ci[1]})
q = pd.DataFrame(q)
print(q.to_string(index=False))

# %% [markdown]
# ## 4 · Check each against the floors (4 min)
#
# The floor is the base model's score less three points, per language. Hanadi's rule: a variant passes when its
# score is at or above the floor **in both languages**. The interval is reported with it, not used to excuse a miss.
#
# ✏️ **Exercise** · Add a column `passes` (per row) and compute `verdict[wd]`, true when both languages pass.

# %%
try:
    # BEGIN SOLUTION
    q["passes"] = [s >= FLOORS[lang] for s, lang in zip(q.score, q.lang)]
    verdict = q.groupby("weights").passes.all().to_dict()
    # END SOLUTION
except NotImplementedError:
    q["passes"] = [s >= FLOORS[lang] for s, lang in zip(q.score, q.lang)]
    verdict = q.groupby("weights").passes.all().to_dict()
print(q.to_string(index=False))
print(verdict)
S.check(verdict == {"bf16": True, "fp8": True, "int4": False}, "4-bit weights fail the Arabic floor; 8-bit weights pass both")

# %% [markdown]
# 🔍 **Look at** the 4-bit rows: English barely moves, Arabic loses seven points. An English-only evaluation would
# have shipped it.
#
# ## 5 · Write the rows of the serving table (3 min)
#
# The serving table collects, for every variant you try this week, its memory, its goodput at the peak and its
# quality, one row each. Labs 3.2 to 3.4 add rows; the capstone starts from it.

# %%
table = []
for wd, v in VARIANTS.items():
    sc = q[q.weights == wd].set_index("lang")
    table.append({"variant": f"weights {wd}, cache bf16", "weights_gb": mem.loc[wd, "weights_gb"], "cache_blocks": int(mem.loc[wd, "cache_blocks"]),
                  "goodput_rps": runs.loc[wd, "goodput_rps"], "attainment": runs.loc[wd, "attainment"],
                  "ar": sc.loc["ar", "score"], "en": sc.loc["en", "score"], "passes_floors": verdict[wd]})
S.save_rows(table, "D3_1", "serving_table")
print(pd.DataFrame(table).to_string(index=False))

# %% [markdown]
# ✱ **Stretch** · A notebook's free T4 has 16 GB, no BF16 and no FP8 tensor cores. Which formats of Musamir's base
# fit, and how many blocks are left? (`accelerator="T4"`, `weight_dtype` in `fp16`, `int8`, `int4`.)

# %%
for wd in ("fp16", "int8", "int4"):
    cfg = base.but(accelerator="T4", weight_dtype=wd)
    print(wd, "weights", round(S.weight_bytes(M, wd) / 1e9, 1), "GB, blocks", S.cache_blocks(cfg))

# %% [markdown]
# ## Check yourself
#
# 1. Why did 8-bit weights raise goodput far more than they shortened one decode step?
#    <details><summary>Answer</summary>Most of the freed memory became cache: three times the blocks, so the engine held enough conversations to keep up with the peak instead of queueing.</details>
# 2. The 4-bit variant has the highest goodput per card. Why not ship it?
#    <details><summary>Answer</summary>It misses the Arabic floor by four points; capacity does not buy back quality the visitors notice.</details>
# 3. The 8-bit Arabic score is 0.757 with an interval from 0.705 to 0.802, and the floor is 0.73. Is the interval a problem?
#    <details><summary>Answer</summary>It says 300 items cannot separate 0.757 from the floor with confidence; report it, and add items if the decision is close.</details>
#
# **Carry forward** · `artifacts/D3_1/serving_table.csv`: the first three rows of the serving table.
