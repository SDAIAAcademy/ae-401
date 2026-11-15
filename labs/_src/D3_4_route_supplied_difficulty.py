# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 3.4 · Route by difficulty
#
# Day 3 · Session 4 (Distilled models and routing to them) · 25 min.
#
# Liwan's data team distilled Musamir's base into a 1.7-billion-parameter model, Musamir-mini, on the
# authority's own traffic. It is much cheaper per token and clearly worse on its own. A supplied classifier
# scores each request's difficulty from 0 (easy) to 1 (hard). You send the requests under a threshold to the
# mini model and the rest to the base, sweep the threshold for quality and cost, choose one that holds both
# floors, and draft the one-page serving decision for Hanadi.
#
# **AI tutor allowed**: it may help with the plot and the wording of the page; it does not choose the threshold.
#
# **Starts from** `data/eval/items.csv` (with `classifier_score`), the recorded outcomes of `kv_fp8` (the base as
# chosen in Lab 3.2) and `mini`, and `data/prices.json` (case figures in SAR).
#
# **Tasks**
# - [ ] 1 · Quality of each model by language (5 min)
# - [ ] 2 · Sweep the threshold: quality against share routed (7 min)
# - [ ] 3 · Cost per million tokens of the mix (5 min)
# - [ ] 4 · Pick the threshold that holds both floors (5 min)
# - [ ] 5 · Draft the one-page serving decision (3 min)
# - [ ] ✱ Stretch · A cascade that escalates on low confidence

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

FLOORS, PRICES = S.floors(), S.load_json("prices.json")
OUT = S.eval_outcomes()
items = pd.DataFrame(S.eval_items()).set_index("item")
items["score"] = items.classifier_score.astype(float)
items["base"] = [OUT["kv_fp8"][i] for i in items.index]
items["mini"] = [OUT["mini"][i] for i in items.index]
print(items[["lang", "task", "context", "score", "base", "mini"]].head())

# %% [markdown]
# ## 1 · Quality of each model by language (5 min)
#
# ✏️ **Exercise** · The share correct of `base` and `mini` per language, with Wilson intervals.

# %%
rows = []
for lang, d in items.groupby("lang"):
    for model in ("base", "mini"):
        try:
            # BEGIN SOLUTION
            k = int(d[model].sum())
            rows.append({"lang": lang, "model": model, "score": round(k / len(d), 3), "interval": S.wilson(k, len(d))})
            # END SOLUTION
        except NotImplementedError:
            k = int(d[model].sum())
            rows.append({"lang": lang, "model": model, "score": round(k / len(d), 3), "interval": S.wilson(k, len(d))})
quality = pd.DataFrame(rows)
print(quality.to_string(index=False))
S.check(all(quality[quality.model == "mini"].score < [FLOORS["ar"], FLOORS["en"]]), "alone, the mini model misses both floors")

# %% [markdown]
# 🔍 **Look at** the Arabic gap: 16 points against 13 in English. Distillation on mostly English traffic shows.
#
# ## 2 · Sweep the threshold: quality against share routed (7 min)
#
# A request whose `score` is below the threshold goes to `mini`, the others to `base`.
#
# ✏️ **Exercise** · `routed(th)` returns, for one threshold, the share sent to mini and the score per language of
# the mix. Sweep thresholds 0.1 to 0.7 and plot the Arabic and English scores against the share routed, with the
# floors as lines.

# %%
THS = [round(0.1 * i, 1) for i in range(1, 8)]


def routed(th):
    # BEGIN SOLUTION
    to_mini = items.score < th
    mix = items.mini.where(to_mini, items.base)
    by = items.assign(mix=mix).groupby("lang").mix.mean()
    return {"threshold": th, "share_mini": round(to_mini.mean(), 3), "ar": round(by["ar"], 3), "en": round(by["en"], 3)}
    # END SOLUTION


def _routed_ref(th):
    to_mini = items.score < th
    by = items.assign(mix=items.mini.where(to_mini, items.base)).groupby("lang").mix.mean()
    return {"threshold": th, "share_mini": round(to_mini.mean(), 3), "ar": round(by["ar"], 3), "en": round(by["en"], 3)}


try:
    sweep = pd.DataFrame([routed(th) for th in THS])
except NotImplementedError:
    sweep = pd.DataFrame([_routed_ref(th) for th in THS])
print(sweep.to_string(index=False))
plt = S.plt()
fig, ax = plt.subplots(figsize=(6, 3.4))
for lang, c in (("ar", "C0"), ("en", "C1")):
    ax.plot(sweep.share_mini, sweep[lang], "o-", color=c, label=f"{lang} score")
    ax.axhline(FLOORS[lang], color=c, ls="--", lw=1)
ax.set(xlabel="share of requests sent to mini", ylabel="share correct", title="Quality against share routed")
ax.legend()
S.show(fig, "route_sweep", "D3_4")

# %% [markdown]
# ## 3 · Cost per million tokens of the mix (5 min)
#
# The cost of a model is the price of its card per hour divided by the output tokens it produces per hour while
# meeting the objective. The two capacities below come from the simulator at the load where each still meets it.
#
# ✏️ **Exercise** · `cost_per_million(tok_s)` in SAR per million output tokens from `PRICES` (card and node share
# per hour), then the cost of the mix for every threshold, assuming each model runs on its own cards.

# %%
pool = S.peak_requests(seed=0)
cap = {}
for model, users in (("base", 48), ("mini", 96)):
    cfg = S.Config.load("tuned_day2").but(model="qwen3-8b" if model == "base" else "musamir-mini",
                                          weight_dtype="fp8", kv_cache_dtype="fp8", max_num_seqs=users)
    s = S.run_closed(cfg, users, pool=pool, n_requests=users * 6, seed=1).summary()
    cap[model] = s["output_tok_s"]
    print(model, users, "users:", s["output_tok_s"], "tokens/s, attainment", s["attainment"])


def cost_per_million(tok_s):
    # BEGIN SOLUTION
    return (PRICES["l4_hour"] + PRICES["node_overhead_hour"]) / (tok_s * 3600) * 1e6
    # END SOLUTION


try:
    c = {m: round(cost_per_million(cap[m]), 2) for m in cap}
except NotImplementedError:
    c = {m: round((PRICES["l4_hour"] + PRICES["node_overhead_hour"]) / (cap[m] * 3600) * 1e6, 2) for m in cap}
sweep["sar_per_million"] = (sweep.share_mini * c["mini"] + (1 - sweep.share_mini) * c["base"]).round(2)
print("SAR per million output tokens:", c)
print(sweep.to_string(index=False))
S.check(c["mini"] < 0.5 * c["base"], "the mini model costs less than half as much per token")

# %% [markdown]
# ## 4 · Pick the threshold that holds both floors (5 min)
#
# ✏️ **Exercise** · Set `choice` to the threshold you would serve and `why` in one sentence. Hanadi's rule is
# both floors; think about how far above the floor the Arabic score is, given the intervals of task 1.

# %%
holds = sweep[(sweep.ar >= FLOORS["ar"]) & (sweep.en >= FLOORS["en"])]
print(holds.to_string(index=False))
try:
    # BEGIN SOLUTION
    choice = 0.4
    why = ("0.5 holds the Arabic floor by 0.3 points, far inside the noise of 300 items; 0.4 keeps 0.7 points of margin "
           "and still sends 36% of requests to the mini model.")
    # END SOLUTION
except NotImplementedError:
    choice, why = float(holds.threshold.max()), "(write why)"
pick = sweep.set_index("threshold").loc[choice]
print("choice:", choice, dict(pick), "|", why)
S.check(choice in set(holds.threshold), "the chosen threshold holds both floors")
S.save_rows([{"variant": f"route at {choice}: mini {pick.share_mini:.0%}, base (fp8, fp8 cache)", "ar": pick.ar, "en": pick.en,
              "sar_per_million": pick.sar_per_million, "passes_floors": True}], "D3_4", "serving_row")

# %% [markdown]
# ## 5 · Draft the one-page serving decision (3 min)
#
# The page states the decision, the evidence and what the Director must accept. Fill the template; it is
# written to `artifacts/D3_4/serving_decision.md`.

# %%
page = f"""# Serving decision: Musamir at the evening peak

**Decision.** Serve the base model with 8-bit weights and an 8-bit cache, behind a router that sends requests with a
difficulty score under {choice} to Musamir-mini ({pick.share_mini:.0%} of requests). Speculative decoding off at the peak.

**Evidence.**
- 8-bit weights: Arabic {0.757:.3f}, English {0.837:.3f}, both above the floors; 4-bit weights miss the Arabic floor.
- 8-bit cache: one card meets the objective at the first minutes of the peak; long Arabic documents lose about 5 points.
- Routing at {choice}: Arabic {pick.ar:.3f}, English {pick.en:.3f}; {pick.sar_per_million:.2f} SAR per million output tokens
  against {c['base']:.2f} for the base alone.

**For the Director to accept.** The long-document loss in Arabic, and a margin over the Arabic floor of
{(pick.ar - FLOORS['ar']) * 100:.1f} points on 300 items.

**Not yet shown.** The whole day (Day 4), the replicas and the router between them, and the cost per successful request.
"""
d = S.artifact_dir("D3_4")
(d / "serving_decision.md").write_text(page, encoding="utf-8")
print(page)

# %% [markdown]
# ✱ **Stretch** · A cascade sends every request to mini first and escalates to the base when mini's confidence is
# low. Here a confidence is not recorded; use `1 - score` as a stand-in, escalate under 0.5, and count the cost: an
# escalated request pays for both models.

# %%
esc = items.score >= 0.5
cascade = items.base.where(esc, items.mini)
cost = (c["mini"] + esc.mean() * c["base"])
print(items.assign(q=cascade).groupby("lang").q.mean().round(3).to_dict(), "SAR per million:", round(cost, 2))

# %% [markdown]
# ## Check yourself
#
# 1. Why route by request rather than replace the base with the mini model?
#    <details><summary>Answer</summary>Alone the mini misses both floors; on easy requests it is as good as the base, so routing keeps the quality and moves a third of the tokens to a model that costs far less.</details>
# 2. The classifier is supplied. What would you check before trusting it in production?
#    <details><summary>Answer</summary>Its scores on new traffic in both languages: if it calls hard Arabic requests easy, the mix falls under the floor without anyone noticing.</details>
# 3. Why is the cost per million tokens not yet the cost the Director pays?
#    <details><summary>Answer</summary>It assumes each card runs at its capacity all day; Day 4 replays the daily pattern, where cards idle at night.</details>
#
# **Carry forward** · `artifacts/D3_4/serving_decision.md`: the first draft of the capstone's decision page.
