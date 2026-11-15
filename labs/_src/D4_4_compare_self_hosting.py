# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 4.4 · Self-hosting or a hosted API
#
# Day 4 · Session 4 (Cost per successful request) · 25 min.
#
# Hanadi has a quote from a hosted API for an 8B-class model and asks one question: at what volume does running
# Musamir on Liwan's own cards cost less per visitor served? You compute the cost per successful request of the
# self-hosted node from the day's replay, compare it with the quote at three volumes, find the break-even, add
# the energy per thousand requests, and plan your capstone.
#
# **AI tutor allowed**: it may help with the plot and the plan's wording; it does not take the decision.
#
# **Starts from** `data/prices.json` (case figures in SAR, fictional), `data/daily_demand.csv`, the trace's
# token counts, and the scaling policy of Lab 4.3 (request rate at 50% of capacity, at least one replica).
#
# **Tasks**
# - [ ] 1 · Cost per successful request on the node (6 min)
# - [ ] 2 · The hosted API at three volumes (5 min)
# - [ ] 3 · The break-even volume (4 min)
# - [ ] 4 · Energy per thousand requests (4 min)
# - [ ] 5 · Plan the capstone (6 min)
# - [ ] ✱ Stretch · How sensitive the break-even is to the lease price

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
import scaling_sim as Z

P = S.load_json("prices.json")
print(P["note"])
CAPACITY = 2.5                                    # good requests per second per replica (Lab 4.3)
usual = Z.load_demand()
trace = S.read_trace(limit=40_000)
mean_in = sum(S.prompt_tokens(r, "stable_first") for r in trace) / len(trace)
mean_out = sum(r.out_tokens for r in trace) / len(trace)
print(f"a mean request: {mean_in:.0f} prompt tokens, {mean_out:.0f} answer tokens")


def replay(scale, lease=P["l4_hour"]):
    demand = [round(scale * x) for x in usual]
    _, s = Z.simulate(demand, Z.rate_policy(0.5), CAPACITY, min_replicas=1, max_replicas=64, cold=8)
    s["sar"] = round(s["replica_hours"] * (lease + P["node_overhead_hour"]), 1)
    return s

# %% [markdown]
# ## 1 · Cost per successful request on the node (6 min)
#
# The node is paid by the replica-hour (`l4_hour` plus `node_overhead_hour`), whether it serves or idles. Only the
# requests that met the objective count as served.
#
# ✏️ **Exercise** · `per_success(summary)`: SAR per successful request for one replayed day. Compute it for a usual
# day (40,000 requests).

# %%
def per_success(s):
    # BEGIN SOLUTION
    return s["sar"] / s["met"]
    # END SOLUTION


day = replay(1)
try:
    node_usual = per_success(day)
except NotImplementedError:
    node_usual = day["sar"] / day["met"]
print(day, f"-> {node_usual:.5f} SAR per successful request")

# %% [markdown]
# ## 2 · The hosted API at three volumes (5 min)
#
# The quote is per million input and output tokens. A hosted API's own latency and failures are not in the quote;
# assume here that it meets the objective for every request.
#
# ✏️ **Exercise** · `hosted_per_request()` from the quote and the mean request; then a table for 10,000, 40,000 and
# 160,000 requests a day: node SAR per day, node SAR per successful request, hosted SAR per day and per request.

# %%
def hosted_per_request():
    # BEGIN SOLUTION
    return mean_in * P["hosted_input_per_million"] / 1e6 + mean_out * P["hosted_output_per_million"] / 1e6
    # END SOLUTION


try:
    hosted = hosted_per_request()
except NotImplementedError:
    hosted = mean_in * P["hosted_input_per_million"] / 1e6 + mean_out * P["hosted_output_per_million"] / 1e6
rows = []
for scale in (0.25, 1, 4):
    s = replay(scale)
    rows.append({"requests_per_day": s["requests"], "replica_hours": s["replica_hours"], "node_sar_day": s["sar"],
                 "node_per_success": round(s["sar"] / s["met"], 5), "hosted_sar_day": round(hosted * s["requests"], 1),
                 "hosted_per_request": round(hosted, 5)})
volumes = pd.DataFrame(rows)
print(volumes.to_string(index=False))
S.save_rows(volumes.to_dict("records"), "D4_4", "volumes")
S.check(volumes.node_per_success.iloc[1] > volumes.hosted_per_request.iloc[1], "at Liwan's usual volume the hosted quote is cheaper per request")

# %% [markdown]
# 🔍 **Look at** the replica-hours column: at 10,000 and 40,000 requests a day the node is mostly one card idling
# at night. The cost per request falls with volume because that card is shared by more visitors.
#
# ## 3 · The break-even volume (4 min)
#
# ✏️ **Exercise** · Find the smallest multiple of the usual day (in steps of 0.5) at which the node's cost per
# successful request is at or below the hosted price, and plot both against requests per day.

# %%
scales = [0.5 * i for i in range(1, 21)]
curve = [{"scale": k, **replay(k)} for k in scales]
for c in curve:
    c["node_per_success"] = c["sar"] / c["met"]
try:
    # BEGIN SOLUTION
    even = next(c for c in curve if c["node_per_success"] <= hosted)
    # END SOLUTION
except NotImplementedError:
    even = next(c for c in curve if c["node_per_success"] <= hosted)
print(f"break-even near {even['requests']:,} requests a day ({even['scale']} times a usual day)")
plt = S.plt()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot([c["requests"] for c in curve], [1000 * c["node_per_success"] for c in curve], "o-", label="node, per successful request")
ax.axhline(1000 * hosted, color="C3", ls="--", label="hosted quote, per request")
ax.set(xlabel="requests per day", ylabel="SAR per 1,000 requests", title="Self-hosting against the hosted quote")
ax.legend()
S.show(fig, "break_even", "D4_4")
S.check(even["scale"] > 2, "self-hosting pays only well above the usual volume")

# %% [markdown]
# ## 4 · Energy per thousand requests (4 min)
#
# A replica draws the card (72 W at most) plus its share of the host (`host_power_w`), times the data centre's
# `pue`. Using the card's maximum is an upper bound: an idle card draws less.
#
# ✏️ **Exercise** · Watt-hours per thousand successful requests at the three volumes of task 2.

# %%
card_w = S.accelerators()["L4"]["tdp_w"]
energy = []
for scale in (0.25, 1, 4):
    s = replay(scale)
    try:
        # BEGIN SOLUTION
        wh = s["replica_hours"] * (card_w + P["host_power_w"]) * P["pue"]
        energy.append({"requests_per_day": s["requests"], "kwh_day": round(wh / 1000, 1), "wh_per_1000": round(1000 * wh / s["met"], 1)})
        # END SOLUTION
    except NotImplementedError:
        wh = s["replica_hours"] * (card_w + P["host_power_w"]) * P["pue"]
        energy.append({"requests_per_day": s["requests"], "kwh_day": round(wh / 1000, 1), "wh_per_1000": round(1000 * wh / s["met"], 1)})
energy = pd.DataFrame(energy)
print(energy.to_string(index=False))
S.check(energy.wh_per_1000.iloc[0] > 3 * energy.wh_per_1000.iloc[2], "idle cards dominate the energy at low volume")

# %% [markdown]
# ## 5 · Plan the capstone (6 min)
#
# The capstone gives you `data/deployments/capstone_start.json`: the deployment as it was before this week. You
# tune it to meet the objective at minimum cost per successful request, and defend it in 12 minutes plus 8 of
# questions. Fill the plan; it is written to `artifacts/D4_4/capstone_plan.md`.

# %%
plan = {
    "first_measurement": "The baseline's goodput curve at the evening peak, three runs a point, raw rows kept",
    "changes_in_order": "Cap and chunked prefill; prompt layout and salt; 8-bit weights; 8-bit cache; routing to the mini model",
    "quality_checks": "Each change scored per language against the floors, paired differences for the cache",
    "topology": "Replicas behind a router by conversation with a load limit; no disaggregation under the chat objective",
    "cost_answer": f"Cost per successful request against the hosted quote; break-even near {even['requests']:,} requests a day",
    "open_question_for_the_director": "Whether data may be processed outside Liwan, which the hosted quote assumes",
}
md = "# Capstone plan\n\n" + "\n".join(f"- **{k.replace('_', ' ').capitalize()}.** {v}" for k, v in plan.items()) + "\n"
(S.artifact_dir("D4_4") / "capstone_plan.md").write_text(md, encoding="utf-8")
print(md)

# %% [markdown]
# ✱ **Stretch** · Change the lease price by ±30% and find the break-even again. Which input moves the decision most:
# the lease, the volume or the capacity of a replica?

# %%
for lease in (P["l4_hour"] * 0.7, P["l4_hour"] * 1.3):
    c = next((k for k in scales if (lambda s: s["sar"] / s["met"])(replay(k, lease)) <= hosted), None)
    print(f"lease {lease:.2f} SAR/h: " + (f"break-even at {c} times a usual day" if c else "no break-even up to 10 times a usual day"))

# %% [markdown]
# ## Check yourself
#
# 1. Busy all the time, one replica serves 2.5 requests a second for about 0.0005 SAR each, half the quote. Why does a usual day cost 0.0027?
#    <details><summary>Answer</summary>Over a real day the minimum replica idles at night and replicas scale ahead of demand at 50% of capacity; the card is paid by the hour, not by the request.</details>
# 2. What would make self-hosting the right answer at Liwan's usual volume?
#    <details><summary>Answer</summary>A requirement the quote does not price, such as where the data is processed, or a much cheaper lease; the decision is the Director's, with this table.</details>
# 3. Why divide by successful requests, not all requests?
#    <details><summary>Answer</summary>A request that missed the objective was paid for and did not serve the visitor; counting it would reward overload.</details>
#
# **Carry forward** · `artifacts/D4_4/capstone_plan.md`: the plan you start the capstone from.
