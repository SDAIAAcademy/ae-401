# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 4.3 · Scaling over the day
#
# Day 4 · Session 3 (Autoscaling and cold starts) · 25 min.
#
# The Qamra Festival's closing day is forecast at four times a usual day: 160,000 requests, quiet at night,
# a lunch rise and an evening peak of about eleven requests a second. Kenan wants replicas that follow the
# demand instead of enough cards for the peak all day. You derive one replica's capacity from its curve, replay
# the day minute by minute in the course's scaling simulator with two scaling signals, add the cold start, and
# choose the signal and the minimum number of replicas by cost and by the worst hour.
#
# **AI tutor allowed**: it may explain the simulator's code; it does not choose the policy.
#
# **Starts from** `data/daily_demand.csv` (requests per minute of a usual day) and `scaling_sim.py` (a fluid model
# of replicas, standard library only; read its first lines).
#
# **Tasks**
# - [ ] 1 · Load the daily pattern and a replica's capacity (4 min)
# - [ ] 2 · Scale on request rate (5 min)
# - [ ] 3 · Scale on queue length (5 min)
# - [ ] 4 · Cold start and minimum replicas (6 min)
# - [ ] 5 · Choose signal and minimum by cost and attainment (5 min)
# - [ ] ✱ Stretch · Scale to zero overnight and measure the first morning visitors

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

import math
import pandas as pd
import servelab as S
import scaling_sim as Z

OBJ, PRICES = S.objectives(), S.load_json("prices.json")
HOUR = PRICES["l4_hour"] + PRICES["node_overhead_hour"]
usual = Z.load_demand()
closing = [4 * x for x in usual]
print(f"closing day: {sum(closing):,} requests; busiest minute {max(closing) / 60:.1f} per second; quietest hour "
      f"{min(sum(closing[h * 60:(h + 1) * 60]) for h in range(24)) / 3600:.2f} per second")

# %% [markdown]
# ## 1 · Load the daily pattern and a replica's capacity (4 min)
#
# A replica's capacity is the goodput it sustains while meeting the objective. Day 3's engine (8-bit weights and
# cache) at 48 concurrent users is just under the knee of its curve.
#
# ✏️ **Exercise** · Measure `capacity`: the goodput at 48 users, rounded to one decimal, and check that its
# attainment meets the target. Plot the closing day's demand in requests per second against the hour.

# %%
cfg = S.Config.load("tuned_day2").but(weight_dtype="fp8", kv_cache_dtype="fp8", max_num_seqs=48)
try:
    # BEGIN SOLUTION
    s = S.run_closed(cfg, 48, n_requests=288, seed=1).summary()
    capacity = round(s["goodput_rps"], 1)
    S.check(s["attainment"] >= OBJ["attainment_target"], "48 users is still under the knee")
    # END SOLUTION
except NotImplementedError:
    capacity = 2.5
print("one replica serves", capacity, "good requests per second")
plt = S.plt()
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot([m / 60 for m in range(len(closing))], [x / 60 for x in closing], lw=0.8)
ax.axhline(capacity * 0.9, color="C3", ls="--", lw=1, label="one replica at 90%")
ax.set(xlabel="hour of the day", ylabel="requests per second", title="Closing day, minute by minute")
ax.legend()
S.show(fig, "closing_day", "D4_3")

# %% [markdown]
# ## 2 · Scale on request rate (5 min)
#
# A policy is a function of the state at the end of each minute (`arrivals`, `backlog`, `util`, `capacity`,
# `wanted`, `ready`) that returns the number of replicas wanted.
#
# ✏️ **Exercise** · `by_rate(state)`: enough replicas for last minute's arrivals at 50% of capacity. Replay the
# closing day with at least one replica and a cold start of 4 minutes.

# %%
def by_rate(s, target=0.5):
    # BEGIN SOLUTION
    return math.ceil(s["arrivals"] / 60 / (s["capacity"] * target))
    # END SOLUTION


def cost(summary):
    return round(summary["replica_hours"] * HOUR, 1)


try:
    by_rate({"arrivals": 60, "capacity": 1.0})
    rate_policy = by_rate
except NotImplementedError:
    rate_policy = Z.rate_policy(0.5)
rows_rate, rate = Z.simulate(closing, rate_policy, capacity, min_replicas=1, cold=4)
print("rate:", rate, "SAR", cost(rate))
fixed_rows, fixed = Z.simulate(closing, lambda s: 5, capacity, min_replicas=5, cold=4)
print("five replicas all day:", fixed, "SAR", cost(fixed))
S.check(rate["worst_hour"] >= OBJ["attainment_target"] and cost(rate) < 0.6 * cost(fixed),
        "scaling on rate meets the objective in every hour for well under the cost of the peak all day")

# %% [markdown]
# ## 3 · Scale on queue length (5 min)
#
# The queue is the signal visitors feel. `Z.queue_policy(up_at=20, down_after=10)` adds a replica while more than
# 20 requests wait and removes one after 10 quiet minutes.
#
# ✏️ **Exercise** · Replay the same day with the queue policy and plot the replicas paid for, by minute, for both
# policies.

# %%
try:
    # BEGIN SOLUTION
    rows_q, queue = Z.simulate(closing, Z.queue_policy(), capacity, min_replicas=1, cold=4)
    # END SOLUTION
except NotImplementedError:
    rows_q, queue = Z.simulate(closing, Z.queue_policy(), capacity, min_replicas=1, cold=4)
print("queue:", queue, "SAR", cost(queue))
fig, ax = plt.subplots(figsize=(7, 3))
for rows, name in ((rows_rate, "rate"), (rows_q, "queue")):
    ax.plot([r["minute"] / 60 for r in rows], [r["replicas_paid"] for r in rows], lw=1, label=name)
ax.set(xlabel="hour of the day", ylabel="replicas paid", title="Replicas over the closing day")
ax.legend()
S.show(fig, "replicas_by_signal", "D4_3")

# %% [markdown]
# 🔍 **Look at** the evening: the queue policy adds replicas only once requests already wait, then keeps adding
# while the first ones are still starting. It overshoots, and the overshoot costs.
#
# ## 4 · Cold start and minimum replicas (6 min)
#
# A replica needs minutes to start: the machine, the engine, the 8 GB of weights, the warm-up. Yamen measured
# 4 minutes; with weights pulled over the network it can be 8.
#
# ✏️ **Exercise** · For both signals, cold starts of 2, 4 and 8 minutes and minimums of 1 and 2, collect the
# worst hour, the replica-hours and the cost.

# %%
grid = []
for name, pol in (("rate", rate_policy), ("queue", Z.queue_policy())):
    for cold in (2, 4, 8):
        for mn in (1, 2):
            try:
                # BEGIN SOLUTION
                _, sm = Z.simulate(closing, pol, capacity, min_replicas=mn, cold=cold)
                grid.append({"signal": name, "cold_min": cold, "min_replicas": mn, "worst_hour": sm["worst_hour"],
                             "attainment": sm["attainment"], "replica_hours": sm["replica_hours"], "sar": cost(sm)})
                # END SOLUTION
            except NotImplementedError:
                _, sm = Z.simulate(closing, pol, capacity, min_replicas=mn, cold=cold)
                grid.append({"signal": name, "cold_min": cold, "min_replicas": mn, "worst_hour": sm["worst_hour"],
                             "attainment": sm["attainment"], "replica_hours": sm["replica_hours"], "sar": cost(sm)})
grid = pd.DataFrame(grid)
print(grid.to_string(index=False))
S.save_rows(grid.to_dict("records"), "D4_3", "scaling_grid")

# %% [markdown]
# ## 5 · Choose signal and minimum by cost and attainment (5 min)
#
# ✏️ **Exercise** · Among the rows whose worst hour meets the target at a cold start of 8 minutes (plan for the
# slow case), choose the cheapest, and write `policy` in one sentence.

# %%
ok = grid[(grid.cold_min == 8) & (grid.worst_hour >= OBJ["attainment_target"])]
try:
    # BEGIN SOLUTION
    best = ok.sort_values("sar").iloc[0]
    policy = (f"Scale on the request rate at 50% of a replica's capacity, never below {best.min_replicas} replica, planned for an "
              f"8-minute cold start: {best.replica_hours} replica-hours and {best.sar} SAR for the closing day.")
    # END SOLUTION
except NotImplementedError:
    best, policy = ok.sort_values("sar").iloc[0], "(write your policy)"
print(best.to_dict())
print(policy)
S.check(best.signal == "rate", "on this day the rate signal is cheaper at the same attainment")

# %% [markdown]
# ✱ **Stretch** · Allow zero replicas overnight. What happens to the first visitors of the morning, and to the
# worst hour?

# %%
rows0, zero = Z.simulate(closing, rate_policy, capacity, min_replicas=0, cold=4)
first = next(r for r in rows0 if r["arrivals"] and r["minute"] > 300 and r["replicas_ready"] == 0)
print("min 0:", zero, "SAR", cost(zero))
print("a minute with visitors and no replica ready:", first)

# %% [markdown]
# ## Check yourself
#
# 1. Why scale on the request rate rather than the queue for this day?
#    <details><summary>Answer</summary>The rate rises before the queue does, so replicas start while there is still room; the queue signal reacts once visitors already wait and overshoots during the cold start.</details>
# 2. Why plan with the slow cold start?
#    <details><summary>Answer</summary>The evening rise is steep; a policy that only works with a 2-minute start fails on the day the weights come over a slow network.</details>
# 3. What does the minimum of one replica cost, and buy?
#    <details><summary>Answer</summary>A card idling most of the night, and no visitor ever waiting minutes for the first replica of the morning.</details>
#
# **Carry forward** · `artifacts/D4_3/scaling_grid.csv` and your policy: the scaling section of the capstone.
