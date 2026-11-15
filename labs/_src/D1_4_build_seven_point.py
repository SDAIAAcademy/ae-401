# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 1.4 · A seven-point goodput curve
#
# Day 1 · Session 4 (Curves, not single numbers) · 25 min.
#
# One run at one load is an anecdote. You combine supplied runs of the baseline at six concurrency levels
# (three repeats each, raw rows) with three repeats of your own at a seventh level, aggregate each point with
# its spread, plot goodput and the 90th-percentile latencies, and write the capacity claim with its variance.
# This is the outcome's evidence: a repeatable curve of at least seven points with raw data and run-to-run
# variance.
#
# **AI tutor allowed** for plotting syntax.
#
# **Starts from** `data/reference/curve_runs.csv` (synthetic: the course simulator, seeds 101 to 103).
#
# **Tasks**
# - [ ] 1 · Load the supplied runs and their raw rows (3 min)
# - [ ] 2 · Aggregate each point with mean and spread (6 min)
# - [ ] 3 · Add your own point with three repeats (6 min)
# - [ ] 4 · Plot goodput and p90 latencies over concurrency (5 min)
# - [ ] 5 · Find the knee and write the capacity claim (5 min)
# - [ ] ✱ Stretch · Compare closed-loop users with open-loop arrivals at the same rate

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
cfg = S.Config.load("baseline")

# %% [markdown]
# ## 1 · Load the supplied runs and their raw rows (3 min)
#
# One row per request: the level (`users`), the `repeat`, when it was sent and how long each part took. The
# simulator already dropped the first 10% of each run as warm-up.

# %%
supplied = pd.read_csv(S.DATA / "reference" / "curve_runs.csv")
print(supplied.groupby(["users", "repeat"]).size().unstack())
print(supplied.head(3).to_string(index=False))

# %% [markdown]
# ## 2 · Aggregate each point with mean and spread (6 min)
#
# For each run (a level and a repeat): goodput = requests that met the objective / (last end - first send),
# and the p90 of TTFT and TPOT. Then for each level the mean and the standard deviation over its repeats.
#
# ✏️ **Exercise** · `per_run(df)` returns one row per (users, repeat) with `goodput_rps`, `throughput_rps`,
# `ttft_p90`, `tpot_p90`; `per_level(runs)` returns the mean and standard deviation of each by level.

# %%
def per_run(df):
    # BEGIN SOLUTION
    out = []
    for (u, rep), x in df.groupby(["users", "repeat"]):
        dur = x.end.max() - x.arrival.min()
        out.append({"users": u, "repeat": rep, "goodput_rps": x.met.sum() / dur, "throughput_rps": len(x) / dur,
                    "ttft_p90": x.ttft_s.quantile(0.9), "tpot_p90": x.tpot_s.quantile(0.9)})
    return pd.DataFrame(out)
    # END SOLUTION


def per_level(runs):
    # BEGIN SOLUTION
    return runs.groupby("users")[["goodput_rps", "throughput_rps", "ttft_p90", "tpot_p90"]].agg(["mean", "std"])
    # END SOLUTION

# %%
def _ref_per_run(df):
    out = []
    for (u, rep), x in df.groupby(["users", "repeat"]):
        dur = x.end.max() - x.arrival.min()
        out.append({"users": u, "repeat": rep, "goodput_rps": x.met.sum() / dur, "throughput_rps": len(x) / dur,
                    "ttft_p90": x.ttft_s.quantile(0.9), "tpot_p90": x.tpot_s.quantile(0.9)})
    return pd.DataFrame(out)


try:
    runs = per_run(supplied)
    levels = per_level(runs)
except NotImplementedError:
    runs = _ref_per_run(supplied)
    levels = runs.groupby("users")[["goodput_rps", "throughput_rps", "ttft_p90", "tpot_p90"]].agg(["mean", "std"])
print(levels.round(3).to_string())
S.check(len(levels) == 6, "six supplied levels")

# %% [markdown]
# ## 3 · Add your own point with three repeats (6 min)
#
# The supplied levels jump from 4 to 8 users, exactly where the curve bends. Run 6 users three times with
# different seeds, in the same format, and append the rows.
#
# ✏️ **Exercise** · Three runs of `S.run_closed(cfg, 6, n_requests=..., seed=...)`; turn each `.table()` into
# rows with `users`, `repeat` and `end = arrival + e2e_s`.

# %%
own = []
try:
    # BEGIN SOLUTION
    for rep in (1, 2, 3):
        run = S.run_closed(cfg, 6, n_requests=S.pick(160, 100), seed=200 + rep)
        for x in run.table():
            own.append({"users": 6, "repeat": rep, **x, "end": x["arrival"] + x["e2e_s"]})
    # END SOLUTION
except NotImplementedError:
    own = S.reference("d1_4")["own_rows"]
own = pd.DataFrame(own)
allrows = pd.concat([supplied, own[supplied.columns.intersection(own.columns)]], ignore_index=True)
S.save_rows(allrows.to_dict("records"), "D1_4", "curve_raw")
try:
    runs = per_run(allrows)
    levels = per_level(runs)
except NotImplementedError:
    runs = _ref_per_run(allrows)
    levels = runs.groupby("users")[["goodput_rps", "throughput_rps", "ttft_p90", "tpot_p90"]].agg(["mean", "std"])
print(levels.round(3).to_string())
S.check(len(levels) >= 7, "a curve of seven points, each with three repeats")

# %% [markdown]
# ## 4 · Plot goodput and p90 latencies over concurrency (5 min)

# %%
plt = S.plt()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
x = levels.index
ax[0].errorbar(x, levels[("throughput_rps", "mean")], yerr=levels[("throughput_rps", "std")], marker="o", label="throughput", capsize=3)
ax[0].errorbar(x, levels[("goodput_rps", "mean")], yerr=levels[("goodput_rps", "std")], marker="o", label="goodput", capsize=3)
ax[0].set(xlabel="concurrent users", ylabel="requests per second", title="Baseline: served against served well")
ax[0].legend()
ax[1].plot(x, levels[("ttft_p90", "mean")], marker="o", label="TTFT p90 (s)")
ax[1].plot(x, levels[("tpot_p90", "mean")] * 10, marker="o", label="TPOT p90 (x10, s)")
ax[1].axhline(OBJ["ttft_s"], ls="--", c="grey")
ax[1].set(xlabel="concurrent users", yscale="log", title="Tail latency")
ax[1].legend()
S.show(fig, "curve", "D1_4")

# %% [markdown]
# ## 5 · Find the knee and write the capacity claim (5 min)
#
# The knee is the level with the highest mean goodput. A capacity claim names the level, the goodput with its
# spread, the number of repeats and the objectives.
#
# ✏️ **Exercise** · Find the knee and fill `claim`.

# %%
knee = levels[("goodput_rps", "mean")].idxmax()
g, sd = levels.loc[knee, ("goodput_rps", "mean")], levels.loc[knee, ("goodput_rps", "std")]
try:
    # BEGIN SOLUTION
    claim = (f"The baseline serves at most {g:.2f} ± {sd:.2f} good requests per second (mean ± sd of 3 runs) "
             f"at {knee} concurrent users, under TTFT ≤ {OBJ['ttft_s']} s, TPOT ≤ {OBJ['tpot_s'] * 1000:.0f} ms and no stall over {OBJ['stall_s']} s; "
             f"beyond it goodput falls while throughput keeps rising.")
    # END SOLUTION
except NotImplementedError:
    claim = "(write your capacity claim)"
print(claim)
S.check(4 <= knee <= 8, "the knee is between 4 and 8 users")
peak_rate = S.load_json("traffic_profile.json")["peak_requests"] / 3600
print(f"the evening peak brings {peak_rate:.2f} requests per second: {peak_rate / g:.0f} times the baseline's best goodput")

# %% [markdown]
# ✱ **Stretch** · Closed-loop users slow down when the engine slows down; real visitors do not. Replay ten
# minutes of the peak at their own arrival times (`S.run_open(cfg, requests)`) and compare its goodput and TTFT
# with the closed-loop point at the same throughput.

# %%
s0, _ = S.peak_window()
open_run = S.run_open(cfg, S.read_trace(start=s0, end=s0 + S.pick(600, 120))).summary()
print({k: open_run[k] for k in ("throughput_rps", "goodput_rps", "attainment", "ttft_p90")})

# %% [markdown]
# ## Check yourself
#
# 1. Why three repeats per point?
#    <details><summary>Answer</summary>To see the run-to-run variance: a difference between two configurations smaller than that spread is not a difference.</details>
# 2. Why is the knee not the level with the highest throughput?
#    <details><summary>Answer</summary>Past the knee the extra requests finish late or slowly; throughput counts them, goodput does not.</details>
# 3. What does a closed-loop test hide that open-loop replay shows?
#    <details><summary>Answer</summary>Closed-loop users wait for each answer before sending the next, so the load eases when the engine is slow; real arrivals keep coming and the queue grows without bound.</details>
#
# **Carry forward** · `artifacts/D1_4/curve_raw.csv` and the claim are the baseline of your capstone report.
