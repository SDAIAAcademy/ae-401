# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 4.2 · Round robin or prefix-aware
#
# Day 4 · Session 2 (Routing across replicas) · 25 min.
#
# For the Qamra Festival's closing night Liwan forecasts four times the usual evening peak, and Yamen puts three
# replicas behind a load balancer. Round robin spreads the requests evenly, but the turns of one conversation land
# on different replicas, so each replica recomputes history another one already cached. You replay the night on
# three replicas, write a prefix-aware router, compare hits, latency and balance, find the hot spot of a router
# that keys on the tenant, and choose a rule with a fallback.
#
# **AI tutor allowed**: it may help with Python; it does not choose the routing rule.
#
# **Starts from** the evening peak of `data/trace_day.csv` played four times faster, and the engine of Day 3
# (8-bit weights and cache, salted by tenant).
#
# **Tasks**
# - [ ] 1 · Replay round robin on three replicas (5 min)
# - [ ] 2 · Write a prefix-aware router (7 min)
# - [ ] 3 · Compare hit rate, TTFT and balance (6 min)
# - [ ] 4 · Find the hot spot of one tenant (4 min)
# - [ ] 5 · Choose, with a fallback rule (3 min)
# - [ ] ✱ Stretch · Session affinity with a load cap

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

import zlib
import pandas as pd
import servelab as S

OBJ = S.objectives()
cfg = S.Config.load("tuned_day2").but(weight_dtype="fp8", kv_cache_dtype="fp8", max_num_seqs=32)
SPEED, REPLICAS = 4, 3
start, _ = S.peak_window()
night = []
for r in S.read_trace(start=start, end=start + S.pick(600, 300)):
    r2 = S.fresh(r)
    r2.t = (r.t - start) / SPEED
    night.append(r2)
print(f"{len(night)} requests at {len(night) / night[-1].t:.1f} per second on {REPLICAS} replicas")


def measure(router):
    run = S.run_open(cfg, night, replicas=REPLICAS, router=router, seed=1)
    s, t = run.summary(), pd.DataFrame(run.table())
    share = t.replica.value_counts(normalize=True)
    return t, {"hit_rate": s["hit_rate"], "ttft_p50": s["ttft_p50"], "ttft_p90": s["ttft_p90"], "attainment": s["attainment"],
               "goodput_rps": s["goodput_rps"], "busiest_share": round(share.max(), 3)}

# %% [markdown]
# ## 1 · Replay round robin on three replicas (5 min)

# %%
t_rr, rr = measure("round_robin")
print("round robin:", rr)
same = pd.DataFrame([{"session": r.session, "id": r.id} for r in night]).merge(t_rr[["id", "replica"]], on="id")
moved = same.groupby("session").replica.nunique()
print(f"conversations with more than one turn whose turns landed on more than one replica: {(moved > 1).mean():.0%}")

# %% [markdown]
# ## 2 · Write a prefix-aware router (7 min)
#
# A router is a function `router(engines, request, state)` that returns one of the engines. Each engine has
# `running` and `waiting` lists; `state` is a dictionary kept for you across calls. The request has `tenant`,
# `session` (the conversation) and `turn`.
#
# ✏️ **Exercise** · `by_session` sends every turn of a conversation to the same replica (hash the session with
# `zlib.crc32`, so the choice is stable), and falls back to the least-loaded replica when that one has more than
# `LIMIT` requests running and waiting.

# %%
LIMIT = 24


def load(e):
    return len(e.running) + len(e.waiting)


def by_session(engines, r, state, limit=LIMIT):
    # BEGIN SOLUTION
    i = zlib.crc32(r.session.encode()) % len(engines)
    if limit is not None and load(engines[i]) > limit:
        i = min(range(len(engines)), key=lambda j: load(engines[j]))
    return engines[i]
    # END SOLUTION


def _by_session_ref(engines, r, state, limit=LIMIT):
    i = zlib.crc32(r.session.encode()) % len(engines)
    if limit is not None and load(engines[i]) > limit:
        i = min(range(len(engines)), key=lambda j: load(engines[j]))
    return engines[i]


class _Idle:                       # a stand-in engine with nothing running, to test the router once
    running, waiting = [], []


try:
    by_session([_Idle(), _Idle(), _Idle()], night[0], {})
    router = by_session
except NotImplementedError:
    router = _by_session_ref

# %% [markdown]
# ## 3 · Compare hit rate, TTFT and balance (6 min)

# %%
t_s, sess = measure(router)
t_nl, nolimit = measure(lambda e, r, s: router(e, r, s, limit=None))
comp = pd.DataFrame([rr, nolimit, sess], index=["round robin", "session, no limit", f"session, limit {LIMIT}"])
print(comp.to_string())
S.save_rows([{"router": i, **r} for i, r in comp.iterrows()], "D4_2", "routers")
S.check(sess["attainment"] > rr["attainment"] and sess["hit_rate"] >= rr["hit_rate"], "session routing with a limit beats round robin")
S.check(nolimit["busiest_share"] > rr["busiest_share"] and nolimit["attainment"] < sess["attainment"],
        "without a limit, affinity piles work on one replica and serves fewer visitors")

# %% [markdown]
# 🔍 **Look at** the hit rate: it rises only a little, because every replica already caches each tenant's shared
# instructions. The gain is in the tail: no replica recomputes a long history, and the limit absorbs the bursts
# that pure affinity piles on one replica.
#
# ## 4 · Find the hot spot of one tenant (4 min)
#
# A simpler idea keys on the tenant, so each tenant's rules stay cached on one replica.
#
# ✏️ **Exercise** · Write `by_tenant`, run it, and show the share of requests on each replica and its attainment.

# %%
def by_tenant(engines, r, state):
    # BEGIN SOLUTION
    return engines[zlib.crc32(r.tenant.encode()) % len(engines)]
    # END SOLUTION


try:
    t_t, ten = measure(by_tenant)
except NotImplementedError:
    t_t, ten = measure(lambda e, r, s: e[zlib.crc32(r.tenant.encode()) % len(e)])
print("by tenant:", ten)
print(t_t.groupby("replica").agg(requests=("id", "size"), attainment=("met", "mean"), tenants=("tenant", lambda x: ", ".join(sorted(set(x))))))
S.check(ten["busiest_share"] > 0.45 and ten["attainment"] < 0.8, "one replica carries the largest tenant and misses the objective")

# %% [markdown]
# ## 5 · Choose, with a fallback rule (3 min)
#
# ✏️ **Exercise** · Write `rule` in two sentences: the router you would deploy and when it falls back.

# %%
try:
    # BEGIN SOLUTION
    rule = (f"Route by conversation, so every turn reuses its own history, with a fallback to the least-loaded replica "
            f"when the chosen one holds more than {LIMIT} requests. Never route by tenant: the largest tenant becomes a hot spot.")
    # END SOLUTION
except NotImplementedError:
    rule = "(write your rule)"
print(rule)

# %% [markdown]
# ✱ **Stretch** · Try limits of 8, 16 and 32. Which keeps the most hits while meeting the objective?

# %%
for lim in (8, 16, 32):
    print(lim, measure(lambda e, r, s, lim=lim: router(e, r, s, limit=lim))[1])

# %% [markdown]
# ## Check yourself
#
# 1. Round robin is perfectly balanced. Why does it lose to session routing with a limit?
#    <details><summary>Answer</summary>Each new turn lands on a replica that has not cached the conversation's history, so it recomputes it, and those long prefills fall at the busiest moments.</details>
# 2. Why does session routing without a limit gain so little?
#    <details><summary>Answer</summary>A hash spreads conversations evenly on average, but bursts pile on one replica while others idle and nothing moves the work; on the longer replay it loses to round robin.</details>
# 3. What does routing by tenant cost, and why is it tempting?
#    <details><summary>Answer</summary>Its cache hits are the highest, but the largest tenant's replica runs over capacity while another idles.</details>
#
# **Carry forward** · `artifacts/D4_2/routers.csv` and your rule: the router section of the capstone.
