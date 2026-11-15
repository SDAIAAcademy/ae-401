# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 2.3 · Prompt layout and tenant isolation
#
# Day 2 · Session 3 (Prefix caching starts with prompt layout) · 25 min.
#
# The engine already has prefix caching switched on, and it hits almost nothing: Musamir's prompt starts with a
# header that changes on every request. You reorder the prompt so that what is stable comes first, measure the
# hit rate and the time to first token, then find the cache blocks that one tenant reuses from another and
# remove them with a per-tenant salt. The isolation test is part of the outcome's evidence.
#
# **AI tutor allowed**: it may explain how block hashes work; it does not choose the layout.
#
# **Starts from** `data/tenants.json` (the prompt as deployed) and `data/deployments/tuned_day2.json` with the
# deployed layout and no salt.
#
# **Tasks**
# - [ ] 1 · Hit rate of the prompt as deployed (4 min)
# - [ ] 2 · Reorder the prompt layout (7 min)
# - [ ] 3 · Hit rate and TTFT after the change (5 min)
# - [ ] 4 · Find the cross-tenant hits (4 min)
# - [ ] 5 · Salt by tenant and pass the isolation test (5 min)
# - [ ] ✱ Stretch · Hit rate against cache size

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

T = S.tenants()
print(T["prompt_as_deployed"])
USERS, N = 6, S.pick(300, 200)
start = S.Config.load("tuned_day2").but(layout="deployed", cache_salt="none")
pool = S.peak_requests(seed=0)

# %% [markdown]
# ## 1 · Hit rate of the prompt as deployed (4 min)
#
# The engine hashes each full block of 16 tokens together with everything before it. A request reuses a block
# only if **every token before it** is identical. `hit_rate` is the share of looked-up blocks found in the cache.

# %%
def measure(cfg):
    run = S.run_closed(cfg, USERS, pool=pool, n_requests=N, seed=1)
    s, t = run.summary(), pd.DataFrame(run.table())
    return run, {"hit_rate": s["hit_rate"], "computed_prompt_tokens": int((t.prompt_tokens - t.hit_tokens).mean()),
                 "ttft_p50": s["ttft_p50"], "ttft_p90": s["ttft_p90"], "attainment": s["attainment"], "goodput_rps": s["goodput_rps"]}


run0, before = measure(start)
print("as deployed:", before)
S.check(before["hit_rate"] < 0.05, "the deployed prompt reuses almost nothing")

# %% [markdown]
# ## 2 · Reorder the prompt layout (7 min)
#
# The prompt has six parts: `header` (request number, visitor's first name, the second of the day: different every
# time), `base` (Musamir's instructions, the same for every tenant), `tenant` (the organiser's rules), `document`
# (event terms, when asked), `history` (earlier turns of the conversation) and `question`.
#
# ✏️ **Exercise** · Write `MY_ORDER`, the six names in the order that maximises reuse while keeping every part.

# %%
try:
    # BEGIN SOLUTION
    MY_ORDER = ["base", "tenant", "document", "history", "header", "question"]
    # END SOLUTION
except NotImplementedError:
    MY_ORDER = S.reference("d2_3")["order"]
S.LAYOUTS["mine"] = MY_ORDER
S.check(sorted(MY_ORDER) == sorted(["header", "base", "tenant", "document", "history", "question"]), "every part kept once")
S.check(MY_ORDER.index("header") > MY_ORDER.index("history"), "the per-request header comes after everything shared")

# %% [markdown]
# ## 3 · Hit rate and TTFT after the change (5 min)

# %%
run1, after = measure(start.but(layout="mine"))
comp = pd.DataFrame([before, after], index=["as deployed", "stable first"])
print(comp.to_string())
S.save_rows([{"layout": i, **r} for i, r in comp.iterrows()], "D2_3", "layout_comparison")
S.check(after["hit_rate"] > 0.4 and after["ttft_p50"] < before["ttft_p50"], "most leading blocks reused; the median TTFT falls")

# %% [markdown]
# 🔍 **Look at** `computed_prompt_tokens`: the engine now computes a fraction of each prompt. The history of a
# multi-turn conversation is the biggest gain: turn 3 reuses everything turn 2 computed.
#
# ## 4 · Find the cross-tenant hits (4 min)
#
# The base instructions are the same text for every tenant, so a Sawt Concerts visitor reuses blocks first
# computed for a Qamra Festival visitor. vLLM's documentation explains why that matters: a request that reuses
# a cached block is answered faster, so timing can tell an attacker whether another tenant's prompt starts with
# a given text. Rawya's rule: no cache block crosses tenants.
#
# ✏️ **Exercise** · From `run1.table()`, count per tenant the requests whose `cross_tenant_blocks` is above zero.

# %%
t1 = pd.DataFrame(run1.table())
try:
    # BEGIN SOLUTION
    crossing = t1[t1.cross_tenant_blocks > 0].groupby("tenant").size()
    # END SOLUTION
except NotImplementedError:
    crossing = t1[t1.cross_tenant_blocks > 0].groupby("tenant").size()
print(crossing.to_string())
print("blocks reused across tenants:", int(t1.cross_tenant_blocks.sum()))
S.check(t1.cross_tenant_blocks.sum() > 0, "the shared base text is reused across tenants")

# %% [markdown]
# ## 5 · Salt by tenant and pass the isolation test (5 min)
#
# A **cache salt** is mixed into the hash of the first block (vLLM accepts `cache_salt` per request), so requests
# of different tenants never produce the same hashes. Here `cache_salt="tenant"` salts every request with its
# tenant.
#
# ✏️ **Exercise** · `isolation_test(table)` raises `AssertionError` if any request reused a block computed for another
# tenant, and returns the number of requests checked. Run it on the salted run.

# %%
def isolation_test(table):
    # BEGIN SOLUTION
    bad = [r for r in table if r["cross_tenant_blocks"] > 0]
    assert not bad, f"{len(bad)} requests reused another tenant's blocks, first {bad[0]['id']}"
    return len(table)
    # END SOLUTION


run2, salted = measure(start.but(layout="mine", cache_salt="tenant"))
try:
    checked = isolation_test(run2.table())
except NotImplementedError:
    checked = len(run2.table())
    assert sum(r["cross_tenant_blocks"] for r in run2.table()) == 0
print(f"isolation test passed on {checked} requests")
comp.loc["stable first, salted"] = salted
print(comp.to_string())
S.save_rows([{"layout": i, **r} for i, r in comp.iterrows()], "D2_3", "layout_comparison")
S.check(salted["hit_rate"] > 0.9 * after["hit_rate"], "salting costs little reuse: tenants mostly reuse their own blocks")
try:
    isolation_test(run1.table())
    S.check(False, "the unsalted run should fail the test")
except AssertionError as e:
    print("unsalted run fails as it should:", str(e)[:70])
except NotImplementedError:
    print("write isolation_test to see the unsalted run fail it")

# %% [markdown]
# ✱ **Stretch** · Reuse needs the blocks to still be in the cache. Lower `gpu_memory_utilization` (less cache) and
# raise it, and plot the hit rate.

# %%
for u in (0.84, 0.87, 0.90):
    print(u, measure(start.but(layout="mine", cache_salt="tenant", gpu_memory_utilization=u))[1]["hit_rate"])

# %% [markdown]
# ## Check yourself
#
# 1. Why does one changing token at the start of a prompt defeat the whole prefix cache?
#    <details><summary>Answer</summary>Each block's hash covers every token before it, so a change in block 1 changes the hash of every block after it.</details>
# 2. The base text is public. Why block its reuse across tenants anyway?
#    <details><summary>Answer</summary>The rule must not depend on judging which text is sensitive; and a salt per tenant also stops timing probes on the tenant parts that do differ. It costs little reuse.</details>
# 3. Where should the visitor's name go if the assistant must use it?
#    <details><summary>Answer</summary>After everything shared: in the per-request part next to the question, so it never sits in a block anyone else could reuse.</details>
#
# **Carry forward** · The isolation test is part of the outcome's evidence; put it in your capstone repository
# as a test that runs on every benchmark.
