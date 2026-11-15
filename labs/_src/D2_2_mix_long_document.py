# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 2.2 · Long documents and chat, chunked or not
#
# Day 2 · Session 2 (Long prompts slow short ones) · 25 min.
#
# On festival evenings a third of Musamir's requests paste a long event-terms document. The third seeded
# deployment processes every prompt whole, so each document stalls every chat answer in the batch. You
# measure the chat visitors' time per output token and their longest stall with chunked prefill off, then on at
# three token budgets, and confirm a fix that meets the objective.
#
# **AI tutor allowed**: it may explain engine options; it does not choose settings.
#
# **Starts from** `data/deployments/fault_interference.json` (seeded fault 3) and a mixed pool of peak requests
# with 35% documents (`S.mixed_pool`).
#
# **Tasks**
# - [ ] 1 · Load the mixed trace (3 min)
# - [ ] 2 · Chunked prefill off: chat TPOT and document TTFT (6 min)
# - [ ] 3 · Chunked prefill on at three token budgets (7 min)
# - [ ] 4 · Plot both latencies over the budget (5 min)
# - [ ] 5 · Write the diagnosis and rerun to the objective (4 min)
# - [ ] ✱ Stretch · Give chat requests priority over documents

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
USERS, N = 5, S.pick(260, 180)
cfg = S.Config.load("fault_interference")
print({k: getattr(cfg, k) for k in ("enable_chunked_prefill", "max_num_batched_tokens", "max_num_seqs")})

# %% [markdown]
# ## 1 · Load the mixed trace (3 min)

# %%
pool = S.mixed_pool(share_docs=0.35, n=400, seed=1)
mix = pd.DataFrame([{"kind": r.kind, "prompt_tokens": S.prompt_tokens(r, cfg.layout), "out_tokens": r.out_tokens} for r in pool])
print(mix.groupby("kind").agg(requests=("kind", "size"), prompt_median=("prompt_tokens", "median"), answer_median=("out_tokens", "median")))

# %% [markdown]
# ## 2 · Chunked prefill off: chat TPOT and document TTFT (6 min)
#
# A request's **longest stall** is the largest gap between two of its tokens (`max_gap` on the request). A
# visitor sees it as the answer freezing mid-sentence.
#
# ✏️ **Exercise** · `by_kind(run)` returns one row per kind (`chat`, `doc`): attainment, p90 TTFT, p90 TPOT, and
# p90 of the longest stall.

# %%
def by_kind(run):
    # BEGIN SOLUTION
    t = pd.DataFrame(run.table())
    t["stall_s"] = [r.max_gap for r in run.rows]
    return t.groupby("kind").agg(attainment=("met", "mean"), ttft_p90=("ttft_s", lambda x: x.quantile(.9)),
                                 tpot_p90=("tpot_s", lambda x: x.quantile(.9)), stall_p90=("stall_s", lambda x: x.quantile(.9))).round(3)
    # END SOLUTION


def _by_kind_ref(run):
    t = pd.DataFrame(run.table())
    t["stall_s"] = [r.max_gap for r in run.rows]
    return t.groupby("kind").agg(attainment=("met", "mean"), ttft_p90=("ttft_s", lambda x: x.quantile(.9)),
                                 tpot_p90=("tpot_s", lambda x: x.quantile(.9)), stall_p90=("stall_s", lambda x: x.quantile(.9))).round(3)

# %%
off = S.run_closed(cfg, USERS, pool=pool, n_requests=N, seed=1)
try:
    off_k = by_kind(off)
except NotImplementedError:
    off_k = _by_kind_ref(off)
print("overall attainment", off.summary()["attainment"])
print(off_k.to_string())
S.check(off_k.loc["chat", "stall_p90"] > 1.0, "chat answers freeze for over a second while a document is prefilled")

# %% [markdown]
# ## 3 · Chunked prefill on at three token budgets (7 min)
#
# With chunking the engine schedules decodes first and fills the rest of the step's token budget
# (`max_num_batched_tokens`) with pieces of prompts.
#
# ✏️ **Exercise** · Run budgets 512, 1024 and 2048 and collect, per budget, the overall attainment and the chat and
# document rows of `by_kind`.

# %%
BUDGETS = (512, 1024, 2048)
rows = [{"budget": "off", "attainment": off.summary()["attainment"], "chat_tpot_p90": off_k.loc["chat", "tpot_p90"],
         "chat_stall_p90": off_k.loc["chat", "stall_p90"], "doc_ttft_p90": off_k.loc["doc", "ttft_p90"]}]
try:
    # BEGIN SOLUTION
    for b in BUDGETS:
        run = S.run_closed(cfg.but(enable_chunked_prefill=True, max_num_batched_tokens=b), USERS, pool=pool, n_requests=N, seed=1)
        k = by_kind(run)
        rows.append({"budget": b, "attainment": run.summary()["attainment"], "chat_tpot_p90": k.loc["chat", "tpot_p90"],
                     "chat_stall_p90": k.loc["chat", "stall_p90"], "doc_ttft_p90": k.loc["doc", "ttft_p90"]})
    # END SOLUTION
except NotImplementedError:
    rows = S.reference("d2_2")["rows"]
sweep = pd.DataFrame(rows)
print(sweep.to_string(index=False))
S.save_rows(rows, "D2_2", "budget_sweep")

# %% [markdown]
# ## 4 · Plot both latencies over the budget (5 min)

# %%
plt = S.plt()
on = sweep[sweep.budget != "off"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(on.budget.astype(int), on.chat_stall_p90, marker="o", label="chat: longest stall p90")
ax[0].axhline(float(sweep.iloc[0].chat_stall_p90), ls="--", c="grey", label="chunking off")
ax[0].set(xlabel="token budget per step", ylabel="seconds", xscale="log", title="Chat visitors")
ax[0].legend()
ax[1].plot(on.budget.astype(int), on.doc_ttft_p90, marker="o", c="tab:red", label="documents: TTFT p90")
ax[1].axhline(float(sweep.iloc[0].doc_ttft_p90), ls="--", c="grey", label="chunking off")
ax[1].set(xlabel="token budget per step", xscale="log", title="Document visitors")
ax[1].legend()
S.show(fig, "budget_tradeoff", "D2_2")

# %% [markdown]
# 🔍 **Look at** the two panels together: a smaller budget protects the chat visitors and makes the document
# visitors wait longer for their first token. The budget is a trade-off you choose against the objective.
#
# ## 5 · Write the diagnosis and rerun to the objective (4 min)
#
# ✏️ **Exercise** · Choose `fixed`: the configuration with chunked prefill and the budget that meets the objective
# at this load, and write `diagnosis` in one sentence with its evidence.

# %%
try:
    # BEGIN SOLUTION
    fixed = cfg.but(enable_chunked_prefill=True, max_num_batched_tokens=1024)
    diagnosis = ("Prefill interference: with whole-prompt prefill each document stalls every chat answer "
                 f"(chat stall p90 {off_k.loc['chat', 'stall_p90']:.1f} s, attainment {off.summary()['attainment']:.0%}); "
                 "chunked prefill with a 1,024-token budget fixes it.")
    # END SOLUTION
except NotImplementedError:
    fixed, diagnosis = S.Config.load("tuned_day2").but(layout="deployed", cache_salt="none"), "(write your diagnosis)"
rerun = S.run_closed(fixed, USERS, pool=pool, n_requests=N, seed=1).summary()
print(diagnosis)
print("rerun:", {k: rerun[k] for k in ("attainment", "ttft_p90", "tpot_p90")})
S.check(rerun["attainment"] >= OBJ["attainment_target"], "the rerun meets the objective on the mixed traffic")
S.save_rows([{"deployment": "fault_interference", "diagnosis": diagnosis}], "D2_2", "diagnosis")

# %% [markdown]
# ✱ **Stretch** · `scheduling_policy="chat_first"` lets waiting chat requests go ahead of waiting documents. What
# does it do to the chat TTFT and to the document TTFT?

# %%
pr = S.run_closed(fixed.but(scheduling_policy="chat_first"), USERS, pool=pool, n_requests=N, seed=1)
print(_by_kind_ref(pr).to_string())

# %% [markdown]
# ## Check yourself
#
# 1. Why does a 5,000-token prompt slow down answers that have nothing to do with it?
#    <details><summary>Answer</summary>Without chunking, the engine runs that prefill as one step of over a second, and every running sequence waits for it before its next token.</details>
# 2. What does a smaller token budget cost, and whom?
#    <details><summary>Answer</summary>Long prompts are cut into more pieces, so the document visitors wait longer for their first token.</details>
# 3. Which metric shows interference better: TPOT or the longest stall?
#    <details><summary>Answer</summary>The stall: a one-second freeze spread over 150 tokens moves TPOT by a few milliseconds but the visitor sees the whole second.</details>
#
# **Carry forward** · `artifacts/D2_2/diagnosis.csv`: the third diagnosis. Your capstone report starts with these three.
