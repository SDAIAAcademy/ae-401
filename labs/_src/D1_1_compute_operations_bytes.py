# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 1.1 · Operations and bytes per token
#
# Day 1 · Session 1 (Prefill and decode are different problems) · 25 min.
#
# **AI off.** This is the hand sizing of Day 1: no assistant, a calculator is fine. You size Musamir's base
# model (Qwen3-8B) and the distilled student on the authority's L4 accelerator, decide which resource bounds
# each phase, and check your prediction against a single-request run of the course simulator.
#
# **Starts from** `data/model_cards.json` and `data/accelerators.json` (copied from the official pages).
#
# **Tasks**
# - [ ] 1 · Read the model card and the spec sheet into numbers (4 min)
# - [ ] 2 · Operations and bytes per decoded token for two models (6 min)
# - [ ] 3 · Compute time and memory time of a 2,000-token prefill (6 min)
# - [ ] 4 · Arithmetic intensity against the ridge point (5 min)
# - [ ] 5 · Predict TTFT and TPOT and compare with a single-request run (4 min)
# - [ ] ✱ Stretch · Repeat the sizing for the T4 of a free notebook GPU

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

import servelab as S

cards, accs = S.models(), S.accelerators()
m8, mini, l4 = cards["qwen3-8b"], cards["musamir-mini"], accs["L4"]
print("Qwen3-8B:", {k: m8[k] for k in ("params", "params_nonembedding", "num_hidden_layers", "hidden_size", "vocab_size")})
print("L4 sheet:", l4["sheet"])
print("Node efficiencies measured by the platform team:", {k: S.CAL[k] for k in ("compute_eff", "mem_eff")})

# %% [markdown]
# ## 1 · Read the model card and the spec sheet into numbers (4 min)
#
# The L4 sheet prints tensor-core figures **with sparsity**, and its footnote says the dense figures are half.
# A dense model runs at the dense figure.
#
# ✏️ **Exercise** · Return the dense 16-bit peak of the L4 in operations per second, and its memory bandwidth
# in bytes per second, from the printed sheet.

# %%
def dense_peak_and_bandwidth(sheet):
    # BEGIN SOLUTION
    peak = sheet["BF16 tensor TFLOPS (with sparsity)"] / 2 * 1e12
    bw = sheet["memory bandwidth GB/s"] * 1e9
    return peak, bw
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    PEAK, BW = dense_peak_and_bandwidth(l4["sheet"])
except NotImplementedError:
    PEAK, BW = l4["fp16_dense_tflops"] * 1e12, l4["bandwidth_gbs"] * 1e9
S.check(abs(PEAK - 121e12) < 1e12 and BW == 300e9, "dense BF16 peak 121 TFLOPS and 300 GB/s")

# %% [markdown]
# ## 2 · Operations and bytes per decoded token for two models (6 min)
#
# To decode one token the model multiplies it by every weight once: about **2 operations per weight**
# (a multiply and an add). The embedding table is a lookup, not a multiply, but the output head is a
# multiply of `hidden_size × vocab_size`. And every weight must be **read from memory** once per step.
#
# ✏️ **Exercise** · `decode_cost(m, bytes_per_weight)` returns `(operations, bytes)` for one decoded token at
# batch size 1: operations `2 × (params_nonembedding + hidden_size × vocab_size)`, bytes `params × bytes_per_weight`.

# %%
def decode_cost(m, bytes_per_weight=2):
    # BEGIN SOLUTION
    ops = 2 * (m["params_nonembedding"] + m["hidden_size"] * m["vocab_size"])
    return ops, m["params"] * bytes_per_weight
    # END SOLUTION

# %%
try:
    cost8, costmini = decode_cost(m8), decode_cost(mini)
except NotImplementedError:
    cost8 = (S.flops_per_token(m8), S.weight_bytes(m8))
    costmini = (S.flops_per_token(mini), S.weight_bytes(mini))
for name, (ops, b) in (("qwen3-8b", cost8), ("musamir-mini", costmini)):
    print(f"{name:13s} {ops / 1e9:6.2f} GFLOP per token | {b / 1e9:5.2f} GB read per step | {ops / b:.2f} FLOP per byte")
S.check(abs(cost8[0] - S.flops_per_token(m8)) < 1e6, "operations per token of the 8B model")

# %% [markdown]
# ## 3 · Compute time and memory time of a 2,000-token prefill (6 min)
#
# A prompt is processed in one pass: 2,000 tokens share **one read** of the weights. Use the node's measured
# efficiencies (`S.CAL["compute_eff"]`, `S.CAL["mem_eff"]`): the share of the peak this engine reaches.
# Attention adds `4 × layers × heads × head_dim` operations per token for each earlier token it attends to
# (on average n/2 of them in a prompt of n tokens).
#
# ✏️ **Exercise** · `prefill_times(m, n)` returns `(compute_seconds, memory_seconds)` for an n-token prompt.

# %%
def prefill_times(m, n, peak=PEAK, bw=BW):
    # BEGIN SOLUTION
    ops = n * decode_cost(m)[0] + S.attn_flops_per_token_per_ctx(m) * n * n / 2
    t_comp = ops / (peak * S.CAL["compute_eff"])
    t_mem = decode_cost(m)[1] / (bw * S.CAL["mem_eff"])
    return t_comp, t_mem
    # END SOLUTION

# %%
try:
    tc, tm = prefill_times(m8, 2000)
except NotImplementedError:
    tc = (2000 * S.flops_per_token(m8) + S.attn_flops_per_token_per_ctx(m8) * 2000 * 1000) / (121e12 * S.CAL["compute_eff"])
    tm = S.weight_bytes(m8) / (300e9 * S.CAL["mem_eff"])
print(f"2,000-token prefill: compute {tc:.3f} s, weight reads {tm:.3f} s")
S.check(tc > 5 * tm, "a long prefill is bound by compute, not by memory")

# %% [markdown]
# ## 4 · Arithmetic intensity against the ridge point (5 min)
#
# The **ridge point** is the intensity (operations per byte) at which compute time equals memory time:
# `peak × compute_eff / (bandwidth × mem_eff)`. Below it a step waits on memory, above it on arithmetic.
# Decoding a batch of `b` sequences reads the weights once and does `b` tokens of work.
#
# ✏️ **Exercise** · Return the ridge point, and the smallest batch size at which decoding the 8B model in
# 16 bits would reach it (ignore the cache reads for now).

# %%
def ridge_and_batch(m, peak=PEAK, bw=BW):
    # BEGIN SOLUTION
    ridge = peak * S.CAL["compute_eff"] / (bw * S.CAL["mem_eff"])
    ops, b = decode_cost(m)
    batch = 1
    while batch * ops / b < ridge:
        batch += 1
    return ridge, batch
    # END SOLUTION

# %%
try:
    ridge, b_star = ridge_and_batch(m8)
except NotImplementedError:
    ridge, b_star = S.reference("d1_1")["ridge"], S.reference("d1_1")["batch_at_ridge"]
print(f"ridge point {ridge:.0f} FLOP/byte | decode reaches it at a batch of about {b_star}")
S.check(200 < ridge < 300 and b_star > 100, "decode at batch 1 is far below the ridge")

# %% [markdown]
# 🔍 **Look at** the batch size you found and the cache of Lab 1.2: will Musamir ever decode that many
# sequences at once on this accelerator in 16 bits? Keep the answer for the next session.

# %% [markdown]
# ## 5 · Predict TTFT and TPOT and compare with a single-request run (4 min)
#
# Your prediction for one request alone: time to first token is about the prefill time; time per output token
# is about the time to read the weights once. Then run one request through the course simulator.

# %%
pred_ttft, pred_tpot = max(tc, tm), tm
cfg = S.Config.load("baseline")
run = S.simulate(cfg, S.synthetic_requests(1, 2000, 200), warmup=0)
row = run.table()[0]
print(f"predicted TTFT {pred_ttft:.3f} s, TPOT {pred_tpot * 1000:.1f} ms")
print(f"simulated TTFT {row['ttft_s']:.3f} s, TPOT {row['tpot_s'] * 1000:.1f} ms")
S.check(abs(row["ttft_s"] - pred_ttft) / row["ttft_s"] < 0.15, "TTFT prediction within 15%")
S.check(abs(row["tpot_s"] - pred_tpot) / row["tpot_s"] < 0.15, "TPOT prediction within 15%")

# %% [markdown]
# The gap that is left is the step overhead (scheduling, sampling, launching kernels) and the cache reads,
# which grow with the context: Lab 1.2.
#
# ✱ **Stretch** · Repeat tasks 1 to 4 for the T4 of a free notebook GPU (`accs["T4"]`): its sheet prints
# dense figures, it has no BF16 tensor cores and 16 GB of memory. Does the 8B model in 16 bits even fit?

# %%
t4 = accs["T4"]
print("T4 sheet:", t4["sheet"])
print(f"8B weights in 16 bits: {S.weight_bytes(m8) / 1e9:.1f} GB on a {t4['memory_gb']} GB card")

# %% [markdown]
# ## Check yourself
#
# 1. Why does decoding one token take about as long as reading every weight once?
#    <details><summary>Answer</summary>Each step does about 2 operations per weight but must read every weight; at batch 1 the arithmetic finishes long before the bytes arrive, so the step waits on memory.</details>
# 2. A 2,000-token prompt and a 100-token prompt: which costs more weight reads?
#    <details><summary>Answer</summary>The same: one pass reads the weights once whatever the number of tokens in it. The longer prompt costs more arithmetic.</details>
# 3. The sheet says 242 TFLOPS. Why did you use 121?
#    <details><summary>Answer</summary>The figure is shown with structured sparsity; a dense model gets half, as the footnote says.</details>
#
# **Carry forward** · The ridge point and the two phase costs go into your sizing sheet; Lab 1.2 adds the cache.
