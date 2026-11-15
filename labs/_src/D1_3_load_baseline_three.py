# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 1.3 · The baseline at three concurrency levels
#
# Day 1 · Session 3 (Goodput against an objective) · 25 min.
#
# You load Musamir as deployed through a streaming HTTP endpoint, measure each request from the client's side
# and compute **goodput**: the requests per second that met the objective (time to first token at most 2.0 s,
# time per output token at most 150 ms, no pause between tokens over 1.0 s). The endpoint is the course's
# stand-in engine (`standin_server.py`): it streams tokens at the pace of the simulator, faster by a time scale that you
# divide out. With a notebook GPU and a real engine, set `SERVE_ENGINE_URL` and the same code loads it.
#
# **AI tutor allowed**: it may explain the stream format and the metrics; it does not choose settings.
#
# **Starts from** `data/deployments/baseline.json` and the evening-peak requests of `data/trace_day.csv`.
#
# **Tasks**
# - [ ] 1 · Start the endpoint and stream one request (4 min)
# - [ ] 2 · Measure TTFT, TPOT and end-to-end time from the stream (6 min)
# - [ ] 3 · Load at 2, 8 and 24 concurrent users (6 min)
# - [ ] 4 · Goodput against the objectives (5 min)
# - [ ] 5 · Where the time went: queue, prefill, decode (4 min)
# - [ ] ✱ Stretch · Point the same load generator at a real engine on a notebook GPU

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

import json, time, urllib.request
import pandas as pd
import servelab as S
import loadgen as LG
import standin_server as SI

OBJ = S.objectives()
ENGINE_URL = os.environ.get("SERVE_ENGINE_URL")      # a real engine, if you have one
SCALE = 1.0 if ENGINE_URL else 0.25                   # the stand-in runs four times faster than simulated time
print("objectives:", OBJ["note"])

# %% [markdown]
# ## 1 · Start the endpoint and stream one request (4 min)
#
# The stand-in listens on a free local port in this notebook's process. One request: a 1,500-token prompt and
# 100 tokens of answer. Read three raw events of the stream: this is what every OpenAI-compatible engine sends.

# %%
if ENGINE_URL:
    URL, httpd, app = ENGINE_URL, None, None
else:
    httpd, app = SI.serve("baseline", port=0, scale=SCALE)
    URL = f"http://127.0.0.1:{httpd.server_address[1]}"
print("endpoint:", URL)
body = json.dumps({"model": "musamir", "prompt": LG.prompt_of(1500), "max_tokens": 100, "stream": True}).encode()
req = urllib.request.Request(URL + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
times, raw = [], []
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            times.append(time.perf_counter())
            if len(raw) < 3:
                raw.append(line)
t_end = time.perf_counter()
print("\n".join(raw))
print(len(times), "token events")

# %% [markdown]
# ## 2 · Measure TTFT, TPOT and end-to-end time from the stream (6 min)
#
# **TTFT** is from sending to the first token. **TPOT** is the time per output token after the first one:
# `(last token - first token) / (tokens - 1)`. The **stall** is the longest gap between two tokens. Divide every
# duration by `SCALE` to get engine time.
#
# ✏️ **Exercise** · `from_stream(t0, times, t_end, scale)` returns `{"ttft_s", "tpot_s", "stall_s", "e2e_s"}`.

# %%
def from_stream(t0, times, t_end, scale=SCALE):
    # BEGIN SOLUTION
    ttft = (times[0] - t0) / scale
    tpot = (times[-1] - times[0]) / max(1, len(times) - 1) / scale
    stall = max((b - a for a, b in zip(times, times[1:])), default=0.0) / scale
    return {"ttft_s": ttft, "tpot_s": tpot, "stall_s": stall, "e2e_s": (t_end - t0) / scale}
    # END SOLUTION

# %%
try:
    one = from_stream(t0, times, t_end)
except NotImplementedError:
    one = {"ttft_s": (times[0] - t0) / SCALE, "tpot_s": (times[-1] - times[0]) / (len(times) - 1) / SCALE,
           "stall_s": max(b - a for a, b in zip(times, times[1:])) / SCALE, "e2e_s": (t_end - t0) / SCALE}
print({k: round(v, 3) for k, v in one.items()})
if not ENGINE_URL:
    S.check(0.3 < one["ttft_s"] < 0.9 and 0.06 < one["tpot_s"] < 0.09, "one request alone: TTFT and TPOT as Lab 1.1 predicted")

# %% [markdown]
# ## 3 · Load at 2, 8 and 24 concurrent users (6 min)
#
# Closed loop: each user sends the next request as soon as the last one finished. The jobs are real peak
# requests (prompt and answer lengths of the deployed layout). Raw rows are saved to `artifacts/D1_3/`.

# %%
jobs = S.peak_jobs(200, seed=3)
LEVELS = (2, 8, 24)
raw_rows = []
for users in LEVELS:
    n = S.pick(users * 3, users * 2)
    t_start = time.perf_counter()
    rows = LG.run_users(URL, users, n, jobs=jobs)
    for r in rows:
        r.update({"users": users, "ttft_s": r["ttft_s"] / SCALE, "tpot_s": r["tpot_s"] / SCALE, "stall_s": r["stall_s"] / SCALE,
                  "e2e_s": r["e2e_s"] / SCALE, "start": (r["start"] - t_start) / SCALE, "end": (r["end"] - t_start) / SCALE})
    raw_rows += rows
    print(f"{users:2d} users: {len(rows)} requests in {time.perf_counter() - t_start:.1f} s of wall time")
raw = pd.DataFrame(raw_rows)
S.save_rows(raw_rows, "D1_3", "baseline_three_levels")
print(raw.groupby("users")[["ttft_s", "tpot_s", "stall_s", "e2e_s"]].quantile(0.9).round(3))

# %% [markdown]
# ## 4 · Goodput against the objectives (5 min)
#
# A request **meets** the objective when its TTFT is at most 2.0 s, its TPOT at most 150 ms **and** its longest
# stall at most 1.0 s. Goodput is the number that met it divided by the time the level ran (first start to last
# end). Attainment is the share.
#
# ✏️ **Exercise** · `goodput(df)` returns `(throughput_rps, goodput_rps, attainment)` for one level.

# %%
def goodput(df):
    # BEGIN SOLUTION
    met = (df.ttft_s <= OBJ["ttft_s"]) & (df.tpot_s <= OBJ["tpot_s"]) & (df.stall_s <= OBJ["stall_s"])
    dur = df.end.max() - df.start.min()
    return len(df) / dur, met.sum() / dur, met.mean()
    # END SOLUTION

# %%
summary = []
for users, df in raw.groupby("users"):
    try:
        thr, gp, att = goodput(df)
    except NotImplementedError:
        met = (df.ttft_s <= OBJ["ttft_s"]) & (df.tpot_s <= OBJ["tpot_s"]) & (df.stall_s <= OBJ["stall_s"])
        dur = df.end.max() - df.start.min()
        thr, gp, att = len(df) / dur, met.sum() / dur, met.mean()
    summary.append({"users": users, "throughput_rps": round(thr, 3), "goodput_rps": round(gp, 3), "attainment": round(att, 2)})
summary = pd.DataFrame(summary)
print(summary.to_string(index=False))
if not ENGINE_URL:
    S.check(summary.attainment.iloc[-1] < OBJ["attainment_target"], "at 24 users the baseline misses the 90% objective")

# %% [markdown]
# 🔍 **Look at** throughput against goodput at 24 users. The engine is busy, the visitors are not served.

# %%
if httpd:
    SI.stop_server(httpd, app)       # always stop what you started
    print("endpoint stopped")

# %% [markdown]
# ## 5 · Where the time went: queue, prefill, decode (4 min)
#
# A client sees only the stream. The engine knows when a request was first scheduled. Run the same load in the
# simulator, which reports `queue_s` (arrival to first scheduling); prefill is `ttft_s - queue_s`, decode is
# `e2e_s - ttft_s`.
#
# ✏️ **Exercise** · For each level, the mean seconds of queue, prefill and decode per request.

# %%
cfg = S.Config.load("baseline")
parts = []
for users in LEVELS:
    t = pd.DataFrame(S.run_closed(cfg, users, n_requests=S.pick(200, 100), seed=1).table())
    try:
        # BEGIN SOLUTION
        parts.append({"users": users, "queue_s": t.queue_s.mean(), "prefill_s": (t.ttft_s - t.queue_s).mean(),
                      "decode_s": (t.e2e_s - t.ttft_s).mean()})
        # END SOLUTION
    except NotImplementedError:
        parts = S.reference("d1_3")["parts"]
        break
parts = pd.DataFrame(parts).round(2)
print(parts.to_string(index=False))
S.check(parts.queue_s.iloc[-1] > parts.queue_s.iloc[0], "queueing grows with the number of users")

# %% [markdown]
# ✱ **Stretch** · On a notebook GPU, start an OpenAI-compatible engine with a small open-weight model (for example
# `vllm serve Qwen/Qwen3-0.6B` in a terminal of the notebook runtime; the model is downloaded into that runtime,
# never to your laptop), set `SERVE_ENGINE_URL=http://127.0.0.1:8000` and run tasks 1 to 4 again.
#
# ## Check yourself
#
# 1. Why measure TPOT after the first token rather than over the whole request?
#    <details><summary>Answer</summary>The first token includes the queue and the prefill; TPOT isolates the decoding pace the visitor reads at.</details>
# 2. Throughput rose from 8 to 24 users while goodput fell. How?
#    <details><summary>Answer</summary>More requests finished per second, but more of them waited past the TTFT objective, decoded slower than 150 ms a token or froze while another prompt was processed.</details>
# 3. Why is an average latency not the objective?
#    <details><summary>Answer</summary>The objective is per request: a mean can sit under 2 s while a third of visitors wait longer.</details>
#
# **Carry forward** · `artifacts/D1_3/baseline_three_levels.csv`: three points of the baseline's curve; Lab 1.4
# builds the rest.
