---
course: "Inference Performance and Cost Engineering"
code: "AE-401"
version: "2027.2.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-401/"
---

# Inference Performance and Cost Engineering

Make a language model service meet its latency objectives at the lowest cost you can defend, with quality held in Arabic and English

Participants take the bilingual visitor assistant of a fictional national events authority, too slow at evening peaks and too costly, and make it meet its service objective at the lowest cost they can defend. They build a first-principles cost model, measure goodput curves on replayed traffic, diagnose seeded bottlenecks, then apply batching, caching, quantisation, speculative decoding, distillation, routing, disaggregated serving and autoscaling, re-running an Arabic and English task evaluation after every change.

- **Audience:** Specialists. Engineers who have completed the level 300 core or can show equivalent work: a production AI system they built, operated and can defend.
- **Prerequisites:** AE-305 Operating Machine Learning and Language Model Systems; AE-307 Container Orchestration and Cloud Infrastructure for AI Workloads; Has served an open-weight model behind an HTTP endpoint and load-tested it; Computes percentiles and reads a latency histogram; Estimates memory from parameter count and numeric precision
- **Case study:** Liwan National Events Authority: Musamir, the bilingual visitor assistant of a fictional national events authority, serves three event organisers from one leased accelerator and is too slow at evening peaks and too costly. Every lab works on the same synthetic data: 40,000 Arabic and English visitor requests over a day, model cards and spec sheets, recorded evaluation outcomes for six model variants and a course simulator of the serving engine, all generated with a fixed seed.

## Outcomes

- LO1 (Applied): Compute prefill cost, decode cost and key-value cache memory, and predict the binding resource, given a model card, an accelerator specification sheet and a traffic profile
- LO2 (Performance): Evaluate a serving configuration by measuring its goodput curve over concurrency against latency objectives, given an engine, a small open-weight model and a replayed synthetic trace
- LO3 (Performance): Diagnose why a deployment misses its service objective and confirm the fix, given three seeded faults: queueing, preemption under cache pressure and prefill interference
- LO4 (Performance): Optimise batching, admission, prefix caching and cache precision while holding task quality, given a multi-tenant trace and Arabic and English evaluation sets
- LO5 (Performance): Evaluate quantised, distilled and speculative-decoding variants against the base model and decide which to serve, given a quality floor per language and a cost target
- LO6 (Performance): Design and defend a serving topology covering disaggregation, routing, replica count and scaling policy, given a daily demand pattern, a service objective and a budget

## Day 1: Model the cost first

Prefill, decode, cache and goodput: predict before you measure

#### [day1#title] Model the cost first

*SDAIA Academy · Inference Performance and Cost Engineering*

Day 1: Prefill, decode, cache and goodput: predict before you measure

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-401

#### [day1#four-days-one-service] Four days on one service: predict, measure, fix, then design

*The course in one slide*

- **Day 1 · Model the cost.** Prefill, decode, cache arithmetic, goodput curves.
- **Day 2 · Inside the engine.** Three seeded faults, batching, chunking, caches.
- **Day 3 · Cheaper tokens.** Precision, speculation, distillation, each re-evaluated.
- **Day 4 · Over the day.** Disaggregation, routing, scaling, cost per success.

The capstone tunes the deployment you measure today until it meets its objectives at the lowest cost you can defend.

#### [day1#liwan-and-musamir] Musamir is too slow at the evening peak and too costly

*The case study*

- **Liwan National Events Authority** (fictional) runs Musamir, a bilingual visitor assistant.
- Three tenants share it: Qamra Festival, Sawt Concerts, Athar Heritage Week.
- One leased L4 accelerator serves Qwen3-8B in 16 bits.
- 40,000 synthetic requests a day; 7,656 in the busiest hour; 55% Arabic; 18% ask about long event terms.

- **The complaint.** Visitors wait, answers crawl, and the lease is the largest line of the digital services budget. Dr. Hanadi Al-Kathiri wants the objective met at a cost she can defend.

#### [day1#people-of-the-case] Five people own a piece of the decision

*Who you work with*

| Person | Owns |
| --- | --- |
| Dr. Hanadi Al-Kathiri, head of visitor digital services | the service objective and the budget |
| Yamen Al-Hulaibi, platform engineer | the accelerator node and the engine settings |
| Asrar Al-Rifai, evaluation lead | the Arabic and English evaluation sets and their floors |
| Rawya Al-Twaim, data protection officer | what may be cached and shared between tenants |
| Kenan Al-Ajlan, finance analyst | the node lease and the hosted-API quote |

You are Tamader Al-Harqan, the inference engineer asked to fix it.

#### [day1#objective-of-the-service] The objective is per request, and it has three clocks

*What "fixed" means*

- **2.0 s** time to first token, at most
- **150 ms** time per output token, at most
- **1.0 s** longest pause between two tokens, at most
- **90%** of peak requests meet all three

> **Today's question.** How far is the deployment from that, and can we say so before we measure?

#### [day1#setup-check] A laptop processor or a Colab runtime is enough

*Before the first lab*

- Clone the course repository and open `labs/`.
- Install `uv`, then `uv sync` in `labs/`.
- Or open any lab in Colab from the course page: the first cell clones the repository.
- No model is downloaded to your laptop, and no key is needed today.

```bash
# file: labs/
uv sync
uv run prepare_data.py   # optional: same bytes
uv run jupyter lab
```

#### [day1#how-the-labs-run] The labs run a simulator of the engine; a notebook GPU can run the real one

*Two tiers*

- **Simulator tier (default).** A course simulator of a serving engine: continuous batching, a paged cache with prefix hashing, chunked prefill, preemption. Calibrated from the real model card and spec sheet. Every supplied run comes from it and is marked synthetic.
- **Engine tier (optional).** On a free notebook GPU, an OpenAI-compatible engine with a small open-weight model. The same load generator measures it. The model is downloaded into the notebook runtime, never to a laptop.

#### [day1#agenda] Model the cost first

*Today*

- **Session 1 · Prefill and decode are different problems**: Why a prompt is processed at the speed of arithmetic and an answer at the speed of memory.
- **Session 2 · Key-value cache arithmetic**: The cache, not the weights, decides how many visitors are served at once.
- **Session 3 · Goodput against an objective**: Requests per second that meet the whole latency objective, measured from a stream.
- **Session 4 · Curves, not single numbers**: A capacity is a point on a curve with an error bar, not a number from one run.

### Session 1: Prefill and decode are different problems

#### [day1#s1] Prefill and decode are different problems

*Session 1*

- Compute the operations and bytes of one token for a model card
- Place prefill and decode on the roofline of an accelerator
- Predict time to first token and time per output token from first principles

#### [day1#two-phases] The prompt is read in one pass; the answer is written one token at a time

*One request, two phases*

1. **1 · Prompt** 2,000 tokens arrive
2. **2 · Prefill** all 2,000 in one pass
3. **3 · First token** the visitor sees a reply start
4. **4 · Decode** one pass per token

> **Two phases, two bottlenecks.** Prefill does a lot of work per byte read; decode does almost none.

#### [day1#two-operations-per-weight] A token costs about two operations per weight, and a read of every weight

*Lab 1.1 · task 2*

| Model | Operations per token | Bytes read per step (16-bit) | Operations per byte |
| --- | --- | --- | --- |
| Qwen3-8B (base) | 15.14 G | 16.40 GB | 0.92 |
| Musamir-mini (1.7B student) | 3.42 G | 3.40 GB | 1.01 |

Operations: 2 × (non-embedding parameters + the output head, hidden size × vocabulary). Bytes: every parameter, 2 bytes each.

#### [day1#roofline-for-serving] Below the ridge a step waits on memory; above it, on arithmetic

*The roofline*

- **Intensity** = operations ÷ bytes moved.
- **Ridge point** = reachable operations per second ÷ reachable bytes per second.
- On the L4: 121 dense BF16 TFLOPS at 45%, 300 GB/s at 80%: **227 operations per byte**.
- The sheet prints 242 TFLOPS *with sparsity*; dense is half.

- **Recap from AE-206.** The roofline of a layer and the efficiencies of an accelerator are in AE-206, Day 1. Here we apply it to the two phases of a language model.

#### [day1#prefill-is-compute-bound] A 2,000-token prefill is bound by arithmetic, eight times over

*Lab 1.1 · task 3*

| Item | Value |
| --- | --- |
| Compute time | 0.578 s |
| Weight reads | 0.068 s |

2,000 tokens share one read of the weights. The prompt's length sets the time to first token.

#### [day1#decode-is-memory-bound] Decoding one token does 0.92 operations per byte: it waits on memory

*Lab 1.1 · task 4*

- **0.92** operations per byte at batch 1
- **246** sequences decoded together to reach the ridge

> **Batching is free until the ridge.** Each extra sequence in a decode step reuses the same weight read.

#### [day1#predict-then-measure] The hand prediction lands within 10% of the single-request run

*Lab 1.1 · task 5*

|  | Predicted | Simulated run |
| --- | --- | --- |
| Time to first token, 2,000-token prompt | 0.578 s | 0.599 s |
| Time per output token | 68.3 ms | 73.6 ms |

The gap is the step overhead and the cache reads, which grow with the context.

#### [day1#checkpoint-1] Where does the time go?

*Checkpoint*

**Question:** Musamir decodes one visitor's answer alone on the L4 at about 74 ms a token. Yamen proposes a card with twice the TFLOPS and the same memory bandwidth. What happens to the time per token?

- A. It halves, to about 37 ms
- B. It barely changes: decode waits on reading the weights
- C. It rises, because prefill takes more of each step
- D. It falls only if the card has more memory

#### [day1#lab-1-1] Operations and bytes per token

Lab 1.1 · 25 min

By hand, without AI: operations and bytes per token for two models on the L4, which resource bounds each phase, and a prediction checked against a single-request run.

[D1_1_compute_operations_bytes.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D1_1_compute_operations_bytes.ipynb)

- [ ] Read the model card and the spec sheet into numbers (4 min)
- [ ] Operations and bytes per decoded token for two models (6 min)
- [ ] Compute time and memory time of a 2,000-token prefill (6 min)
- [ ] Arithmetic intensity against the ridge point (5 min)
- [ ] Predict TTFT and TPOT and compare with a single-request run (4 min)
- [ ] ✱ Stretch: Repeat the sizing for the T4 of a free notebook GPU

### Session 2: Key-value cache arithmetic

#### [day1#s2] Key-value cache arithmetic

*Session 2*

- Compute key-value cache bytes per token from a model card
- Predict the concurrency limit at a given context length
- Explain the gap between a prediction and the engine's block count

#### [day1#keys-and-values] Every token leaves a key and a value in every layer

*The cache*

```python
# file: Lab 1.2 · task 1
kv_per_token = 2 * layers * kv_heads * head_dim * bytes
#            = 2 * 36     * 8        * 128      * 2
#            = 147,456 bytes = 144 KiB per token
```

A 2,000-token conversation holds 0.29 GB of cache while it is served.

#### [day1#grouped-query-attention] Grouped-query attention cut the cache by four

*Read the model card*

- **32 query heads.** Used during the step, never stored.
- **8 key-value heads.** Stored for every token: the cache is a quarter of what 32 heads would need.

> **What goes wrong.** Sizing with `num_attention_heads` overestimates the cache four times and underestimates concurrency four times.

#### [day1#memory-left-for-cache] After the weights, 3.6 GB of the 24 are left for conversations

*Lab 1.2 · task 2*

| Item | Value |
| --- | --- |
| Card | 24.0 GB |
| Engine may use (90%) | 21.6 GB |
| Weights, 16-bit | 16.4 GB |
| Reserve | 1.6 GB |
| Cache | 3.6 GB |

#### [day1#pages-not-slabs] The engine hands out the cache in blocks of 16 tokens

*Paged cache*

- A conversation takes blocks as it grows, not a slab sized for the longest possible answer.
- Musamir's engine reports **1,525 blocks = 24,400 tokens** at start.
- Blocks can be shared between requests with the same beginning: Day 2's prefix cache.
- When no block is free, a running sequence is **preempted** and recomputed later.

#### [day1#three-limits] Three context lengths, three limits, all within the 25% band

*Lab 1.2 · tasks 3 to 5*

| Tokens per conversation | Predicted | Engine held at most | Preemptions |
| --- | --- | --- | --- |
| 2,000 | 12.2 | 14 | 14 |
| 4,000 | 6.1 | 7 | 13 |
| 8,000 | 3.0 | 3 | 0 |

The engine admits at the prompt's length, briefly holds one more, then preempts as answers grow.

#### [day1#answers-grow-the-cache] A full cache preempts during decoding, not only at admission

*Preemption*

- Every answer token needs room; at 16 tokens a block, a sequence asks for a new block every 16 tokens.
- No free block: the engine frees the newest sequence and recomputes it later.
- The visitor sees it as a sudden stall, and the work is done twice.

- **Seeded fault 2.** Tomorrow you meet a deployment that preempts all evening. This slide is how you will recognise it.

#### [day1#eight-bit-weights-teaser] With 8-bit weights the cache more than triples

*Lab 1.2 · stretch*

- **24,400** tokens of cache, 16-bit weights
- **80,016** tokens of cache, 8-bit weights

Halving the weights frees 8.2 GB, all of it for conversations. Whether quality holds is Day 3's question.

#### [day1#checkpoint-2] How many fit?

*Checkpoint*

**Question:** Another team serves a model with 32 layers, 32 query heads and 8 key-value heads of size 128, with a 16-bit cache, and has 4.2 GB left for the cache. About how many 4,000-token conversations fit at once?

- A. About 16
- B. About 2
- C. About 4
- D. About 8

#### [day1#lab-1-2] Concurrency limit at three context lengths

Lab 1.2 · 25 min

Predict how many sequences fit in the cache at 2,000, 4,000 and 8,000 tokens, then check against the engine's block count and a saturation run.

[D1_2_predict_concurrency_limit.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D1_2_predict_concurrency_limit.ipynb)

- [ ] Cache bytes per token for the base model (4 min)
- [ ] Memory left for the cache after weights and reserve (5 min)
- [ ] Predict the concurrency limit at three context lengths (5 min)
- [ ] Read the engine's blocks and run it to saturation (6 min)
- [ ] Compare, explain the gap, check the 25% band (5 min)
- [ ] ✱ Stretch: Predict the limit again with 8-bit weights

### Session 3: Goodput against an objective

#### [day1#s3] Goodput against an objective

*Session 3*

- Measure time to first token and time per output token from a streamed response
- Compute goodput against two per-request objectives
- Break a request's time into queue, prefill and decode

#### [day1#three-clocks] Four clocks per request, and the objective uses three of them

*What a visitor feels*

- **TTFT.** From sending to the first token: any upload, then queue plus prefill.
- **TPOT.** Per token after the first: the reading pace.
- **Longest stall.** The largest gap between two tokens: the freeze.
- **End to end.** The whole answer; not in the objective.

#### [day1#goodput-defined] Goodput counts only the requests that met the whole objective

*The measure of this course*

```python
# file: Lab 1.3 · task 4
met = (df.ttft_s <= 2.0) & (df.tpot_s <= 0.150) & (df.stall_s <= 1.0)
goodput = met.sum() / (df.end.max() - df.start.min())
attainment = met.mean()
```

Throughput counts every finished request. Goodput counts the visitors who were served well.

#### [day1#measure-from-the-stream] A client measures from the stream it receives

*Lab 1.3 · tasks 1 and 2*

```python
# file: Lab 1.3
data: {"choices": [{"index": 0, "text": "tok "}]}
data: {"choices": [{"index": 0, "text": "tok "}]}
data: [DONE]

ttft = (times[0] - t0) / scale
tpot = (times[-1] - times[0]) / (len(times) - 1) / scale
```

#### [day1#closed-loop-users] Concurrency is the number of visitors waiting for an answer at once

*Load*

- **Closed loop**: each user sends the next request when the last answer ends, so the load is a number of users, not a rate.
- Easy to control and repeat, which is why curves are drawn over it.
- Kinder than reality: when the engine slows, the users slow too.

#### [day1#throughput-rose-goodput-fell] From 8 to 24 users, throughput rose and goodput fell to zero

*Lab 1.3 · tasks 3 and 4*

| Users | Throughput (req/s) | Goodput (req/s) | Attainment |
| --- | --- | --- | --- |
| 2 | 0.20 | 0.20 | 100% |
| 8 | 0.28 | 0.09 | 31% |
| 24 | 0.51 | 0.00 | 0% |

The baseline as deployed, through the HTTP stand-in, quick-mode run.

#### [day1#where-the-time-went] At 24 users a request spends 28 seconds in the queue

*Lab 1.3 · task 5*

| Users | Queue (s) | Prefill (s) | Decode (s) |
| --- | --- | --- | --- |
| 2 | 0.00 | 0.67 | 12.68 |
| 8 | 0.14 | 1.33 | 17.70 |
| 24 | 28.19 | 3.44 | 19.05 |

Mean seconds per request, from the engine's own times in the simulator.

#### [day1#photo-upload-dominates] In the photo pilot, the upload takes 86% of the time to first token

*Stage timings from the client's clock*

| Stage of a photo question | Median (s) | Share |
| --- | --- | --- |
| Phone prepares the photo | 0.00 | 0% |
| Upload | 3.85 | 86% |
| Server decodes and resizes | 0.14 | 3% |
| Queue | 0.11 | 3% |
| Image encoding and prefill | 0.45 | 9% |

Time to first token: median 4.55 s, p90 8.21 s; 1% of photo questions meet 2.0 s. Synthetic: 3.4 MB median photo, 7 Mbit/s median uplink.

#### [day1#resize-on-the-phone] The model keeps 1,024 pixels, so the phone should send only those

*The fix, confirmed by a rerun*

- The pilot model resizes every photo to 1,024 pixels on the long side; most of the 12 megapixels cross the network to be thrown away.
- The app resizes and recompresses before upload: 3.4 MB becomes 0.24 MB.
- The model receives an image of the same size, so its tokens and prefill do not change.

| Median or share | Before | After |
| --- | --- | --- |
| Phone prepares (s) | 0.00 | 0.07 |
| Upload (s) | 3.85 | 0.36 |
| Server resizes (s) | 0.14 | 0.01 |
| TTFT p90 (s) | 8.21 | 1.26 |
| Meet 2.0 s | 1% | 100% |

#### [day1#same-table-other-decision] Athar's guides wait too: which stage do you work on?

*Your turn · a parallel case*

| Stage of a photo question | Median (s) | Share |
| --- | --- | --- |
| Phone prepares the photo | 0.08 | 2% |
| Upload | 0.16 | 5% |
| Server decodes and resizes | 0.01 | 0% |
| Queue | 2.44 | 80% |
| Image encoding and prefill | 0.45 | 13% |

Guides ask about artefact photos at the evening peak, over venue Wi-Fi; their app already sends 0.38 MB photos. TTFT p90 5.22 s; 10% meet 2.0 s.

#### [day1#checkpoint-3] Which number goes in the report?

*Checkpoint*

**Question:** A run at 16 users shows a mean TTFT of 1.7 s, a p90 TTFT of 9 s, a TPOT p90 of 95 ms, no stall over 1 s and 0.44 requests per second. Hanadi asks whether the objective is met. What do you report?

- A. Yes: the mean TTFT is under 2 s
- B. Yes: it serves 0.44 requests per second
- C. The share of requests that met all three, which the TTFT p90 says is under 90%
- D. Yes: the TPOT p90 is under 150 ms

#### [day1#lab-1-3] The baseline at three concurrency levels

Lab 1.3 · 25 min

Load the as-deployed Musamir through a streaming HTTP endpoint at three concurrency levels and compute goodput against the objectives.

[D1_3_load_baseline_three.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D1_3_load_baseline_three.ipynb)

- [ ] Start the endpoint and stream one request (4 min)
- [ ] Measure TTFT, TPOT and end-to-end time from the stream (6 min)
- [ ] Load at 2, 8 and 24 concurrent users (6 min)
- [ ] Goodput against the objectives (5 min)
- [ ] Where the time went: queue, prefill, decode (4 min)
- [ ] ✱ Stretch: Point the same load generator at a real engine on a notebook GPU

### Session 4: Curves, not single numbers

#### [day1#s4] Curves, not single numbers

*Session 4*

- Build a goodput curve over concurrency with repeated runs
- Read the knee of a curve and state a capacity with its variance
- Judge whether a benchmark is valid

#### [day1#one-run-is-an-anecdote] At 8 users, the p90 TTFT of three runs spreads by 0.7 seconds

*Variance*

- **1.96 s** mean p90 TTFT of three runs
- **± 0.68 s** standard deviation between them

Near the knee, small changes in which requests arrive together decide who waits. A single run could have said 1.3 s or 2.6 s.

#### [day1#baseline-curve] The baseline's goodput peaks at 4 users and collapses by 12

*Lab 1.4 · task 4*

Figure: Mean of three runs per point, error bars one standard deviation. Synthetic: the course simulator.

#### [day1#the-knee] Seven points, three runs each, raw rows kept

*Lab 1.4 · task 2*

| Users | Goodput | ± sd | Throughput | TTFT p90 (s) | TPOT p90 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.079 | 0.003 | 0.080 | 1.41 | 75 |
| 2 | 0.120 | 0.010 | 0.146 | 1.39 | 84 |
| 4 | 0.149 | 0.020 | 0.255 | 1.42 | 99 |
| 6 | 0.140 | 0.033 | 0.333 | 1.51 | 110 |
| 8 | 0.116 | 0.032 | 0.399 | 1.96 | 122 |
| 12 | 0.020 | 0.012 | 0.441 | 11.42 | 128 |
| 16 | 0.006 | 0.009 | 0.443 | 20.81 | 132 |

#### [day1#benchmark-validity] A curve is evidence only if someone else can rerun it

*Benchmark validity*

- The same trace, or a named sample of it, with its seed.
- Warm-up dropped: the first 10% of each run here.
- Enough requests per point to fill the engine many times.
- Three or more repeats, and the spread reported.

- The raw per-request rows kept beside the plot.
- The configuration and the engine version recorded.
- The objectives written on the plot.
- Closed or open loop stated.

#### [day1#closed-loop-hides-the-queue] Replayed at real arrival times, the peak queues for minutes

*Lab 1.4 · stretch*

- **Closed loop, 16 users.** 0.44 requests per second, p90 TTFT 21 s. The users waited for each other.
- **Open loop, two minutes of the peak.** 0.45 requests per second served while 2.13 arrive; p90 TTFT 372 s and growing.

> **What goes wrong.** A closed-loop test at "high concurrency" can look survivable while real arrivals would grow the queue without bound.

#### [day1#capacity-claim] A capacity claim names the point, its spread and the objectives

*Lab 1.4 · task 5*

> **Claim.** The baseline serves at most 0.15 ± 0.02 good requests per second (mean ± sd of 3 runs) at 4 concurrent users, under TTFT ≤ 2.0 s, TPOT ≤ 150 ms and no stall over 1.0 s; beyond it goodput falls while throughput keeps rising.

The evening peak brings 2.13 requests per second: fourteen times this.

#### [day1#checkpoint-4] Is the new setting better?

*Checkpoint*

**Question:** At 8 users, one run with a new engine setting gives a p90 TTFT of 1.5 s against the baseline's 1.96 s mean (sd 0.68 over three runs). What do you conclude?

- A. Nothing yet: the gap is inside the baseline's spread; run three repeats
- B. The setting cuts TTFT by a quarter
- C. It is better because throughput did not drop
- D. The setting is worse: the result is noisy

#### [day1#lab-1-4] A seven-point goodput curve

Lab 1.4 · 25 min

Combine supplied runs at six concurrency levels with three repeats of your own at a seventh, plot the curve with its spread and state the capacity.

[D1_4_build_seven_point.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D1_4_build_seven_point.ipynb)

- [ ] Load the supplied runs and their raw rows (3 min)
- [ ] Aggregate each point with mean and spread (6 min)
- [ ] Add your own point with three repeats (6 min)
- [ ] Plot goodput and p90 latencies over concurrency (5 min)
- [ ] Find the knee and write the capacity claim (5 min)
- [ ] ✱ Stretch: Compare closed-loop users with open-loop arrivals at the same rate

#### [day1#day-1-recap] Predict first, measure with spread, judge by goodput

*Day 1 · three things to keep*

- **Two phases.** Prefill waits on arithmetic, decode on memory. Two lines of arithmetic predict both.
- **The cache decides.** 3.6 GB of cache holds about 12 conversations of 2,000 tokens: that, not compute, limits the baseline.
- **Curves with spread.** Goodput over concurrency, three runs a point, raw rows kept: 0.15 at the knee.

#### [day1#tomorrow-inside-the-engine] Tomorrow: three broken deployments, and the settings that fix them

*Next*

- Keep your sizing sheet and your curve: Day 2 starts from them.
- Read the vLLM page on chunked prefill and preemption (in the sources) before the session.
- Think of one prompt you know whose beginning changes on every request.

## Day 2: Scheduling and caching in the engine

Three seeded faults, then the engine settings and caches that fix them

#### [day2#title] Scheduling and caching in the engine

*SDAIA Academy · Inference Performance and Cost Engineering*

Day 2: Three seeded faults, then the engine settings and caches that fix them

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-401

#### [day2#yesterday-in-one-slide] The baseline serves 0.15 good requests per second; the peak brings 2.13

*Where Day 1 left us*

- **0.15** good requests per second at the knee, 4 users
- **~12** conversations of 2,000 tokens the cache holds
- **18%** of requests freeze over 1 s already at 2 users

Today: the scheduler and the cache, setting by setting, each change judged by a rerun against the objective.

#### [day2#review-checkpoint] Yesterday's claim, read again

*Review*

**Question:** Kenan reads the Day 1 curve and says: "throughput keeps rising to 16 users, so the engine has headroom; let us admit more visitors." What is wrong with the argument?

- A. The curve has too few runs to say anything
- B. A closed loop cannot measure throughput
- C. Throughput counts requests finished, met or not; past 4 users goodput falls
- D. Throughput only counts at the evening peak

#### [day2#three-faults] Yamen hands over three deployments, each broken in one setting

*The day's case*

| Deployment | What the visitors see | Lab |
| --- | --- | --- |
| Fault 1 | Long waits before the first word, even at 6 visitors | 2.1 |
| Fault 2 | Waits and answers that restart, the card's memory full | 2.1 |
| Fault 3 | Answers that freeze mid-sentence on document evenings | 2.2 |

The outcome asks for three diagnoses, each with its evidence and a rerun that meets the objective.

#### [day2#agenda] Scheduling and caching in the engine

*Today*

- **Session 1 · Continuous batching and admission**: How the scheduler admits sequences, and how queueing and preemption look in the metrics.
- **Session 2 · Long prompts slow short ones**: A long prefill stalls every decode in the batch unless it is cut into chunks.
- **Session 3 · Prefix caching starts with prompt layout**: Reuse needs identical leading blocks, and reuse across tenants must be blocked.
- **Session 4 · Semantic caching and wrong answers**: A similar question is not the same question: every semantic hit is a bet.

### Session 1: Continuous batching and admission

#### [day2#s1] Continuous batching and admission

*Session 1*

- Explain iteration-level scheduling and the sequence cap
- Tell queueing from preemption by their metrics
- Choose a concurrency cap from a tail-latency and throughput plot

#### [day2#iteration-level-scheduling] The engine rebuilds the batch at every step, not every request

*Continuous batching*

1. **Step** one forward pass
2. **Finished?** leave the batch at once
3. **Room?** waiting requests join
4. **Next step** a new batch

- A short answer does not wait for the longest answer in its batch to finish.
- Two limits decide who joins: the **sequence cap** (`max_num_seqs`) and the **token budget** per step (`max_num_batched_tokens`).
- A third limit is not a setting: free cache blocks.

#### [day2#engine-gauges] Four gauges tell you what the scheduler is doing

*Reading an engine*

| Gauge | vLLM metric | Reads as |
| --- | --- | --- |
| Running | `vllm:num_requests_running` | Sequences in the batch now |
| Waiting | `vllm:num_requests_waiting` | Requests not yet admitted |
| Cache usage | `vllm:kv_cache_usage_perc` | Share of blocks in use |
| Preemptions | `vllm:num_preemptions_total` | Sequences evicted and recomputed so far |

#### [day2#queueing-signature] Fault 1 runs at its cap with a queue and an empty cache

*Lab 2.1 · task 1*

- **1.93** running, mean (cap 2)
- **3.83** waiting, mean
- **19%** cache in use

> **Queueing.** The engine refuses work it has room for: the cap is the limit, not the card.

#### [day2#cap-sweep] Raising the cap from 2 to 6 takes the TTFT tail from 35 s to 1.5 s

*Lab 2.1 · tasks 2 and 3*

| Cap | Throughput (req/s) | p90 TTFT (s) | Attainment |
| --- | --- | --- | --- |
| 2 | 0.15 | 35.2 | 0% |
| 4 | 0.26 | 11.7 | 0% |
| 6 | 0.35 | 1.5 | 96% |
| 8 | 0.35 | 1.5 | 96% |
| 12 | 0.35 | 1.5 | 96% |

Past the number of users, a closed loop never fills a larger cap. Synthetic: the course simulator.

#### [day2#preemption-signature] Fault 2 admits everyone, fills the cache and throws work away

*Lab 2.1 · task 4*

- **712** cache blocks at 82% utilisation
- **87%** cache in use, mean
- **10.7** preemptions per 100 requests

A preempted sequence loses its cache and is recomputed from the start when room returns.

#### [day2#two-faults-opposite-fixes] Same symptom, opposite fixes: read the cache before you touch the cap

*Lab 2.1 · task 5*

- **Queueing.** Running at the cap, a queue, cache with room. **Raise the cap.**
- **Preemption.** Cache nearly full, preemptions rising. **Lower the cap or give the cache more memory.**

> **The rerun.** Cap 8 and 90% utilisation: 96% of requests meet the objective at 6 users.

#### [day2#admission-by-tokens] The engine admits a request when its next chunk fits, not its whole answer

*Admission*

- Admission checks the blocks for the prompt, or the next chunk of it, plus a small reserve.
- The answer's growth is not reserved: that is why the cache can run out mid-answer.
- A cap sized from Lab 1.2's arithmetic keeps answers from outgrowing the cache.

#### [day2#checkpoint-1] Which fault is it?

*Checkpoint*

**Question:** A deployment misses the objective at 10 visitors. Its gauges: 10 running, 6 waiting, cache 97% in use, preemptions climbing. Yamen wants to double the sequence cap. What do you tell him?

- A. Do not raise the cap: the cache is the limit; lower the cap or free memory for the cache
- B. Raise it: six requests are waiting
- C. Keep the cap and halve the token budget
- D. Rerun with a different load generator first

#### [day2#lab-2-1] The concurrency cap

Lab 2.1 · 25 min

Diagnose two seeded deployments, one queueing and one preempting under cache pressure, by varying the sequence cap and plotting tail latency against throughput.

[D2_1_vary_concurrency_cap.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D2_1_vary_concurrency_cap.ipynb)

- [ ] Run the queueing deployment and read its metrics (5 min)
- [ ] Vary the sequence cap over five values (6 min)
- [ ] Plot p90 TTFT against throughput (5 min)
- [ ] Push into cache pressure and count preemptions (5 min)
- [ ] Choose the cap and write both diagnoses (4 min)
- [ ] ✱ Stretch: Admit by a token budget instead of a sequence count

### Session 2: Long prompts slow short ones

#### [day2#s2] Long prompts slow short ones

*Session 2*

- Recognise prefill interference in time per output token
- Tune chunked prefill with a token budget
- Confirm a fix with a rerun against the objective

#### [day2#one-step-many-requests] A long prompt processed whole stops every answer in the batch

*Interference*

- Prefill and decode share each step. A 5,000-token prefill makes one step last over a second.
- Every running answer waits for that step before its next token: a **stall**.
- Averaged over 150 tokens, a one-second stall barely moves TPOT. The visitor sees the whole second.

#### [day2#chunked-prefill] Decode first, then fill the step's token budget with pieces of prompts

*Chunked prefill*

1. **Decodes** one token per running answer
2. **Budget left?** max_num_batched_tokens
3. **Prompt chunk** the next piece that fits

Every step stays short, so no answer stalls. The price: a long prompt needs several steps to its first token.

#### [day2#interference-measured] With chunking off, a quarter of requests meet the objective

*Lab 2.2 · task 2*

- **27%** attainment, 5 users, 35% documents
- **2.2 s** p90 of the chat answers' longest stall
- **114 ms** p90 chat TPOT: within its objective

TPOT alone says nothing is wrong. The stall says every chat answer froze.

#### [day2#budget-trade-off] A smaller budget protects chat and makes documents wait

*Lab 2.2 · tasks 3 and 4*

| Token budget | Attainment | Chat stall p90 (s) | Document TTFT p90 (s) |
| --- | --- | --- | --- |
| Off, 8,192 | 27% | 2.21 | 2.12 |
| 512 | 93% | 0.18 | 2.47 |
| 1,024 | 96% | 0.35 | 2.20 |
| 2,048 | 96% | 0.68 | 2.09 |

Synthetic: the course simulator, one run per budget. The rerun at 1,024 meets the objective.

#### [day2#third-diagnosis] A diagnosis is a symptom, a metric, a change and a rerun

*Lab 2.2 · task 5*

> **Fault 3.** Chat answers froze for 2.2 s at p90 while documents were prefilled whole (chunked prefill off). With chunked prefill and a budget of 1,024 tokens, the stall p90 is 0.35 s and 96% of requests meet the objective.

The same four parts for Faults 1 and 2 make the three diagnoses of the outcome.

#### [day2#chat-first] A priority for chat moves the wait, it does not remove it

*Stretch*

- A chat-first policy lets waiting chat requests go ahead of waiting documents.
- Chat TTFT falls; document TTFT rises. The sum of the work is the same.
- vLLM exposes a priority policy (`--scheduling-policy priority`) with a priority per request.

#### [day2#checkpoint-2] Where did the freeze come from?

*Checkpoint*

**Question:** Asrar reports that English chat answers freeze for about a second, several times an evening, while TPOT p90 stays at 110 ms. Chunked prefill is on with a budget of 8,192 tokens. What is the most likely cause?

- A. Decoding is too slow for English
- B. The cache is full and answers are preempted
- C. The sequence cap is too low
- D. The budget is so large that a whole long prompt still fits in one step

#### [day2#lab-2-2] Long documents and chat, chunked or not

Lab 2.2 · 25 min

Mix long-document and chat traffic, measure chat TPOT with chunked prefill off and with three token budgets, and confirm the fix.

[D2_2_mix_long_document.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D2_2_mix_long_document.ipynb)

- [ ] Load the mixed trace (3 min)
- [ ] Chunked prefill off: chat TPOT and document TTFT (6 min)
- [ ] Chunked prefill on at three token budgets (7 min)
- [ ] Plot both latencies over the budget (5 min)
- [ ] Write the diagnosis and rerun to the objective (4 min)
- [ ] ✱ Stretch: Give chat requests priority over documents

### Session 3: Prefix caching starts with prompt layout

#### [day2#s3] Prefix caching starts with prompt layout

*Session 3*

- Lay out a prompt so that its stable part comes first
- Measure the prefix-cache hit rate and its effect on time to first token
- Prove that no cache block is shared between tenants

#### [day2#prefix-hashing] A block is reused only if every token before it is identical

*Prefix caching*

- Each full block of 16 tokens is hashed together with the hash of the block before it.
- A new request looks up its blocks in order and stops at the first miss.
- Reused blocks skip prefill: a shorter time to first token and less work per request.

> **So:** one changing token at the start of a prompt makes every block after it a miss.

#### [day2#musamir-prompt] Musamir's prompt starts with the one part that changes every time

*The prompt as deployed*

1. **header** request number, name, second
2. **base** 640 tokens, all tenants
3. **tenant** organiser's rules
4. **document** when asked
5. **history** earlier turns
6. **question** this turn

Hit rate as deployed: 0%. The engine recomputes 2,092 prompt tokens per request on average.

#### [day2#stable-first] Stable parts first: the median first token falls from 432 ms to 90 ms

*Lab 2.3 · tasks 2 and 3*

| Layout | Hit rate | Prompt tokens computed | TTFT p50 |
| --- | --- | --- | --- |
| As deployed | 0% | 2,092 | 0.432 s |
| base, tenant, document, history, header, question | 55% | 894 | 0.090 s |

Six users, the tuned engine of this morning. Synthetic: the course simulator.

#### [day2#cross-tenant] The shared base text lets one tenant reuse another tenant's blocks

*Lab 2.3 · task 4*

- Unsalted, **6,000 blocks** were reused across tenants in one run.
- A reused block answers faster, so timing can tell an attacker whether another tenant's prompt starts with a given text.
- Rawya's rule: no cache block crosses tenants.

#### [day2#cache-salt] A salt per tenant costs almost no reuse and passes the isolation test

*Lab 2.3 · task 5*

- **55.3%** hit rate, salted by tenant
- **0** blocks reused across tenants
- **pass** isolation test, every request

```json
{"prompt": "...", "cache_salt": "tenant-sawt"}
```

#### [day2#checkpoint-3] Where does the name go?

*Checkpoint*

**Question:** Product wants Musamir to greet each visitor by first name, so the name must be in the prompt. Prefix caching and a salt per tenant are on. Where should the name go?

- A. At the start: the salt keeps it private
- B. After everything shared, next to the question
- C. Inside the tenant's rules, so each tenant handles it
- D. Nowhere: names cannot be used with a prefix cache

#### [day2#lab-2-3] Prompt layout and tenant isolation

Lab 2.3 · 25 min

Reorder Musamir's prompt for reuse, measure the hit rate and the gain, find the cross-tenant hits and remove them with a per-tenant salt.

[D2_3_reorder_prompt_reuse.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D2_3_reorder_prompt_reuse.ipynb)

- [ ] Hit rate of the prompt as deployed (4 min)
- [ ] Reorder the prompt layout (7 min)
- [ ] Hit rate and TTFT after the change (5 min)
- [ ] Find the cross-tenant hits (4 min)
- [ ] Salt by tenant and pass the isolation test (5 min)
- [ ] ✱ Stretch: Hit rate against cache size

### Session 4: Semantic caching and wrong answers

#### [day2#s4] Semantic caching and wrong answers

*Session 4*

- Measure hit rate and wrong-answer rate of a semantic cache
- Compare the rates by language
- Keep personal data out of cached entries

#### [day2#semantic-cache-offer] A semantic cache returns a stored answer when a question looks similar enough

*The vendor's offer*

- Each question is embedded; the nearest stored question above a **threshold** returns its stored answer.
- A hit skips the model entirely: no prefill, no decode.
- Unlike the prefix cache, a hit can be a **wrong answer**: the model never sees the new question.

#### [day2#similar-not-same] Questions that differ in one day look more alike than true paraphrases

*Lab 2.4 · task 1*

| Kind of pair | Mean similarity | Highest |
| --- | --- | --- |
| Paraphrase, same answer | 0.61 | 0.83 |
| One slot changed (day, venue, ticket) | 0.63 | 0.93 |
| Other intent | 0.38 | 0.56 |

240 Arabic and English pairs, character trigram cosine. Synthetic pairs.

#### [day2#threshold-table] At every threshold, most hits are wrong answers

*Lab 2.4 · task 2*

| Threshold | Hit rate | Wrong among hits | Wrong per 1,000 |
| --- | --- | --- | --- |
| 0.70 | 22.9% | 58% | 133 |
| 0.80 | 9.6% | 74% | 71 |
| 0.90 | 3.8% | 100% | 38 |

Raising the threshold removes the paraphrases first: what is left is the near-identical wrong pairs.

#### [day2#by-language] Neither language meets one wrong answer in a thousand

*Lab 2.4 · task 3*

| Threshold | Arabic, wrong per 1,000 | English, wrong per 1,000 |
| --- | --- | --- |
| 0.70 | 125 | 142 |
| 0.80 | 67 | 75 |
| 0.90 | 50 | 25 |

At 0.90 Arabic is twice as wrong as English: one threshold does not fit both languages.

#### [day2#personal-data] Mask before the lookup, refuse to store what carries personal data

*Lab 2.4 · task 4*

- Phones (`05` and 8 digits) and bookings (`LW-` and 6 digits) are masked before the similarity is computed.
- An entry whose question carried any is never stored: 42 of 240 refused.
- A cached answer is served to other visitors; one written for a booking must not be.

#### [day2#no-semantic-cache] The decision: no semantic cache on text alone

*Lab 2.4 · task 5*

> **Decision.** No threshold keeps wrong answers under 1 in 1,000 in both languages. Keep the prefix cache, which never returns a wrong answer.

Stretch: keyed on the event, day, venue and ticket as well as the text, the cache hits 9.6% at 0.70 with no wrong answer in the sample.

#### [day2#checkpoint-4] Will a higher threshold fix it?

*Checkpoint*

**Question:** The vendor answers Hanadi: "If wrong answers worry you, raise the threshold to 0.95." What does Lab 2.4's data say about that?

- A. It works: wrong answers fall to zero long before the hits do
- B. At 0.90 the remaining hits were all wrong; a higher threshold leaves few hits, mostly wrong
- C. It works for English only
- D. It works once personal data are masked

#### [day2#lab-2-4] Semantic cache thresholds

Lab 2.4 · 25 min

Score 240 Arabic and English paraphrase pairs at three similarity thresholds for hits and wrong answers, then keep personal data out of the cache.

[D2_4_measure_hit_rate.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D2_4_measure_hit_rate.ipynb)

- [ ] Embed the pairs with character n-grams (5 min)
- [ ] Hit rate and wrong answers at three thresholds (7 min)
- [ ] The same rates by language (4 min)
- [ ] Mask, then refuse to cache personal data (5 min)
- [ ] Choose a threshold, or no semantic cache (4 min)
- [ ] ✱ Stretch: Key the cache on the event and the date as well as the text

#### [day2#day-2-recap] Read the gauges, keep steps short, put what is stable first

*Day 2 · three things to keep*

- **Gauges before settings.** Queueing and preemption look alike in latency and need opposite fixes.
- **Short steps.** Chunked prefill at 1,024 tokens: from 27% to 96% of requests meeting the objective.
- **Stable first, salted.** TTFT p50 from 432 ms to 90 ms, no block across tenants, no semantic cache.

#### [day2#tomorrow-precision-and-models] Tomorrow: fewer bits, a draft model and a smaller model, each against a quality floor

*Next*

- Keep the three diagnoses and the isolation test: the capstone report starts from them.
- Read the abstract of AWQ or LLM.int8() (in the sources).
- Think of one question you would not trust a smaller model to answer in Arabic.

## Day 3: Cheaper tokens, same quality

Smaller numbers, smaller models, guessed tokens: each re-evaluated in two languages

#### [day3#title] Cheaper tokens, same quality

*SDAIA Academy · Inference Performance and Cost Engineering*

Day 3: Smaller numbers, smaller models, guessed tokens: each re-evaluated in two languages

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-401

#### [day3#where-day-2-left-us] A tuned engine still drowns at the peak in 16-bit weights

*Where Day 2 left us*

- **0.49** good requests per second at 8 users, tuned on Day 2
- **2.1** requests per second arriving in the first minutes of the peak
- **0%** of those requests meet the objective in 16 bits

Day 2 tripled the baseline's goodput. The memory is still full of weights.

#### [day3#review-checkpoint] Yesterday's caches, read again

*Review*

**Question:** A colleague proposes to put the visitor's booking reference at the start of Musamir's prompt "so the cache can find the visitor's earlier turns faster". What happens to the prefix cache?

- A. It keeps working, because the salt isolates visitors
- B. Reuse rises, because each visitor's history is found at once
- C. Nothing, as long as the reference is masked
- D. The shared instructions are no longer reused across visitors

#### [day3#two-numbers-per-change] Every change is judged on goodput and on quality in both languages

*The rule for today*

- **Quality floors.** Arabic 0.73 and English 0.81: the base model's scores less three points. A variant passes when it is at or above both.
- **The evaluation set.** 600 items, 300 per language, a third with long event-terms documents. Recorded outcomes, synthetic.

Scores are reported with 95% intervals; comparisons on the same items are paired.

#### [day3#agenda] Cheaper tokens, same quality

*Today*

- **Session 1 · Quantising weights**: Fewer bytes per weight free memory for the cache and speed up decode, if quality holds.
- **Session 2 · Cache precision buys concurrency**: An 8-bit cache doubles the blocks; long contexts pay for it first.
- **Session 3 · Speculative decoding**: A small model guesses, the large one checks: faster when the engine is quiet, a cost when it is full.
- **Session 4 · Distilled models and routing to them**: Send the easy requests to a small model and keep the base for the hard ones.

### Session 1: Quantising weights

#### [day3#s1] Quantising weights

*Session 1*

- Choose a weight format the accelerator supports
- Measure memory, goodput and quality of each precision
- Report quality by language with intervals

#### [day3#formats] Weights in 8 or 4 bits halve or quarter the bytes each step reads

*Number formats*

| Format | Bytes per weight | Runs on the L4 | How |
| --- | --- | --- | --- |
| BF16 | 2 | Yes | As trained |
| FP8 | 1 | Yes, FP8 tensor cores | Per-channel scales, often without calibration data |
| INT4 (weight-only) | 0.53 | Yes, unpacked to 16 bits | Group scales, calibration: AWQ, GPTQ |

4-bit counts 0.53 bytes: the scales take the rest. The L4 sheet's FP8 rate is twice its BF16 rate.

#### [day3#memory-freed] 8-bit weights free 8.2 GB, and the cache more than triples

*Lab 3.1 · task 1*

| Item | Value |
| --- | --- |
| BF16: 16.4 GB of weights | 1,525 blocks |
| FP8: 8.2 GB | 5,001 blocks |
| INT4: 4.3 GB | 6,635 blocks |

In conversations of 2,000 tokens: 12, 40 and 53.

#### [day3#peak-by-format] At the peak, 16-bit weights serve no one and 8-bit weights serve most

*Lab 3.1 · task 2*

| Weights | Goodput (req/s) | Attainment | p90 TTFT |
| --- | --- | --- | --- |
| BF16 | 0.00 | 0% | 312 s |
| FP8 | 1.59 | 82% | 3.7 s |
| INT4 | 1.89 | 96% | 1.4 s |

Open loop, first five minutes of the evening peak (2.1 requests per second), one L4, cap 32. Synthetic: the course simulator.

#### [day3#quality-by-language] 4-bit weights lose seven points in Arabic and almost none in English

*Lab 3.1 · tasks 3 and 4*

| Weights | Arabic (95% interval) | English (95% interval) | Both floors |
| --- | --- | --- | --- |
| BF16 | 0.763 (0.712 to 0.808) | 0.843 (0.798 to 0.880) | Pass |
| FP8 | 0.757 (0.705 to 0.802) | 0.837 (0.791 to 0.874) | Pass |
| INT4 | 0.693 (0.639 to 0.743) | 0.833 (0.787 to 0.871) | Fails Arabic |

Floors: Arabic 0.73, English 0.81. 300 items per language, recorded outcomes, synthetic.

#### [day3#serving-table] One serving table collects every variant you try this week

*Lab 3.1 · task 5*

| Variant | Weights | Blocks | Attainment | Arabic | English |
| --- | --- | --- | --- | --- | --- |
| BF16 weights, BF16 cache | 16.4 GB | 1,525 | 0% | 0.763 | 0.843 |
| FP8 weights, BF16 cache | 8.2 GB | 5,001 | 82% | 0.757 | 0.837 |
| INT4 weights, BF16 cache | 4.3 GB | 6,635 | 96% | 0.693 | 0.833 |

Labs 3.2 to 3.4 add rows. The capstone's design document starts from this table.

#### [day3#checkpoint-1] Ship the fastest variant?

*Checkpoint*

**Question:** Yamen wants 4-bit weights: 96% attainment at the peak, and "English quality is unchanged". The Arabic score is 0.693 against a floor of 0.73. What do you recommend?

- A. Ship it: the Arabic interval reaches above the floor
- B. Ship it: the average of both languages is 0.76
- C. Keep 16-bit weights and buy a larger card
- D. Serve 8-bit weights, which pass both floors, and close the gap to the objective with the cache

#### [day3#lab-3-1] Two weight precisions

Lab 3.1 · 25 min

Compare two weight precisions, 8-bit and 4-bit, against the 16-bit base model on memory, goodput at the peak and Arabic and English task scores with intervals.

[D3_1_compare_two_weight.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D3_1_compare_two_weight.ipynb)

- [ ] Memory and cache left for each precision (5 min)
- [ ] Goodput at the peak for each precision (6 min)
- [ ] Arabic and English scores with intervals (7 min)
- [ ] Check each against the floors (4 min)
- [ ] Write the rows of the serving table (3 min)
- [ ] ✱ Stretch: Which formats the T4 of a notebook can run

### Session 2: Cache precision buys concurrency

#### [day3#s2] Cache precision buys concurrency

*Session 2*

- Compute the blocks gained by a lower cache precision
- Measure the concurrency and goodput gained
- Measure long-context accuracy lost, per language, as a paired difference

#### [day3#cache-bytes] An 8-bit cache stores each key and value in one byte: twice the blocks

*Lab 3.2 · task 1*

- **147,456** bytes per token, 16-bit cache
- **73,728** bytes per token, 8-bit cache
- **10,002** blocks with 8-bit weights and cache

2 × 36 layers × 8 key-value heads × 128 × bytes per value.

#### [day3#cache-gain] With an 8-bit cache, one card meets the objective in the first minutes of the peak

*Lab 3.2 · task 2*

| Cache | Mean running | Mean waiting | Cache in use | Attainment | p90 TTFT |
| --- | --- | --- | --- | --- | --- |
| BF16 | 23.8 | 1.24 | 36% | 82% | 3.7 s |
| FP8 | 16.4 | 0.00 | 14% | 100% | 0.12 s |

Per million good output tokens: 4.69 SAR before, 3.77 after (case prices). The 16-bit cache was only a third full: the gain came from reading half the bytes every step.

#### [day3#long-context-loss] The loss is all in long contexts, and mostly in Arabic

*Lab 3.2 · task 3*

| Items | 16-bit cache | 8-bit cache | Change |
| --- | --- | --- | --- |
| Arabic, short | 0.777 | 0.777 | 0.0 |
| Arabic, long (103 items) | 0.718 | 0.670 | −4.8 points |
| English, short | 0.848 | 0.848 | 0.0 |
| English, long (109 items) | 0.817 | 0.807 | −0.9 points |

Both with 8-bit weights. Long items carry an event-terms document of 6,000 to 10,000 tokens.

#### [day3#paired-difference] Paired on the same items, the Arabic loss is real; separate intervals hide it

*Lab 3.2 · task 4*

- **Paired difference, Arabic long.** −4.8 points, 95% bootstrap interval −9.7 to −1.0: the whole interval below zero.
- **Two separate intervals.** 0.625 to 0.796 and 0.574 to 0.753: they overlap, and a reader would call it noise.

```python
d = items[(items.lang == "ar") & (items.context == "long")]
S.paired_diff(list(d.w_fp8), list(d.kv_fp8))   # (-0.0485, (-0.0971, -0.0097))
```

#### [day3#cache-decision] The 8-bit cache passes the floors; the long-document loss is the Director's call

*Lab 3.2 · task 5*

> **Decision.** Arabic 0.740 and English 0.833 overall, both above the floors, and one card meets the objective at the peak. Long Arabic documents lose about five points: Hanadi accepts that loss, or long-document requests keep a 16-bit cache in a pool of their own.

Stretch: a second pool for long contexts keeps Arabic at 0.757, and costs a card.

#### [day3#checkpoint-2] Is the loss real?

*Checkpoint*

**Question:** On 103 long Arabic items, the 16-bit cache scores 0.718 (interval 0.625 to 0.796) and the 8-bit cache 0.670 (0.574 to 0.753). Kenan says the intervals overlap, so there is no loss. What is the right test?

- A. A paired difference on the same items, which here lies wholly below zero
- B. Run each cache three times and average
- C. Pool the long and short items to get more data
- D. Accept Kenan's reading: overlap means no difference

#### [day3#lab-3-2] Lower cache precision

Lab 3.2 · 25 min

Switch the cache to 8 bits, measure the concurrency and goodput gained and the long-context accuracy lost in each language.

[D3_2_lower_cache_precision.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D3_2_lower_cache_precision.ipynb)

- [ ] Blocks at 16-bit and 8-bit cache (4 min)
- [ ] Concurrency and goodput gained (6 min)
- [ ] Long-context accuracy by language (7 min)
- [ ] Paired difference with an interval (5 min)
- [ ] Decide, with the floor (3 min)
- [ ] ✱ Stretch: Keep the 16-bit cache only for long-context requests

### Session 3: Speculative decoding

#### [day3#s3] Speculative decoding

*Session 3*

- Compute the expected tokens per step from an acceptance rate
- Measure acceptance and time per output token at low and high load
- Decide when to switch speculation on

#### [day3#draft-and-verify] A small model drafts a few tokens; the base model checks them in one step

*Speculative decoding*

1. **Draft** Qwen3-0.6B proposes k tokens
2. **Verify** the base scores all k at once
3. **Keep** the correct run, plus one token of its own

With greedy decoding the answer is exactly what the base model alone would have written.

#### [day3#tokens-per-step] The draft is accepted less in Arabic: 1.8 tokens a step against 2.4

*Lab 3.3 · tasks 1 and 2*

| Language | Per-token acceptance | Predicted tokens per step (k = 3) | Measured |
| --- | --- | --- | --- |
| Arabic | 0.46 | 1.77 | 1.75 |
| English | 0.66 | 2.38 | 2.39 |

Expected tokens per step: (1 − ak+1) / (1 − a). Acceptance figures are synthetic.

#### [day3#two-loads] Speculation cuts TPOT by a fifth when quiet and costs a fifth of attainment at the peak

*Lab 3.3 · tasks 3 and 4*

| Load | Speculation | TPOT p50 | p90 TTFT | Attainment | Blocks |
| --- | --- | --- | --- | --- | --- |
| 4 users | Off | 42 ms | 0.56 s | 100% | 10,002 |
| 4 users | On, k = 3 | 35 ms | 0.64 s | 100% | 5,054 |
| 64 users | Off | 94 ms | 4.2 s | 89% | 10,002 |
| 64 users | On, k = 3 | 74 ms | 6.0 s | 70% | 5,054 |

8-bit weights and cache. The draft's weights and cache take half the blocks. Synthetic.

#### [day3#identical-outputs] Not one recorded answer changed; the rule is about load

*Lab 3.3 · task 5*

**0 of 600** items answered differently with speculation

- **Rule.** On in the quiet hours, off at the evening peak. Day 4's scaling policy can switch it by hour.

#### [day3#checkpoint-3] When does speculation pay?

*Checkpoint*

**Question:** In the quietest hour of the night Musamir serves three visitors at a time; at the evening peak about sixty. The draft is accepted at 0.46 per token in Arabic. Where should speculation be on?

- A. Always: answers are identical, so it can only help
- B. Never for Arabic: acceptance under 0.5 makes it slower
- C. At night, where decode waits on memory and the cache has room
- D. Only at the peak, where speed matters most

#### [day3#lab-3-3] Acceptance rate at two loads

Lab 3.3 · 25 min

Measure the draft model's acceptance rate by language and the time per output token with and without speculation at low and at high load.

[D3_3_measure_acceptance_rate.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D3_3_measure_acceptance_rate.ipynb)

- [ ] Expected tokens per step from acceptance (5 min)
- [ ] Acceptance rate by language (5 min)
- [ ] TPOT at low load, on and off (5 min)
- [ ] TPOT at high load, on and off (5 min)
- [ ] Check the outputs are identical and decide (5 min)
- [ ] ✱ Stretch: Find the best draft length at each load

### Session 4: Distilled models and routing to them

#### [day3#s4] Distilled models and routing to them

*Session 4*

- Compare a distilled model with the base by language
- Choose a routing threshold from a quality and cost sweep
- Write a one-page serving decision

#### [day3#musamir-mini] Musamir-mini is a 1.7B model trained on the base model's answers

*Distillation*

- **Cheaper.** 1,171 output tokens a second on one L4 while meeting the objective, against 441 for the base.
- **Worse alone.** Arabic 0.583, English 0.703: under both floors by far.

Musamir-mini is part of the case: Qwen3-1.7B distilled on the authority's traffic. Its outcomes are synthetic.

#### [day3#easy-requests] On easy requests the mini model is as good as the base

*Lab 3.4 · task 1*

| Difficulty score | Mini | Base (8-bit) |
| --- | --- | --- |
| 0.0 to 0.2 | 0.96 | 0.96 |
| 0.2 to 0.4 | 0.93 | 0.94 |
| 0.4 to 0.6 | 0.73 | 0.85 |
| 0.6 to 0.8 | 0.40 | 0.68 |
| 0.8 to 1.0 | 0.19 | 0.50 |

Share correct by the supplied classifier's score, both languages. Synthetic.

#### [day3#route-sweep] Up to half the requests can go to the mini model before Arabic breaks the floor

*Lab 3.4 · tasks 2 and 3*

| Threshold | Sent to mini | Arabic | English | SAR per million tokens |
| --- | --- | --- | --- | --- |
| 0.2 | 15% | 0.740 | 0.833 | 2.34 |
| 0.4 | 36% | 0.737 | 0.833 | 2.00 |
| 0.5 | 50% | 0.733 | 0.830 | 1.78 |
| 0.6 | 62% | 0.690 | 0.820 | 1.58 |

Base alone: 2.58 SAR per million output tokens; mini alone: 0.97. Case prices, synthetic capacities.

#### [day3#choose-with-margin] 0.5 holds the floor by 0.3 points; 0.4 keeps a margin worth having

*Lab 3.4 · task 4*

- At 0.5, Arabic is 0.733 against 0.73: inside the noise of 300 items.
- At 0.4, Arabic is 0.737, 36% of requests go to mini, the cost falls by 22%.
- The classifier was tuned on this set: on new traffic, expect the margin to shrink.

#### [day3#decision-page] The one-page decision: what to serve, the evidence, what the Director accepts

*Lab 3.4 · task 5*

- **Decision.** 8-bit weights and cache; route under 0.4 to mini; speculation off at the peak.
- **Evidence.** The serving table: memory, goodput, scores with intervals, cost per million tokens.
- **To accept.** Long Arabic documents lose five points; the Arabic margin is under one point.

#### [day3#checkpoint-4] What breaks first?

*Checkpoint*

**Question:** Finance asks for a threshold of 0.6 to cut cost further: 62% of requests to the mini model, English 0.820, Arabic 0.690. What do you answer?

- A. Accept: English still passes its floor
- B. Refuse: Arabic falls under its floor; 0.4 is the largest saving that holds both with a margin
- C. Go further and send everything to the mini model
- D. Accept, and recover quality by quantising the mini model

#### [day3#lab-3-4] Route by difficulty

Lab 3.4 · 25 min

Route requests by a supplied difficulty classifier between the distilled model and the base, sweep the threshold for quality and cost, and pick one that holds both floors.

[D3_4_route_supplied_difficulty.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D3_4_route_supplied_difficulty.ipynb)

- [ ] Quality of each model by language (5 min)
- [ ] Sweep the threshold: quality against share routed (7 min)
- [ ] Cost per million tokens of the mix (5 min)
- [ ] Pick the threshold that holds both floors (5 min)
- [ ] Draft the one-page serving decision (3 min)
- [ ] ✱ Stretch: A cascade that escalates on low confidence

#### [day3#day-3-recap] Measure quality per language, pair the comparison, name what is accepted

*Day 3 · three things to keep*

- **Per language.** 4-bit weights pass in English and fail in Arabic. An average would have shipped them.
- **Paired.** The cache's long-context loss is real on the same items and invisible in two separate intervals.
- **Memory is the currency.** Weights, cache precision and the draft model all trade in the same 24 GB.

#### [day3#tomorrow-topology] Tomorrow: the whole day, several cards, and the cost of a successful request

*Next*

- Bring your serving table and your decision page: the capstone starts from them.
- Read the abstract of DistServe or Splitwise (in the sources).
- Look at the daily demand file: when is the quietest hour, and how quiet?

## Day 4: Across replicas and over the day

Topology, routing, scaling and the cost of a successful request

#### [day4#title] Across replicas and over the day

*SDAIA Academy · Inference Performance and Cost Engineering*

Day 4: Topology, routing, scaling and the cost of a successful request

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-401

#### [day4#where-day-3-left-us] One card in 8 bits meets the objective in the first minutes of the peak

*Where Day 3 left us*

- **100%** attainment, 8-bit weights and cache, first five minutes of the peak
- **2.5** good requests per second one replica sustains
- **0.737** Arabic score with routing at 0.4, floor 0.73

Today: a whole day, a festival's closing night, several cards, and the bill.

#### [day4#review-checkpoint] Yesterday's trade, read again

*Review*

**Question:** A new 3-bit format frees 1.2 GB more than 4-bit and scores 0.845 in English. Nobody has scored it in Arabic yet. What do you tell Yamen?

- A. Ship it: English is above its floor, so Arabic will be too
- B. Ship it: the extra cache is worth any small loss
- C. Never use formats under 8 bits
- D. Score it in Arabic against the floor before any decision

#### [day4#agenda] Across replicas and over the day

*Today*

- **Session 1 · Disaggregated prefill and decode**: Separate pools stop prefills from stalling decodes, at the price of moving the cache.
- **Session 2 · Routing across replicas**: Where a request lands decides whether its prefix is already cached.
- **Session 3 · Autoscaling and cold starts**: Scale on the signal users feel, and keep enough warm replicas for the cold start.
- **Session 4 · Cost per successful request**: The number that decides: cost divided by the requests that met the objective.

### Session 1: Disaggregated prefill and decode

#### [day4#s1] Disaggregated prefill and decode

*Session 1*

- Predict the load on separate prefill and decode pools
- Compute the cost of moving the cache between them
- State when disaggregation pays for a traffic mix

#### [day4#split-the-phases] Prefill on one pool, decode on another, the cache moved between them

*Disaggregation*

1. **Prefill pool** computes the prompt
2. **Transfer** the prompt's cache, over the link
3. **Decode pool** steps never carry a prompt

- Each pool can be sized and configured for its phase.
- The price: the transfer, and two pools to keep busy.

#### [day4#predict-pools] One prefill card would be busy 38% of the time; the transfer costs 12 ms

*Lab 4.1 · tasks 1 and 2*

- **38%** busy share of one prefill card, 2.7 requests a second, half with documents
- **34** sequences decoding at once, by Little's law
- **12 ms** cache transfer for the mean prompt; 37 ms for 10,000 tokens

#### [day4#predict-steps] The prediction matches the supplied runs: shorter stalls when split, but no gain

*Lab 4.1 · tasks 3 and 4*

| Two cards | Predicted longest step | Stall p90 | TPOT p50 | Document TTFT p90 | Attainment |
| --- | --- | --- | --- | --- | --- |
| Together, chunked at 1,024 | 0.168 s | 0.18 s | 63 ms | 0.88 s | 96.5% |
| Split, one prefill and one decode | 0.074 s | 0.13 s | 74 ms | 1.14 s | 93.9% |

32 users, 50% documents, 8-bit weights and cache. Supplied runs of the course simulator.

#### [day4#when-it-pays] With a 150 ms stall limit, only the split meets the objective

*Lab 4.1 · task 5*

| Item | Value |
| --- | --- |
| Together, stall limit 1.0 s | 96.5% |
| Split, stall limit 1.0 s | 93.9% |
| Together, stall limit 0.15 s | 13.0% |
| Split, stall limit 0.15 s | 92.2% |

> **Condition.** Disaggregation pays when the pause limit is shorter than a step that carries a prompt chunk.

#### [day4#checkpoint-1] Split the cards?

*Checkpoint*

**Question:** Another authority's assistant answers short questions with 3,000-token answers, under a 100 ms TPOT limit and a 1 s stall limit. Chunked prefill keeps its stall at 0.2 s. It has two cards. Should it disaggregate?

- A. Yes: disaggregation always lowers TPOT
- B. Not on this evidence: the stall already meets its limit and prompts are short
- C. No: the cache transfer would break the TTFT objective
- D. Yes: long answers need a dedicated prefill card

#### [day4#lab-4-1] When disaggregation pays

Lab 4.1 · 25 min

Predict when splitting prefill and decode pays on long prompts, then check the prediction against a supplied run.

[D4_1_predict_when_disaggregation.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D4_1_predict_when_disaggregation.ipynb)

- [ ] Predict prefill and decode pool load (6 min)
- [ ] Cache transfer time per request (4 min)
- [ ] Predicted TTFT and TPOT, together and split (5 min)
- [ ] Compare with the supplied run (6 min)
- [ ] State the condition when it pays (4 min)
- [ ] ✱ Stretch: Change the share of long prompts and predict again

### Session 2: Routing across replicas

#### [day4#s2] Routing across replicas

*Session 2*

- Compare round-robin and prefix-aware routing on cache hits and latency
- Write a prefix-aware router with a load limit
- Detect a hot spot and correct it

#### [day4#where-it-lands] Round robin sends a conversation's next turn to a replica that has not seen it

*Routing*

- Each replica has its own cache: history computed on one is not on another.
- A prefix-aware router sends each turn where its prefix is cached.
- Affinity piles bursts on one replica unless the router also watches load.

Recap from AE-307, Day 4: inference-aware gateways. Here the question is narrower: what affinity does to one tenant's night and to the tail of the objective.

#### [day4#closing-night] Affinity with a load limit serves 99% on the closing night; round robin 94%

*Lab 4.2 · tasks 1 to 3*

| Router, 3 replicas | Hit rate | p90 TTFT | Attainment |
| --- | --- | --- | --- |
| Round robin | 77.3% | 1.55 s | 93.5% |
| By conversation, no limit | 80.8% | 3.77 s | 78.7% |
| By conversation, limit 24 | 77.9% | 0.78 s | 99.4% |

The evening peak played four times faster: about 8 requests per second. Synthetic: the course simulator.

#### [day4#hot-spot] Routing by tenant puts half the night on one replica

*Lab 4.2 · task 4*

| Item | Value |
| --- | --- |
| Replica 1, Qamra Festival | 49% of requests |
| Replica 2, Sawt Concerts | 33% |
| Replica 0, Athar Heritage Week | 18% |

Attainment 52%: the Qamra replica drowns while the Athar replica idles. Highest hit rate of all routers.

#### [day4#router-rule] Route by conversation, fall back to the least-loaded replica

*Lab 4.2 · task 5*

```python
def by_session(engines, r, state, limit=24):
    i = zlib.crc32(r.session.encode()) % len(engines)
    if load(engines[i]) > limit:
        i = min(range(len(engines)), key=lambda j: load(engines[j]))
    return engines[i]
```

A stable hash, so a turn finds its history; a limit, so no replica takes a burst alone.

#### [day4#checkpoint-2] Read the replica metrics

*Checkpoint*

**Question:** On three replicas, one shows 30 requests waiting and the cache 95% full; the other two show nothing waiting and the cache 20% full. The router hashes the tenant. What is the fix?

- A. Add a fourth replica
- B. Raise the sequence cap on the busy replica
- C. Route by conversation with a load limit, so the busy tenant spreads over all replicas
- D. Remove the cache salt so the replicas can share blocks

#### [day4#lab-4-2] Round robin or prefix-aware

Lab 4.2 · 25 min

Replay a multi-turn trace on three replicas, round robin against a prefix-aware router you write, and compare hits, latency and balance.

[D4_2_compare_round_robin.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D4_2_compare_round_robin.ipynb)

- [ ] Replay round robin on three replicas (5 min)
- [ ] Write a prefix-aware router (7 min)
- [ ] Compare hit rate, TTFT and balance (6 min)
- [ ] Find the hot spot of one tenant (4 min)
- [ ] Choose, with a fallback rule (3 min)
- [ ] ✱ Stretch: Session affinity with a load cap

### Session 3: Autoscaling and cold starts

#### [day4#s3] Autoscaling and cold starts

*Session 3*

- Derive a replica's capacity at the objective from its curve
- Compare scaling signals on the same day
- Choose a minimum replica count from cold start and cost

#### [day4#closing-day] The closing day: 160,000 requests, from 0.3 to 11 per second

*Lab 4.3 · task 1*

Figure: Four times a usual day, minute by minute. Synthetic demand.

#### [day4#two-signals] Scaling on the request rate costs 41% of five cards all day

*Lab 4.3 · tasks 2 and 3*

| Policy, at least 1 replica, 4-minute cold start | Replica-hours | SAR | Worst hour |
| --- | --- | --- | --- |
| Five replicas all day | 120.0 | 492 | 100% |
| Request rate, at 50% of capacity | 49.6 | 203 | 99.8% |
| Queue over 20, down after 10 quiet minutes | 50.5 | 207 | 96.8% |

The course's scaling simulator, a fluid model calibrated from the capacity runs.

#### [day4#cold-start] An 8-minute cold start breaks the queue signal, not the rate signal

*Lab 4.3 · task 4*

| Signal | Cold 2 min | Cold 4 min | Cold 8 min |
| --- | --- | --- | --- |
| Request rate, worst hour | 100% | 99.8% | 99.6% |
| Queue, worst hour | 97.9% | 96.8% | 88.5% |

At least one replica. A second minimum replica adds 12.5 replica-hours and changes little.

#### [day4#scale-to-zero] Scaling to zero saves almost nothing and strands the first visitors of the morning

*Stretch · recap from AE-307, Day 3*

- With a minimum of zero, the closing day costs 0.1 replica-hours less.
- The first morning visitors wait a whole cold start: the worst hour falls to 77%.
- One replica at night is the price of never making a visitor wait minutes.

#### [day4#checkpoint-3] Which signal?

*Checkpoint*

**Question:** A replica takes 8 minutes to start. Demand rises from 2 to 10 requests a second in half an hour every evening. Which policy keeps the worst hour above 90% at the lowest cost?

- A. Scale up when more than 20 requests are waiting
- B. Run enough replicas for the peak all day
- C. Scale on the rate with a minimum of zero
- D. Scale on the request rate at half of a replica's capacity, never below one replica

#### [day4#lab-4-3] Scaling over the day

Lab 4.3 · 25 min

Replay the daily demand pattern in the course scaling simulator and choose the scaling signal and the minimum number of replicas.

[D4_3_replay_daily_pattern.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D4_3_replay_daily_pattern.ipynb)

- [ ] Load the daily pattern and a replica's capacity (4 min)
- [ ] Scale on request rate (5 min)
- [ ] Scale on queue length (5 min)
- [ ] Cold start and minimum replicas (6 min)
- [ ] Choose signal and minimum by cost and attainment (5 min)
- [ ] ✱ Stretch: Scale to zero overnight and measure the first morning visitors

### Session 4: Cost per successful request

#### [day4#s4] Cost per successful request

*Session 4*

- Compute cost and energy per successful request
- Compare self-hosting with a hosted API at three volumes
- Plan the capstone from the course's measurements

#### [day4#per-successful-request] Cost per successful request: what was paid, over who was served

*The number that decides*

- **Paid.** Every replica-hour, starting, serving or idle: card lease plus the node's share.
- **Served.** Only the requests that met the objective: a late answer was paid for and did not serve.

Case prices in SAR (fictional): 3.20 per L4 hour, 0.90 node share, hosted 0.40 and 1.60 per million input and output tokens.

#### [day4#three-volumes] At Liwan's usual volume, the hosted quote costs 2.5 times less per request

*Lab 4.4 · tasks 1 and 2*

| Requests a day | Replica-hours | Node, SAR per 1,000 served | Hosted, SAR per 1,000 |
| --- | --- | --- | --- |
| 10,000 | 24.0 | 9.83 | 1.05 |
| 40,000 | 26.2 | 2.69 | 1.05 |
| 160,000 | 49.7 | 1.27 | 1.05 |

A mean request: 2,004 prompt tokens and 157 answer tokens. Scaling on rate, at least one replica, 8-minute cold start.

#### [day4#break-even] Self-hosting breaks even near 360,000 requests a day, nine times today's

*Lab 4.4 · task 3*

Figure: Case prices; the course's scaling simulator. With the lease 30% cheaper, the break-even falls to about 140,000 a day.

#### [day4#energy] Idle cards dominate the energy: 117 Wh per 1,000 requests on a usual day

*Lab 4.4 · task 4*

| Item | Value |
| --- | --- |
| 10,000 a day | 426 Wh |
| 40,000 a day | 117 Wh |
| 160,000 a day | 55 Wh |

Per 1,000 successful requests: card at its 72 W maximum, 55 W of host, PUE 1.4. An upper bound.

#### [day4#capstone-plan] The capstone starts from the deployment of Monday morning

*Lab 4.4 · task 5*

- **You get.** The baseline configuration, the trace, the evaluation outcomes, the prices, the simulator and the stand-in endpoint.
- **You submit.** A public repository: raw benchmark files, engine logs, quality by language, a design document and the commit history.

Defence: 12 minutes of talk and 8 of questions, in front of a panel playing Hanadi, Rawya and Kenan.

#### [day4#checkpoint-4] Which number goes to the Director?

*Checkpoint*

**Question:** Kenan's slide says self-hosting costs 2.58 SAR per million output tokens against a hosted 1.60, "so the hosted API is cheaper". Hanadi asks if that is the right comparison. What do you answer?

- A. No: compare cost per successful request over a replayed day, input tokens included
- B. Yes: per-token prices are how providers compare
- C. No: idle hours should be left out of the node's cost
- D. No: compare the node at its peak capacity only

#### [day4#lab-4-4] Self-hosting or a hosted API

Lab 4.4 · 25 min

Compare the cost per successful request of the node with a hosted API quote at three volumes, add energy per thousand requests and plan the capstone.

[D4_4_compare_self_hosting.ipynb](https://sdaiaacademy.github.io/ae-401/labs/D4_4_compare_self_hosting.ipynb)

- [ ] Cost per successful request on the node (6 min)
- [ ] The hosted API at three volumes (5 min)
- [ ] The break-even volume (4 min)
- [ ] Energy per thousand requests (4 min)
- [ ] Plan the capstone (6 min)
- [ ] ✱ Stretch: How sensitive the break-even is to the lease price

#### [day4#course-recap] Predict, measure with spread, change one thing, re-evaluate in both languages, price it

*Four days · what to keep*

- **From 0.15 to 2.5.** Good requests per second on one L4: scheduling, the prompt, 8-bit weights and cache.
- **Quality held.** Every change scored against the floors per language, with intervals and paired differences.
- **The honest bill.** Cost per successful request over a real day, against the quote, with the break-even.

#### [day4#next-capstone] Capstone: tune Monday's deployment and defend it

*Next*

- Start from `labs/capstone/` and your plan from Lab 4.4.
- Commit as you go: the history is part of the evidence.
- Bring the one decision you want the Director to take.

## Capstone: Capstone project presentations

Twelve minutes of evidence and eight of questions, with one traffic change predicted and rerun

#### [capstone#title] Capstone project presentations

*SDAIA Academy · Inference Performance and Cost Engineering*

Capstone: Twelve minutes of evidence and eight of questions, with one traffic change predicted and rerun

AI Engineering Track · 4 teaching days + capstone day · 12 min talk and demo · 8 min Q&A · AE-401

#### [capstone#agenda] How the presentations run

*Capstone*

- **Expectations and final prep**: the rubric, a last rehearsal, then the first presentations
- **Presentations**: each one timed, with questions after it
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#the-task] Monday's deployment, tuned to the objective at the lowest cost per successful request

*Capstone*

- **Hanadi's brief.** 90% of peak requests within 2.0 s to the first token, 150 ms a token and no pause over 1.0 s; Arabic at least 0.73 and English 0.81; no cache block across tenants; no personal data in a cached entry.
- **The evidence.** A public repository: raw benchmark rows, engine logs, quality by language, the isolation test, a design document with cost and energy per request, and the commit history.

#### [capstone#rubric] The rubric

*How presentations are scored*

| Item | Value |
| --- | --- |
| Cost model | 15% |
| Benchmark method | 25% |
| Optimisations with re-evaluation | 25% |
| Topology design | 20% |
| Reproducibility | 15% |

> **Three runs a point, raw rows in the repository.** A claim without a number is a plan, not a result.

#### [capstone#format] 12 minutes of talk, 8 minutes of questions

*Format*

- **1 · The baseline.** The curve of the provided deployment and the three diagnoses.
- **2 · What you changed.** Each change with goodput and quality per language; the topology.
- **3 · The decision.** Cost per successful request against the quote, and what the Director must accept.

> **In the questions.** The assessor changes the traffic mix; you predict the goodput effect, then rerun one point.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Project |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first presentation*

Rehearse once, record a backup, try the demo on this screen.

#### [capstone#now-presenting] Talk

*Now presenting*

##### Questions

#### [capstone#review] What the presentations showed

*Trainer review*

- **Strongest patterns.** What several projects got right.
- **Common gaps.** What most projects would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#what-next] Keep going

*After the course*

- **Your project stays yours:** keep it reproducible and add the next improvement you named today
- **The follow-up task:** within a month, repeat the method on your own deployment or a public trace: baseline curve, one change with its re-evaluation, cost per successful request

#### [capstone#thanks] Thank you

*SDAIA Academy*

Inference Performance and Cost Engineering
