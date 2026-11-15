<!-- academy:title -->
# Inference Performance and Cost Engineering

هندسة أداء الاستنتاج وتكلفته

AI Engineering Track · SDAIA Academy · AE-401 · Level 400

REPLACE: one line under the title: what participants can do after the course

Participants take the bilingual visitor assistant of a fictional national events authority, too slow at evening peaks and too costly, and make it meet its service objective at the lowest cost they can defend. They build a first-principles cost model, measure goodput curves on replayed traffic, diagnose seeded bottlenecks, then apply batching, caching, quantisation, speculative decoding, distillation, routing, disaggregated serving and autoscaling, re-running an Arabic and English task evaluation after every change.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ae-401/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Compute prefill cost, decode cost and key-value cache memory, and predict the binding resource, given a model card, an accelerator specification sheet and a traffic profile | Applied | a sizing sheet whose predicted concurrency limit is within 25% of a measured run |
| LO2 | Evaluate a serving configuration by measuring its goodput curve over concurrency against latency objectives, given an engine, a small open-weight model and a replayed synthetic trace | Performance | a repeatable curve of at least seven points with raw data and run-to-run variance |
| LO3 | Diagnose why a deployment misses its service objective and confirm the fix, given three seeded faults: queueing, preemption under cache pressure and prefill interference | Performance | a diagnosis per fault with its evidence, and a rerun that meets the objective |
| LO4 | Optimise batching, admission, prefix caching and cache precision while holding task quality, given a multi-tenant trace and Arabic and English evaluation sets | Performance | before-and-after goodput, cost per million tokens and quality with intervals, and a passed cross-tenant cache test |
| LO5 | Evaluate quantised, distilled and speculative-decoding variants against the base model and decide which to serve, given a quality floor per language and a cost target | Performance | a quality and cost table by language and a one-page serving decision |
| LO6 | Design and defend a serving topology covering disaggregation, routing, replica count and scaling policy, given a daily demand pattern, a service objective and a budget | Performance | a design document with predicted and measured goodput and cost per successful request |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Model the cost first | Prefill and decode are different problems; Key-value cache arithmetic; Goodput against an objective; Curves, not single numbers | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Scheduling and caching in the engine | Continuous batching and admission; Long prompts slow short ones; Prefix caching starts with prompt layout; Semantic caching and wrong answers | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Cheaper tokens, same quality | Quantising weights; Cache precision buys concurrency; Speculative decoding; Distilled models and routing to them | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Across replicas and over the day | Disaggregated prefill and decode; Routing across replicas; Autoscaling and cold starts; Cost per successful request | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Compute operations and bytes per token for two models and predict what bounds each phase | [`D1_1_compute_operations_bytes.ipynb`](labs/D1_1_compute_operations_bytes.ipynb) |
| 1.2 | Day 1 · Session 2 | Predict the concurrency limit at three context lengths and check it against the engine | [`D1_2_predict_concurrency_limit.ipynb`](labs/D1_2_predict_concurrency_limit.ipynb) |
| 1.3 | Day 1 · Session 3 | Load the baseline at three concurrency levels and compute goodput | [`D1_3_load_baseline_three.ipynb`](labs/D1_3_load_baseline_three.ipynb) |
| 1.4 | Day 1 · Session 4 | Build a seven-point goodput curve from supplied runs plus one of your own | [`D1_4_build_seven_point.ipynb`](labs/D1_4_build_seven_point.ipynb) |
| 2.1 | Day 2 · Session 1 | Vary the concurrency cap and plot tail latency against throughput | [`D2_1_vary_concurrency_cap.ipynb`](labs/D2_1_vary_concurrency_cap.ipynb) |
| 2.2 | Day 2 · Session 2 | Mix long-document and chat traffic with chunked prefill on and off | [`D2_2_mix_long_document.ipynb`](labs/D2_2_mix_long_document.ipynb) |
| 2.3 | Day 2 · Session 3 | Reorder a prompt for reuse, measure the hit rate, then block cross-tenant hits | [`D2_3_reorder_prompt_reuse.ipynb`](labs/D2_3_reorder_prompt_reuse.ipynb) |
| 2.4 | Day 2 · Session 4 | Measure hit rate and wrong answers on paraphrase pairs at three thresholds | [`D2_4_measure_hit_rate.ipynb`](labs/D2_4_measure_hit_rate.ipynb) |
| 3.1 | Day 3 · Session 1 | Compare two weight precisions on memory, goodput and bilingual task scores | [`D3_1_compare_two_weight.ipynb`](labs/D3_1_compare_two_weight.ipynb) |
| 3.2 | Day 3 · Session 2 | Lower cache precision and measure concurrency gained and long-context accuracy lost | [`D3_2_lower_cache_precision.ipynb`](labs/D3_2_lower_cache_precision.ipynb) |
| 3.3 | Day 3 · Session 3 | Measure acceptance rate and time per output token at low and high load | [`D3_3_measure_acceptance_rate.ipynb`](labs/D3_3_measure_acceptance_rate.ipynb) |
| 3.4 | Day 3 · Session 4 | Route by a supplied difficulty classifier between a distilled model and the base | [`D3_4_route_supplied_difficulty.ipynb`](labs/D3_4_route_supplied_difficulty.ipynb) |
| 4.1 | Day 4 · Session 1 | Predict when disaggregation pays on long prompts and check against a supplied run | [`D4_1_predict_when_disaggregation.ipynb`](labs/D4_1_predict_when_disaggregation.ipynb) |
| 4.2 | Day 4 · Session 2 | Compare round-robin with prefix-aware routing on a multi-turn trace | [`D4_2_compare_round_robin.ipynb`](labs/D4_2_compare_round_robin.ipynb) |
| 4.3 | Day 4 · Session 3 | Replay the daily pattern in the simulator and choose the scaling signal and minimum replicas | [`D4_3_replay_daily_pattern.ipynb`](labs/D4_3_replay_daily_pattern.ipynb) |
| 4.4 | Day 4 · Session 4 | Compare self-hosting with a hosted API at three volumes and plan the capstone | [`D4_4_compare_self_hosting.ipynb`](labs/D4_4_compare_self_hosting.ipynb) |
<!-- /academy:labs -->

## Capstone

Tune a provided under-performing deployment of the case assistant to meet its latency objectives at minimum cost per successful request, submitted as a public repository with raw benchmark data, quality by language and a design document, and defended in 20 minutes. The follow-up task repeats the method on the participant's own deployment or a public trace

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- AE-305 Operating Machine Learning and Language Model Systems
- AE-307 Container Orchestration and Cloud Infrastructure for AI Workloads
- Has served an open-weight model behind an HTTP endpoint and load-tested it
- Computes percentiles and reads a latency histogram
- Estimates memory from parameter count and numeric precision

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Inference engine:** vLLM, SGLang, llama.cpp server
- **Load generator with trace replay:** inference-perf, Locust
- **Profiling and metrics:** PyTorch profiler, Prometheus with Grafana
- **Analytical performance model:** MLSys·im, a spreadsheet roofline and queueing model
- **Quality regression harness:** lm-evaluation-harness, Inspect
- **Autoscaling:** a course-supplied scaling simulator, KEDA on the provided node
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
- Regulation on Personal Data Transfer outside the Kingdom: REPLACE: the address of the official text
- NDMO Data Classification Policy: REPLACE: the address of the official text
- SDAIA Generative AI Guidelines: REPLACE: the address of the official text
