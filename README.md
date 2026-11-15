<!-- academy:title -->
# Inference Performance and Cost Engineering

هندسة أداء الاستدلال وتكلفته

AI Engineering Track · SDAIA Academy · AE-401 · Level 400

Make a language model service meet its latency objectives at the lowest cost you can defend, with quality held in Arabic and English

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

**Liwan National Events Authority** (fictional). Its bilingual visitor assistant, Musamir, serves three event organisers (Qamra Festival, Sawt Concerts and Athar Heritage Week) from one leased NVIDIA L4 accelerator running Qwen3-8B, and is too slow at the evening peak and too costly. The labs work on 40,000 synthetic Arabic and English visitor requests over a day, the real model cards and accelerator sheets, recorded evaluation outcomes for six model variants on 600 Arabic and English items, 240 paraphrase pairs and a course simulator of the serving engine, all generated with a fixed seed (`labs/prepare_data.py`, seed 401). The people of the case: Dr. Hanadi Al-Kathiri (head of visitor digital services, owner of the objective and the budget), Yamen Al-Hulaibi (platform engineer), Asrar Al-Rifai (evaluation lead), Rawya Al-Twaim (data protection officer), Kenan Al-Ajlan (finance analyst); participants play Tamader Al-Harqan, the inference engineer.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Operations and bytes per token | [`D1_1_compute_operations_bytes.ipynb`](labs/D1_1_compute_operations_bytes.ipynb) |
| 1.2 | Day 1 · Session 2 | Concurrency limit at three context lengths | [`D1_2_predict_concurrency_limit.ipynb`](labs/D1_2_predict_concurrency_limit.ipynb) |
| 1.3 | Day 1 · Session 3 | The baseline at three concurrency levels | [`D1_3_load_baseline_three.ipynb`](labs/D1_3_load_baseline_three.ipynb) |
| 1.4 | Day 1 · Session 4 | A seven-point goodput curve | [`D1_4_build_seven_point.ipynb`](labs/D1_4_build_seven_point.ipynb) |
| 2.1 | Day 2 · Session 1 | The concurrency cap | [`D2_1_vary_concurrency_cap.ipynb`](labs/D2_1_vary_concurrency_cap.ipynb) |
| 2.2 | Day 2 · Session 2 | Long documents and chat, chunked or not | [`D2_2_mix_long_document.ipynb`](labs/D2_2_mix_long_document.ipynb) |
| 2.3 | Day 2 · Session 3 | Prompt layout and tenant isolation | [`D2_3_reorder_prompt_reuse.ipynb`](labs/D2_3_reorder_prompt_reuse.ipynb) |
| 2.4 | Day 2 · Session 4 | Semantic cache thresholds | [`D2_4_measure_hit_rate.ipynb`](labs/D2_4_measure_hit_rate.ipynb) |
| 3.1 | Day 3 · Session 1 | Two weight precisions | [`D3_1_compare_two_weight.ipynb`](labs/D3_1_compare_two_weight.ipynb) |
| 3.2 | Day 3 · Session 2 | Lower cache precision | [`D3_2_lower_cache_precision.ipynb`](labs/D3_2_lower_cache_precision.ipynb) |
| 3.3 | Day 3 · Session 3 | Acceptance rate at two loads | [`D3_3_measure_acceptance_rate.ipynb`](labs/D3_3_measure_acceptance_rate.ipynb) |
| 3.4 | Day 3 · Session 4 | Route by difficulty | [`D3_4_route_supplied_difficulty.ipynb`](labs/D3_4_route_supplied_difficulty.ipynb) |
| 4.1 | Day 4 · Session 1 | When disaggregation pays | [`D4_1_predict_when_disaggregation.ipynb`](labs/D4_1_predict_when_disaggregation.ipynb) |
| 4.2 | Day 4 · Session 2 | Round robin or prefix-aware | [`D4_2_compare_round_robin.ipynb`](labs/D4_2_compare_round_robin.ipynb) |
| 4.3 | Day 4 · Session 3 | Scaling over the day | [`D4_3_replay_daily_pattern.ipynb`](labs/D4_3_replay_daily_pattern.ipynb) |
| 4.4 | Day 4 · Session 4 | Self-hosting or a hosted API | [`D4_4_compare_self_hosting.ipynb`](labs/D4_4_compare_self_hosting.ipynb) |
<!-- /academy:labs -->

## Capstone

Tune a provided under-performing deployment of the case assistant to meet its latency objectives at minimum cost per successful request, submitted as a public repository with raw benchmark data, quality by language and a design document, and defended in 20 minutes. The follow-up task repeats the method on the participant's own deployment or a public trace

**Submission:** A public GitHub repository link: raw benchmark files, engine logs, quality by language, the design document and a commit history; submitted before the defence

The defence is 12 minutes of talk and 8 minutes of questions. The rubric: cost model 15, benchmark method 25, optimisations with re-evaluation 25, topology design 20, reproducibility 15. The brief, the files given and what full marks look like are in [`labs/capstone/README.md`](labs/capstone/README.md); `labs/capstone/starter.py` measures the provided deployment and keeps the raw rows. AI assistants may be used throughout the capstone; every number in the defence must come from a run in the participant's repository.

## Assessment

- **How the course is assessed:** supervised entry and exit checks, AI off (memory and prefill arithmetic, reading a goodput curve, choosing a first optimisation), 20%; the capstone rubric, 60%; the live 20-minute defence, 20%. Passing is 70% overall and at least 60% on both the capstone rubric and the supervised check. The certificate is completion with assessed work.
- **Authenticity:** the raw benchmark files, engine logs and commit history of the capstone repository; in the defence the assessor changes the traffic mix and the participant predicts the goodput effect before re-running one point.
- **AI policy:** AI off for the supervised checks and the hand sizing of Labs 1.1 and 1.2; an AI tutor in the other labs, which explains engine options and profiles but does not choose settings; AI allowed throughout the capstone, which is defended live.
- **Checkpoints:** every session ends with a scenario question in the deck, each wrong option naming the misconception behind it.
- **Practice questions:** [`assessments/bank.json`](assessments/bank.json) holds 30 scenario questions, five per outcome, with explanations. Exam items are kept outside this repository.
- **Labs:** each lab's checks and its carried-forward file are the evidence of the outcome it serves (see the Evidence column above).

## How the labs run

All sixteen labs run on a laptop processor or a free Colab runtime, with no model download and no API key. They use a simulator of the serving engine calibrated from the published model cards and accelerator sheets, an OpenAI-compatible streaming stand-in for client-side measurement, and recorded evaluation outcomes; a real engine on a notebook GPU can replace the stand-in in Lab 1.3 and the capstone. The numbers on the slides come from these runs and are synthetic. Details, calibration and data: [`labs/README.md`](labs/README.md).

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

### Models and accelerators

- [Qwen3-8B model card and configuration (Hugging Face)](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3-1.7B model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3-0.6B model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-0.6B)
- [NVIDIA L4 Tensor Core GPU specifications](https://www.nvidia.com/en-us/data-center/l4/)
- [NVIDIA T4 Tensor Core GPU specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/)

### Serving engines and their documentation

- [vLLM: engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- [vLLM: optimisation and tuning (preemption, chunked prefill)](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM: automatic prefix caching and cache salts](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- [vLLM: metrics](https://docs.vllm.ai/en/latest/design/metrics/)
- [vLLM: quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM: speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM: quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)
- [vLLM (GitHub)](https://github.com/vllm-project/vllm)
- [SGLang (GitHub)](https://github.com/sgl-project/sglang)
- [llama.cpp (GitHub)](https://github.com/ggml-org/llama.cpp)

### Research

- [Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)](https://arxiv.org/abs/2309.06180)
- [Yu et al., Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Agrawal et al., Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (2024)](https://arxiv.org/abs/2403.02310)
- [Zhong et al., DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (2024)](https://arxiv.org/abs/2401.09670)
- [Patel et al., Splitwise: Efficient generative LLM inference using phase splitting (2023)](https://arxiv.org/abs/2311.18677)
- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding (2022)](https://arxiv.org/abs/2211.17192)
- [Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling (2023)](https://arxiv.org/abs/2302.01318)
- [Hinton et al., Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531)
- [Lin et al., AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (2023)](https://arxiv.org/abs/2306.00978)
- [Dettmers et al., LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)](https://arxiv.org/abs/2208.07339)

### Tools

- [inference-perf (Kubernetes SIGs)](https://github.com/kubernetes-sigs/inference-perf)
- [Locust](https://locust.io/)
- [Prometheus](https://prometheus.io/)
- [MLSYSIM whitepaper (Harvard)](https://harvard-edge.github.io/cs249r_book_dev/mlsysim/whitepaper.html)
- [lm-evaluation-harness (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness)
- [Inspect](https://inspect.aisi.org.uk/)
- [KEDA](https://keda.sh/)

### National references

- [Personal Data Protection Law (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [Regulation on Personal Data Transfer outside the Kingdom (SDAIA)](https://sdaia.gov.sa/Documents/RegulationonPersonalDataEN.pdf)
- [Data Classification Policy (NDMO, SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf)
- [Generative AI Guidelines for Government (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf)
<!-- /academy:sources -->

### National references

- Personal Data Protection Law and its implementing regulations: <https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf> and <https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf>
- Regulation on Personal Data Transfer outside the Kingdom: <https://sdaia.gov.sa/Documents/RegulationonPersonalDataEN.pdf>
- NDMO Data Classification Policy: <https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf>
- SDAIA Generative AI Guidelines (the Generative AI Guideline for Government): <https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf>
