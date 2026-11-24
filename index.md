---
course: "Inference Performance and Cost Engineering"
code: "AE-401"
version: "2027.2.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-401/index.html"
---

# Inference Performance and Cost Engineering

هندسة أداء الاستدلال وتكلفته

AI Engineering Track · 4 teaching days + capstone day · 26 hours · Level 400

Make a language model service meet its latency objectives at the lowest cost you can defend, with quality held in Arabic and English

Participants take the bilingual visitor assistant of a fictional national events authority, too slow at evening peaks and too costly, and make it meet its service objective at the lowest cost they can defend. They build a first-principles cost model, measure goodput curves on replayed traffic, diagnose seeded bottlenecks, then apply batching, caching, quantisation, speculative decoding, distillation, routing, disaggregated serving and autoscaling, re-running an Arabic and English task evaluation after every change.

**Who it is for:** Specialists. Engineers who have completed the level 300 core or can show equivalent work: a production AI system they built, operated and can defend.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Liwan National Events Authority.** Musamir, the bilingual visitor assistant of a fictional national events authority, serves three event organisers from one leased accelerator and is too slow at evening peaks and too costly. Every lab works on the same synthetic data: 40,000 Arabic and English visitor requests over a day, model cards and spec sheets, recorded evaluation outcomes for six model variants and a course simulator of the serving engine, all generated with a fixed seed.

## The course

### Day 1: Model the cost first

Slides: https://sdaiaacademy.github.io/ae-401/slides/day1.md

- **Session 1: Prefill and decode are different problems**
  - Compute the operations and bytes of one token for a model card
  - Place prefill and decode on the roofline of an accelerator
  - Predict time to first token and time per output token from first principles
  - [Lab 1.1: Operations and bytes per token (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D1_1_compute_operations_bytes.ipynb)
- **Session 2: Key-value cache arithmetic**
  - Compute key-value cache bytes per token from a model card
  - Predict the concurrency limit at a given context length
  - Explain the gap between a prediction and the engine's block count
  - [Lab 1.2: Concurrency limit at three context lengths (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D1_2_predict_concurrency_limit.ipynb)
- **Session 3: Goodput against an objective**
  - Measure time to first token and time per output token from a streamed response
  - Compute goodput against two per-request objectives
  - Break a request's time into queue, prefill and decode
  - [Lab 1.3: The baseline at three concurrency levels (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D1_3_load_baseline_three.ipynb)
- **Session 4: Curves, not single numbers**
  - Build a goodput curve over concurrency with repeated runs
  - Read the knee of a curve and state a capacity with its variance
  - Judge whether a benchmark is valid
  - [Lab 1.4: A seven-point goodput curve (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D1_4_build_seven_point.ipynb)

### Day 2: Scheduling and caching in the engine

Slides: https://sdaiaacademy.github.io/ae-401/slides/day2.md

- **Session 1: Continuous batching and admission**
  - Explain iteration-level scheduling and the sequence cap
  - Tell queueing from preemption by their metrics
  - Choose a concurrency cap from a tail-latency and throughput plot
  - [Lab 2.1: The concurrency cap (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D2_1_vary_concurrency_cap.ipynb)
- **Session 2: Long prompts slow short ones**
  - Recognise prefill interference in time per output token
  - Tune chunked prefill with a token budget
  - Confirm a fix with a rerun against the objective
  - [Lab 2.2: Long documents and chat, chunked or not (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D2_2_mix_long_document.ipynb)
- **Session 3: Prefix caching starts with prompt layout**
  - Lay out a prompt so that its stable part comes first
  - Measure the prefix-cache hit rate and its effect on time to first token
  - Prove that no cache block is shared between tenants
  - [Lab 2.3: Prompt layout and tenant isolation (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D2_3_reorder_prompt_reuse.ipynb)
- **Session 4: Semantic caching and wrong answers**
  - Measure hit rate and wrong-answer rate of a semantic cache
  - Compare the rates by language
  - Keep personal data out of cached entries
  - [Lab 2.4: Semantic cache thresholds (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D2_4_measure_hit_rate.ipynb)

### Day 3: Cheaper tokens, same quality

Slides: https://sdaiaacademy.github.io/ae-401/slides/day3.md

- **Session 1: Quantising weights**
  - Choose a weight format the accelerator supports
  - Measure memory, goodput and quality of each precision
  - Report quality by language with intervals
  - [Lab 3.1: Two weight precisions (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D3_1_compare_two_weight.ipynb)
- **Session 2: Cache precision buys concurrency**
  - Compute the blocks gained by a lower cache precision
  - Measure the concurrency and goodput gained
  - Measure long-context accuracy lost, per language, as a paired difference
  - [Lab 3.2: Lower cache precision (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D3_2_lower_cache_precision.ipynb)
- **Session 3: Speculative decoding**
  - Compute the expected tokens per step from an acceptance rate
  - Measure acceptance and time per output token at low and high load
  - Decide when to switch speculation on
  - [Lab 3.3: Acceptance rate at two loads (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D3_3_measure_acceptance_rate.ipynb)
- **Session 4: Distilled models and routing to them**
  - Compare a distilled model with the base by language
  - Choose a routing threshold from a quality and cost sweep
  - Write a one-page serving decision
  - [Lab 3.4: Route by difficulty (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D3_4_route_supplied_difficulty.ipynb)

### Day 4: Across replicas and over the day

Slides: https://sdaiaacademy.github.io/ae-401/slides/day4.md

- **Session 1: Disaggregated prefill and decode**
  - Predict the load on separate prefill and decode pools
  - Compute the cost of moving the cache between them
  - State when disaggregation pays for a traffic mix
  - [Lab 4.1: When disaggregation pays (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D4_1_predict_when_disaggregation.ipynb)
- **Session 2: Routing across replicas**
  - Compare round-robin and prefix-aware routing on cache hits and latency
  - Write a prefix-aware router with a load limit
  - Detect a hot spot and correct it
  - [Lab 4.2: Round robin or prefix-aware (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D4_2_compare_round_robin.ipynb)
- **Session 3: Autoscaling and cold starts**
  - Derive a replica's capacity at the objective from its curve
  - Compare scaling signals on the same day
  - Choose a minimum replica count from cold start and cost
  - [Lab 4.3: Scaling over the day (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D4_3_replay_daily_pattern.ipynb)
- **Session 4: Cost per successful request**
  - Compute cost and energy per successful request
  - Compare self-hosting with a hosted API at three volumes
  - Plan the capstone from the course's measurements
  - [Lab 4.4: Self-hosting or a hosted API (25 min)](https://sdaiaacademy.github.io/ae-401/labs/D4_4_compare_self_hosting.ipynb)

### Capstone: Capstone project presentations

Slides: https://sdaiaacademy.github.io/ae-401/slides/capstone.md

## Capstone

Tune a provided under-performing deployment of the case assistant to meet its latency objectives at minimum cost per successful request, submitted as a public repository with raw benchmark data, quality by language and a design document, and defended in 20 minutes. The follow-up task repeats the method on the participant's own deployment or a public trace

**What you submit:** A public GitHub repository link: raw benchmark files, engine logs, quality by language, the design document and a commit history; submitted before the defence

**Rubric:**

- Cost model: 15%
- Benchmark method: 25%
- Optimisations with re-evaluation: 25%
- Topology design: 20%
- Reproducibility: 15%

## Outcomes

- **LO1** Compute prefill cost, decode cost and key-value cache memory, and predict the binding resource, given a model card, an accelerator specification sheet and a traffic profile (Applied) Evidence: a sizing sheet whose predicted concurrency limit is within 25% of a measured run
- **LO2** Evaluate a serving configuration by measuring its goodput curve over concurrency against latency objectives, given an engine, a small open-weight model and a replayed synthetic trace (Performance) Evidence: a repeatable curve of at least seven points with raw data and run-to-run variance
- **LO3** Diagnose why a deployment misses its service objective and confirm the fix, given three seeded faults: queueing, preemption under cache pressure and prefill interference (Performance) Evidence: a diagnosis per fault with its evidence, and a rerun that meets the objective
- **LO4** Optimise batching, admission, prefix caching and cache precision while holding task quality, given a multi-tenant trace and Arabic and English evaluation sets (Performance) Evidence: before-and-after goodput, cost per million tokens and quality with intervals, and a passed cross-tenant cache test
- **LO5** Evaluate quantised, distilled and speculative-decoding variants against the base model and decide which to serve, given a quality floor per language and a cost target (Performance) Evidence: a quality and cost table by language and a one-page serving decision
- **LO6** Design and defend a serving topology covering disaggregation, routing, replica count and scaling policy, given a daily demand pattern, a service objective and a budget (Performance) Evidence: a design document with predicted and measured goodput and cost per successful request

## Tools

- **Inference engine:** vLLM, SGLang, llama.cpp server
- **Load generator with trace replay:** inference-perf, Locust
- **Profiling and metrics:** PyTorch profiler, Prometheus with Grafana
- **Analytical performance model:** MLSys·im, a spreadsheet roofline and queueing model
- **Quality regression harness:** lm-evaluation-harness, Inspect
- **Autoscaling:** a course-supplied scaling simulator, KEDA on the provided node

## Prerequisites

- AE-305 Operating Machine Learning and Language Model Systems
- AE-307 Container Orchestration and Cloud Infrastructure for AI Workloads
- Has served an open-weight model behind an HTTP endpoint and load-tested it
- Computes percentiles and reads a latency histogram
- Estimates memory from parameter count and numeric precision

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

---

SDAIA Academy · AI Engineering Track · Inference Performance and Cost Engineering

Course version 2027.2.0 · engine 1.0.0 · built 2026-11-24 · commit 3238690
