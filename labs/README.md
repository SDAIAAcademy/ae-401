# Inference Performance and Cost Engineering: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Prefill and decode are different problems | [`D1_1_compute_operations_bytes.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D1_1_compute_operations_bytes.ipynb) |
| Day 1 | Session 2 · Key-value cache arithmetic | [`D1_2_predict_concurrency_limit.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D1_2_predict_concurrency_limit.ipynb) |
| Day 1 | Session 3 · Goodput against an objective | [`D1_3_load_baseline_three.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D1_3_load_baseline_three.ipynb) |
| Day 1 | Session 4 · Curves, not single numbers | [`D1_4_build_seven_point.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D1_4_build_seven_point.ipynb) |
| Day 2 | Session 1 · Continuous batching and admission | [`D2_1_vary_concurrency_cap.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D2_1_vary_concurrency_cap.ipynb) |
| Day 2 | Session 2 · Long prompts slow short ones | [`D2_2_mix_long_document.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D2_2_mix_long_document.ipynb) |
| Day 2 | Session 3 · Prefix caching starts with prompt layout | [`D2_3_reorder_prompt_reuse.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D2_3_reorder_prompt_reuse.ipynb) |
| Day 2 | Session 4 · Semantic caching and wrong answers | [`D2_4_measure_hit_rate.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D2_4_measure_hit_rate.ipynb) |
| Day 3 | Session 1 · Quantising weights | [`D3_1_compare_two_weight.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D3_1_compare_two_weight.ipynb) |
| Day 3 | Session 2 · Cache precision buys concurrency | [`D3_2_lower_cache_precision.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D3_2_lower_cache_precision.ipynb) |
| Day 3 | Session 3 · Speculative decoding | [`D3_3_measure_acceptance_rate.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D3_3_measure_acceptance_rate.ipynb) |
| Day 3 | Session 4 · Distilled models and routing to them | [`D3_4_route_supplied_difficulty.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D3_4_route_supplied_difficulty.ipynb) |
| Day 4 | Session 1 · Disaggregated prefill and decode | [`D4_1_predict_when_disaggregation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D4_1_predict_when_disaggregation.ipynb) |
| Day 4 | Session 2 · Routing across replicas | [`D4_2_compare_round_robin.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D4_2_compare_round_robin.ipynb) |
| Day 4 | Session 3 · Autoscaling and cold starts | [`D4_3_replay_daily_pattern.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D4_3_replay_daily_pattern.ipynb) |
| Day 4 | Session 4 · Cost per successful request | [`D4_4_compare_self_hosting.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-401/blob/main/labs/D4_4_compare_self_hosting.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.

## Two tiers: what runs where

Every lab runs on a laptop processor or a free Colab runtime, with no model download and no API key.

| Tier | What it is | Used in |
|---|---|---|
| **Simulator** (`servelab.py`) | A discrete-event model of one serving engine and several replicas: continuous batching, a sequence cap and a token budget, chunked prefill, a paged cache of 16-token blocks with prefix hashing and a per-tenant salt, preemption by recompute, 16/8/4-bit weights and 8-bit cache, speculative decoding with a draft model, disaggregated prefill and decode pools, routers | Every lab |
| **Stand-in endpoint** (`standin_server.py`) | An OpenAI-compatible streaming server on a local port that serves tokens at the simulator's pace, so `loadgen.py` measures from the client's side exactly as against a real engine | Lab 1.3 |
| **A real engine** (optional) | `vllm serve Qwen/Qwen3-0.6B` in a notebook GPU runtime; set `SERVE_ENGINE_URL` and Lab 1.3 loads it with the same code | Lab 1.3 stretch, capstone |

Quality is never simulated from scratch: the evaluation outcomes in `data/eval/` are recorded per item and per
variant, so every comparison is paired and repeatable. They are synthetic.

## How the simulator is calibrated

Its step time is the larger of compute time and memory time, plus overheads (`S.CAL`):

| Constant | Value | Meaning |
|---|---|---|
| `mem_eff` | 0.80 | Share of the card's peak memory bandwidth reached |
| `compute_eff` | 0.45 | Share of the dense peak operations reached |
| `reserve_gb` | 1.6 | Memory for activations, graphs and runtime outside the cache |
| `step_overhead_s` | 0.004 | Scheduling, sampling and launch per step |
| `int4_dequant` | 1.15 | Extra compute of 4-bit weights unpacked to 16 bits |
| `jitter` | 0.03 | Run-to-run noise on step time |
| `link_gbs` | 20 | Effective cache transfer between two cards of a node |

The model cards (Qwen3-8B, 1.7B, 0.6B) and the accelerator figures (NVIDIA L4 and T4) are the published ones,
with their sources in `data/model_cards.json` and `data/accelerators.json`; the L4 sheet's tensor figures are
quoted with sparsity, and the simulator uses the dense half. A single request alone lands within about 10% of
the hand arithmetic of Lab 1.1. The simulator is not a measurement of any engine: it is a consistent model
in which the effects taught here appear with plausible sizes.

## The data

`prepare_data.py` writes everything in `data/` from seed 401 (standard library only):

| File | Contents |
|---|---|
| `trace_day.csv` | 40,000 requests of one day: tenant, language, chat or document, conversation and turn, token counts, difficulty |
| `daily_demand.csv`, `traffic_profile.json` | Requests per minute, and the evening peak (7,656 requests in its busiest hour) |
| `tenants.json` | The three tenants and Musamir's prompt as deployed |
| `objectives.json` | TTFT 2.0 s, TPOT 150 ms, longest stall 1.0 s, 90% of requests |
| `deployments/` | The baseline, three seeded faults, Day 2's tuned engine and the capstone's start |
| `eval/` | 600 items (300 Arabic, 300 English, a third with long documents), outcomes of six variants, the floors |
| `spec_acceptance.json` | The draft model's per-token acceptance by language |
| `paraphrase_pairs.jsonl` | 240 question pairs for the semantic-cache lab; about a fifth carry a phone or booking number |
| `prices.json` | The case's prices in SAR: fictional, not vendor prices |
| `reference/` | Supplied runs and the results the safety nets fall back to (`_tools/make_reference.py`) |

## Planted problems

The case data hides problems for the labs to find: a queueing cap, a deployment that preempts under cache
pressure, prefill interference with chunked prefill off, a per-request header at the start of the prompt,
cache blocks shared across tenants, near-identical questions with different answers, a 4-bit variant that
holds English and fails Arabic, a cache format that loses accuracy on long Arabic documents, a draft model
accepted less in Arabic, and a distilled model that is fine on easy requests only.

## Working conventions

- Each notebook starts from a supplied reference state and has a solution in `solutions/`. If an exercise is
  left unfinished, its safety net (`except NotImplementedError`) uses the reference result so the next task runs.
- `SERVE_QUICK=1` makes every lab smaller and faster (the course's checks run that way); without it the labs
  run at the sizes quoted on the slides.
- Raw rows and charts go to `artifacts/<lab>/`, the evidence you carry forward.
- Locally, use `uv`: `cd labs && uv run jupyter lab`. On Windows, if a clone fails on long file names, run
  `git config --global core.longpaths true` first.
