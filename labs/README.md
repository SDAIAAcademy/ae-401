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
