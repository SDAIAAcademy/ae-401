# Capstone: tune Musamir to its objective at the lowest cost per successful request

Liwan National Events Authority gives you Musamir as it was deployed before this course
(`data/deployments/capstone_start.json`): Qwen3-8B in 16 bits on one NVIDIA L4, a sequence cap of 48,
chunked prefill off, a prompt that starts with a per-request header and no cache salt. Hanadi's brief:

- **Objective.** At the evening peak, 90% of requests with a time to first token of at most 2.0 s, a time
  per output token of at most 150 ms and no pause between tokens over 1.0 s.
- **Quality.** Arabic at least 0.73 and English at least 0.81 on the evaluation set.
- **Isolation.** No cache block reused across tenants.
- **Personal data.** Never in a cached entry.
- **Cost.** The lowest cost per successful request over a whole day, compared with the hosted quote.

You tune the deployment, re-evaluate every change, design the topology for a usual day and for the closing
day, and defend the result in 12 minutes of talk and 8 minutes of questions.

## What you are given

| File | What it is |
|---|---|
| `data/deployments/capstone_start.json` | The deployment to improve |
| `data/trace_day.csv`, `data/daily_demand.csv` | A synthetic day of 40,000 requests, and requests per minute |
| `data/eval/` | Recorded Arabic and English outcomes of six variants, and the floors |
| `data/prices.json` | The case's prices in SAR (fictional) |
| `servelab.py`, `standin_server.py`, `loadgen.py`, `scaling_sim.py` | The simulator, the streaming stand-in, the load generator and the scaling simulator |
| `capstone/starter.py` | Measures the provided deployment and writes raw rows to `capstone/results/` |

Start with `cd labs && uv run python capstone/starter.py`. With a notebook GPU you may also load a real
engine through `loadgen.py` (set the URL of an OpenAI-compatible server); say which numbers came from it.

## What you submit

A link to a **public GitHub repository**, before the defence, containing:

1. **Raw benchmark files**: every run's per-request rows, three runs per point, and the engine logs (`run.engine_log()`).
2. **Quality by language**: the scores of the variant you serve, with intervals, and paired differences where two
   variants answered the same items.
3. **A design document** (about four pages): the three diagnoses, each change with its measurement and
   re-evaluation, the topology (disaggregation or not, router, replicas, scaling policy), the cost per successful
   request against the hosted quote at three volumes with the break-even, and the energy per thousand requests.
4. **The isolation test** as code that runs on every benchmark.
5. **A commit history** that shows the work in order.

## How it is scored

| Criterion | Weight | Full marks look like |
|---|---|---|
| Cost model | 15 | Cost per successful request over a replayed day, input and output tokens in the quote, the break-even and its sensitivity to the lease |
| Benchmark method | 25 | Curves over concurrency with three runs a point, the objective's three clocks, raw rows kept, open loop for the peak |
| Optimisations with re-evaluation | 25 | One change at a time, each with goodput and quality per language, paired differences, what the Director must accept |
| Topology design | 20 | A router with a load limit, a scaling signal and minimum justified by cold start and cost, a stated condition for disaggregation |
| Reproducibility | 15 | Someone else reruns your repository and gets your tables |

AI assistants may be used throughout the capstone. Every number in the defence must come from a run in your
repository.

In the questions the assessor changes the traffic mix (for example the share of long documents or of Arabic
requests): you predict the effect on goodput first, then rerun one point from your repository. Keep the
command that reruns a point ready.

The capstone rubric is 60% of the course mark, the live defence 20% and the supervised entry and exit checks
20%. Passing is 70% overall and at least 60% on both the capstone rubric and the supervised check.

## The follow-up task

Within a month, repeat the method on a deployment of your own or on a public trace: baseline curve, one
change with its re-evaluation, cost per successful request. Submit the repository link the same way.
