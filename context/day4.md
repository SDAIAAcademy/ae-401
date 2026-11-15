---
course: "Inference Performance and Cost Engineering"
code: "AE-401"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-401/slides/day4.html"
---

# Inference Performance and Cost Engineering

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
