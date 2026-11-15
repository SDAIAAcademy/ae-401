---
course: "Inference Performance and Cost Engineering"
code: "AE-401"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-401/slides/day3.html"
---

# Inference Performance and Cost Engineering

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
