---
course: "Inference Performance and Cost Engineering"
code: "AE-401"
version: "2027.2.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-401/slides/day2.html"
---

# Inference Performance and Cost Engineering

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
