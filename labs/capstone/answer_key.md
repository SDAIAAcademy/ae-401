# Capstone: what a strong submission finds (for trainers)

The capstone is open: several answers are defensible. These are the results of the course's reference path,
run with the simulator and the recorded outcomes, so a trainer can tell a sound result from an error. All
numbers are synthetic.

## The three diagnoses in the provided deployment

| Symptom | Evidence | Fix | Rerun |
|---|---|---|---|
| Answers freeze mid-sentence | Stall p90 over 2 s with chunked prefill off | Chunked prefill, budget 1,024 | Stall p90 about 0.35 s (Lab 2.2) |
| First tokens late from 8 users | Cache full, preemptions; 1,525 blocks in 16 bits | 8-bit weights, then an 8-bit cache | 5,001 then 10,002 blocks (Labs 3.1, 3.2) |
| No reuse at all | Hit rate 0: the header comes first | Stable parts first, salt per tenant | Hit rate about 55%, TTFT p50 432 to 90 ms (Lab 2.3) |

The cap of 48 is not a fault once the cache is large enough; with 16-bit weights it causes preemption at high
load. A submission that lowers the cap without fixing memory gets a smaller tail and a smaller goodput.

## The reference path

| Step | Goodput or attainment | Quality |
|---|---|---|
| Provided deployment | About 0.15 good requests per second at its knee | Arabic 0.763, English 0.843 |
| Day 2 settings (cap 8, chunked 1,024, layout, salt) | About 0.49 at 8 users | Unchanged |
| 8-bit weights | 82% of the first five minutes of the peak | 0.757, 0.837 |
| 8-bit weights and cache | 100% of the same; about 2.5 good requests per second per card | 0.740, 0.833; long Arabic −4.8 points (paired interval −9.7 to −1.0) |
| Routing under 0.4 to Musamir-mini | Cost per million output tokens 2.58 to 2.00 SAR | 0.737, 0.833 |

Errors to look for: 4-bit weights served (fails the Arabic floor); speculation on at the peak (attainment falls);
a semantic cache (no threshold meets one wrong answer in 1,000); the floors judged on an average of both
languages; a single run per point.

## Topology and cost

- No disaggregation under the chat objective: two cards together serve 96.5% against 93.9% split; split pays only
  with a stall limit near 150 ms (Lab 4.1).
- Router by conversation with a load limit: 99.4% on the closing night against 93.5% for round robin; by tenant,
  52% (Lab 4.2).
- Scaling on the request rate at 50% of capacity, at least one replica: 49.6 replica-hours and 203 SAR for the
  closing day, against 492 SAR for five cards all day (Lab 4.3).
- Cost per successful request on a usual day: 2.69 SAR per 1,000 against 1.05 for the hosted quote; break-even near
  360,000 requests a day; about 117 Wh per 1,000 requests (Lab 4.4).

A strong defence says plainly that at Liwan's volume the hosted quote is cheaper per request, and names what the
quote does not price (where the data is processed, dependence on a provider) as the Director's decision.

## Questions for the panel

- Hanadi: "Which single number would you put on my dashboard, and what value would make you act?"
- Rawya: "Show me the test that proves no tenant reuses another's cache, and when it runs."
- Kenan: "Your cost per million tokens is lower than the quote. Why do you say hosted is cheaper?"
- Asrar: "What does the 8-bit cache cost an Arabic visitor who pastes the event terms?"
