---
course: "Inference Performance and Cost Engineering"
code: "AE-401"
version: "2027.2.0"
deck: "capstone"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-401/slides/capstone.html"
---

# Inference Performance and Cost Engineering

## Capstone: Capstone project presentations

Twelve minutes of evidence and eight of questions, with one traffic change predicted and rerun

#### [capstone#title] Capstone project presentations

*SDAIA Academy · Inference Performance and Cost Engineering*

Capstone: Twelve minutes of evidence and eight of questions, with one traffic change predicted and rerun

AI Engineering Track · 4 teaching days + capstone day · 12 min talk and demo · 8 min Q&A · AE-401

#### [capstone#agenda] How the presentations run

*Capstone*

- **Expectations and final prep**: the rubric, a last rehearsal, then the first presentations
- **Presentations**: each one timed, with questions after it
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#the-task] Monday's deployment, tuned to the objective at the lowest cost per successful request

*Capstone*

- **Hanadi's brief.** 90% of peak requests within 2.0 s to the first token, 150 ms a token and no pause over 1.0 s; Arabic at least 0.73 and English 0.81; no cache block across tenants; no personal data in a cached entry.
- **The evidence.** A public repository: raw benchmark rows, engine logs, quality by language, the isolation test, a design document with cost and energy per request, and the commit history.

#### [capstone#rubric] The rubric

*How presentations are scored*

| Item | Value |
| --- | --- |
| Cost model | 15% |
| Benchmark method | 25% |
| Optimisations with re-evaluation | 25% |
| Topology design | 20% |
| Reproducibility | 15% |

> **Three runs a point, raw rows in the repository.** A claim without a number is a plan, not a result.

#### [capstone#format] 12 minutes of talk, 8 minutes of questions

*Format*

- **1 · The baseline.** The curve of the provided deployment and the three diagnoses.
- **2 · What you changed.** Each change with goodput and quality per language; the topology.
- **3 · The decision.** Cost per successful request against the quote, and what the Director must accept.

> **In the questions.** The assessor changes the traffic mix; you predict the goodput effect, then rerun one point.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Project |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first presentation*

Rehearse once, record a backup, try the demo on this screen.

#### [capstone#now-presenting] Talk

*Now presenting*

##### Questions

#### [capstone#review] What the presentations showed

*Trainer review*

- **Strongest patterns.** What several projects got right.
- **Common gaps.** What most projects would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#what-next] Keep going

*After the course*

- **Your project stays yours:** keep it reproducible and add the next improvement you named today
- **The follow-up task:** within a month, repeat the method on your own deployment or a public trace: baseline curve, one change with its re-evaluation, cost per successful request

#### [capstone#thanks] Thank you

*SDAIA Academy*

Inference Performance and Cost Engineering
