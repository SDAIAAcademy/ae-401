# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 2.4 · Semantic cache thresholds
#
# Day 2 · Session 4 (Semantic caching and wrong answers) · 25 min.
#
# A vendor offers Hanadi a semantic cache: when a new question is similar enough to one already answered, the
# stored answer is returned without calling the model. You score 240 Arabic and English pairs (a question
# already answered, a new question, and whether the stored answer is right for the new one) at three
# similarity thresholds, count hits and wrong answers by language, then keep personal data out of every
# cached entry, as Rawya requires.
#
# **AI tutor allowed**: it may explain similarity measures; it does not choose the threshold.
#
# **Starts from** `data/paraphrase_pairs.jsonl` (synthetic; 20% of the questions carry a phone or booking number).
#
# **Tasks**
# - [ ] 1 · Embed the pairs with character n-grams (5 min)
# - [ ] 2 · Hit rate and wrong answers at three thresholds (7 min)
# - [ ] 3 · The same rates by language (4 min)
# - [ ] 4 · Mask, then refuse to cache personal data (5 min)
# - [ ] 5 · Choose a threshold, or no semantic cache (4 min)
# - [ ] ✱ Stretch · Key the cache on the event and the date as well as the text

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ae-401.git"
    if not Path("/content/ae-401").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ae-401"], check=True)
    os.chdir("/content/ae-401/labs")
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import json, math, re
from collections import Counter
import pandas as pd
import servelab as S

pairs = [json.loads(line) for line in (S.DATA / "paraphrase_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
df = pd.DataFrame(pairs)
print(df.groupby(["lang", "kind"]).size().unstack())
print(df[["lang", "kind", "cached", "new", "same_answer"]].head(4).to_string(index=False))

# %% [markdown]
# ## 1 · Embed the pairs with character n-grams (5 min)
#
# A production cache uses a neural embedding model; here a character trigram vector stands in for it, with no
# download. It is enough to show the problem: wording close, meaning different.
#
# ✏️ **Exercise** · `embed(text)` lower-cases the text, removes punctuation, and returns a `Counter` of its
# character trigrams. `cosine(a, b)` is given.

# %%
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def embed(text):
    # BEGIN SOLUTION
    t = " " + PUNCT.sub(" ", text.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    return Counter(t[i:i + 3] for i in range(len(t) - 2))
    # END SOLUTION


def cosine(a, b):
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def _embed_ref(text):
    t = re.sub(r"\s+", " ", " " + PUNCT.sub(" ", text.lower()) + " ")
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


try:
    df["sim"] = [cosine(embed(a), embed(b)) for a, b in zip(df.cached, df.new)]
except NotImplementedError:
    df["sim"] = [cosine(_embed_ref(a), _embed_ref(b)) for a, b in zip(df.cached, df.new)]
print(df.groupby("kind").sim.describe()[["mean", "min", "50%", "max"]].round(3))

# %% [markdown]
# 🔍 **Look at** the `slot_changed` row: a question about Friday and a question about Saturday, in the same words,
# are more alike than two honest paraphrases.
#
# ## 2 · Hit rate and wrong answers at three thresholds (7 min)
#
# A **hit** is a pair whose similarity is at or above the threshold: the cached answer is served. A hit is a
# **wrong answer** when `same_answer` is false. The pairs were drawn as near neighbours from the logs, so the rates
# are those of questions that reach the similarity check, not of all traffic; the comparison between thresholds
# and languages is what counts.
#
# ✏️ **Exercise** · `rates(d, th)` returns hit rate (hits over all pairs), wrong-answer rate (wrong over hits) and
# wrong answers per 1,000 questions.

# %%
THRESHOLDS = (0.70, 0.80, 0.90)


def rates(d, th):
    # BEGIN SOLUTION
    hit = d.sim >= th
    wrong = hit & ~d.same_answer
    return {"threshold": th, "hit_rate": round(hit.mean(), 3),
            "wrong_of_hits": round(wrong.sum() / max(1, hit.sum()), 3), "wrong_per_1000": round(1000 * wrong.mean(), 1)}
    # END SOLUTION


def _rates_ref(d, th):
    hit = d.sim >= th
    wrong = hit & ~d.same_answer
    return {"threshold": th, "hit_rate": round(hit.mean(), 3),
            "wrong_of_hits": round(wrong.sum() / max(1, hit.sum()), 3), "wrong_per_1000": round(1000 * wrong.mean(), 1)}


try:
    table = pd.DataFrame([rates(df, th) for th in THRESHOLDS])
except NotImplementedError:
    table = pd.DataFrame([_rates_ref(df, th) for th in THRESHOLDS])
print(table.to_string(index=False))
S.save_rows(table.to_dict("records"), "D2_4", "thresholds")
S.check(table.wrong_of_hits.min() > 0.5, "at every threshold most hits are wrong answers")

# %% [markdown]
# ## 3 · The same rates by language (4 min)
#
# ✏️ **Exercise** · The table of task 2 for Arabic and for English separately.

# %%
rows = []
for lang, d in df.groupby("lang"):
    for th in THRESHOLDS:
        try:
            # BEGIN SOLUTION
            rows.append({"lang": lang, **rates(d, th)})
            # END SOLUTION
        except NotImplementedError:
            rows.append({"lang": lang, **_rates_ref(d, th)})
by_lang = pd.DataFrame(rows)
print(by_lang.to_string(index=False))

# %% [markdown]
# ## 4 · Mask, then refuse to cache personal data (5 min)
#
# A cached entry is served to other visitors. An answer about "booking LW-123456" or one written for the visitor
# whose number is in the question must never be stored. Two rules: mask personal data before computing the
# similarity, and refuse to store an entry whose question or answer carries any.
#
# ✏️ **Exercise** · `mask(text)` replaces Saudi mobile numbers (`05` and 8 digits) with `<phone>` and booking
# references (`LW-` and 6 digits) with `<booking>`. `cacheable(question)` is false when masking changed the text.

# %%
PHONE = re.compile(r"\b05\d{8}\b")
BOOKING = re.compile(r"\bLW-\d{6}\b")


def mask(text):
    # BEGIN SOLUTION
    return BOOKING.sub("<booking>", PHONE.sub("<phone>", text))
    # END SOLUTION


def cacheable(question):
    # BEGIN SOLUTION
    return mask(question) == question
    # END SOLUTION


try:
    store = [p["cached"] for p in pairs if cacheable(p["cached"])]
    masked_new = [mask(p["new"]) for p in pairs]
except NotImplementedError:
    def mask(text):
        return BOOKING.sub("<booking>", PHONE.sub("<phone>", text))
    store = [p["cached"] for p in pairs if mask(p["cached"]) == p["cached"]]
    masked_new = [mask(p["new"]) for p in pairs]
leaks = [q for q in store if PHONE.search(q) or BOOKING.search(q)]
print(f"{len(store)} of {len(pairs)} questions may be cached; {len(pairs) - len(store)} refused")
S.check(not leaks, "no cached entry carries a phone or booking number")
S.check(not any(PHONE.search(q) or BOOKING.search(q) for q in masked_new), "lookups are made on masked text")

# %% [markdown]
# ## 5 · Choose a threshold, or no semantic cache (4 min)
#
# Hanadi's limit: at most 1 wrong answer per 1,000 questions. Asrar adds that a wrong answer in Arabic is no more
# acceptable than one in English.
#
# ✏️ **Exercise** · Set `decision` to the threshold that meets the limit in both languages, or to `None` if none
# does, and `reason` in one sentence.

# %%
try:
    # BEGIN SOLUTION
    ok = by_lang.groupby("threshold").wrong_per_1000.max()
    meets = [th for th, w in ok.items() if w <= 1.0]
    decision = min(meets) if meets else None
    reason = ("No threshold keeps wrong answers under 1 in 1,000 in both languages: questions that differ only in "
              "the day or the venue are more similar than true paraphrases. Keep the exact prefix cache; no semantic cache.")
    # END SOLUTION
except NotImplementedError:
    decision, reason = None, "(write your reason)"
print("decision:", decision, "|", reason)
S.check(decision is None, "on this traffic a text-similarity cache cannot meet the limit")

# %% [markdown]
# ✱ **Stretch** · Key the cache on the extracted slots (event, day, venue, ticket) as well as the text: a hit needs
# the same slots and a similarity above the threshold. Recompute the table. (The slots are in `slots_cached` and
# `slots_new`; in production an extraction step would produce them.)

# %%
df["same_slots"] = [a == b for a, b in zip(df.slots_cached, df.slots_new)]
for th in THRESHOLDS:
    hit = (df.sim >= th) & df.same_slots
    print(th, "hit rate", round(hit.mean(), 3), "wrong per 1,000", round(1000 * (hit & ~df.same_answer).mean(), 1))

# %% [markdown]
# ## Check yourself
#
# 1. Why is a high similarity threshold not enough to make a semantic cache safe here?
#    <details><summary>Answer</summary>The dangerous pairs differ in one slot and share almost every character, so they score higher than honest paraphrases.</details>
# 2. Why mask before computing the similarity, and not only before storing?
#    <details><summary>Answer</summary>A phone number changes the similarity and would be written into the cache's index and logs; masking first keeps it out of both.</details>
# 3. The prefix cache of Lab 2.3 never returns a wrong answer. Why?
#    <details><summary>Answer</summary>It reuses computed state for identical tokens only; the model still writes every answer.</details>
#
# **Carry forward** · The threshold decision and its table go into the capstone's design document.
