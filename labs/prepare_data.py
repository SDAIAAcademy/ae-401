# /// script
# requires-python = ">=3.11"
# ///
"""Writes the synthetic case data of AE-401 (Liwan National Events Authority) into labs/data/.

Standard library only, seed 401: the same bytes on every run.

    uv run prepare_data.py

Model cards and accelerator sheets are copied from the official pages listed in labs/README.md;
everything else is synthetic. The planted problems are listed in solutions/answer_key.md.
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import random
from pathlib import Path

SEED = 401
DATA = Path(__file__).resolve().parent / "data"


def write_json(rel, obj):
    p = DATA / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(rel, rows, cols):
    p = DATA / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------- cards (from official pages)

MODEL_CARDS = {
    "qwen3-8b": {
        "source": "https://huggingface.co/Qwen/Qwen3-8B", "licence": "Apache-2.0",
        "params": 8.2e9, "params_nonembedding": 6.95e9, "num_hidden_layers": 36, "hidden_size": 4096,
        "intermediate_size": 12288, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "tie_word_embeddings": False, "max_position_embeddings": 40960,
        "role": "the base model of Musamir"},
    "qwen3-1.7b": {
        "source": "https://huggingface.co/Qwen/Qwen3-1.7B", "licence": "Apache-2.0",
        "params": 1.7e9, "params_nonembedding": 1.4e9, "num_hidden_layers": 28, "hidden_size": 2048,
        "intermediate_size": 6144, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "tie_word_embeddings": True, "max_position_embeddings": 40960,
        "role": "the base of the distilled student"},
    "qwen3-0.6b": {
        "source": "https://huggingface.co/Qwen/Qwen3-0.6B", "licence": "Apache-2.0",
        "params": 0.6e9, "params_nonembedding": 0.44e9, "num_hidden_layers": 28, "hidden_size": 1024,
        "intermediate_size": 3072, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "tie_word_embeddings": True, "max_position_embeddings": 40960,
        "role": "the draft model for speculative decoding"},
}
# the distilled student: Qwen3-1.7B tuned on Musamir's traffic by the authority (fictional)
MODEL_CARDS["musamir-mini"] = dict(MODEL_CARDS["qwen3-1.7b"], source="synthetic: Qwen3-1.7B distilled from Musamir's base on the authority's traffic",
                                   role="the distilled student (fictional)")

ACCELERATORS = {
    "L4": {"source": "https://www.nvidia.com/en-us/data-center/l4/", "memory_gb": 24, "bandwidth_gbs": 300,
           "sheet": {"BF16 tensor TFLOPS (with sparsity)": 242, "FP8 tensor TFLOPS (with sparsity)": 485,
                     "INT8 tensor TOPS (with sparsity)": 485, "GPU memory GB": 24, "memory bandwidth GB/s": 300,
                     "max TDP W": 72},
           "fp16_dense_tflops": 121, "fp8_dense_tflops": 242.5, "int8_dense_tops": 242.5, "tdp_w": 72,
           "fp8": True, "bf16": True,
           "note": "The sheet shows tensor-core figures with sparsity (BF16 242, FP8 485); dense is half."},
    "T4": {"source": "https://www.nvidia.com/en-us/data-center/tesla-t4/", "memory_gb": 16, "bandwidth_gbs": 320,
           "sheet": {"FP16 mixed precision TFLOPS": 65, "INT8 TOPS": 130, "GPU memory GB": 16,
                     "memory bandwidth GB/s": 320, "power W": 70},
           "fp16_dense_tflops": 65, "fp8_dense_tflops": None, "int8_dense_tops": 130, "tdp_w": 70,
           "fp8": False, "bf16": False,
           "note": "Mixed precision FP16/FP32 65 TFLOPS, INT8 130 TOPS; no FP8 or BF16 tensor cores."},
}

# ---------------------------------------------------------------- the case

TENANTS = {
    "qamra": {"name": "Qamra Festival", "name_ar": "مهرجان قمرة", "tokens": 512},
    "sawt": {"name": "Sawt Concerts", "name_ar": "حفلات صوت", "tokens": 448},
    "athar": {"name": "Athar Heritage Week", "name_ar": "أسبوع أثر للتراث", "tokens": 576},
}
SHARE = {"qamra": 0.45, "sawt": 0.35, "athar": 0.20}

PROMPT_DEPLOYED = """[header] Request {request_id}. Visitor: {visitor_first_name}. Received at second {seconds_of_day} of the day.
[base] You are Musamir, the visitor assistant of Liwan National Events Authority. Answer in the visitor's language...
[tenant] Event rules and venue notes of {tenant}...
[document] The event's terms, when the visitor asks about them...
[history] The earlier turns of this conversation...
[question] {question}"""


def day_weights():
    """Relative arrivals per minute over a day: quiet night, a midday bump, the evening peak."""
    w = []
    for m in range(1440):
        h = m / 60
        night = 0.06
        midday = 0.35 * math.exp(-((h - 13.0) / 2.2) ** 2)
        afternoon = 0.45 * math.exp(-((h - 17.0) / 1.8) ** 2)
        evening = 1.45 * math.exp(-((h - 20.5) / 1.35) ** 2)
        w.append(night + midday + afternoon + evening)
    return w


def lognorm(rng, median, sigma, lo, hi):
    return int(min(hi, max(lo, round(median * math.exp(rng.gauss(0, sigma))))))


def make_trace(rng):
    weights = day_weights()
    cum, s = [], 0.0
    for x in weights:
        s += x
        cum.append(s)
    docs = {t: [(f"{t}-terms-{i:02d}", lognorm(rng, 3600, 0.25, 2200, 6400)) for i in range(1, 13)] for t in TENANTS}
    rows, sid = [], 0
    target = 40000
    while len(rows) < target:
        sid += 1
        m = bisect.bisect_left(cum, rng.random() * s)
        t = m * 60 + rng.random() * 60
        tenant = rng.choices(list(SHARE), weights=list(SHARE.values()))[0]
        lang = "ar" if rng.random() < 0.55 else "en"
        kind = "doc" if rng.random() < 0.18 else "chat"
        doc_id, doc_tokens = rng.choice(docs[tenant]) if kind == "doc" else ("", 0)
        turns = rng.choices([1, 2, 3, 4], weights=[50, 25, 15, 10])[0]
        hist = []
        for k in range(1, turns + 1):
            if len(rows) >= target or t >= 86400:
                break
            qmed = 38 * (1.35 if lang == "ar" else 1.0)
            q = lognorm(rng, qmed, 0.45, 8, 220)
            omed = (175 if kind == "doc" else 115) * (1.2 if lang == "ar" else 1.0)
            o = lognorm(rng, omed, 0.5, 16, 520)
            rows.append({"id": f"R{len(rows) + 1:05d}", "t": round(t, 3), "tenant": tenant, "lang": lang, "kind": kind,
                         "session": f"S{sid:05d}", "turn": k, "doc_id": doc_id, "doc_tokens": doc_tokens,
                         "hist_tokens": sum(hist), "hist_parts": "|".join(str(x) for x in hist),
                         "q_tokens": q, "out_tokens": o, "difficulty": round(rng.gauss(0, 1), 3)})
            hist.append(q + o)
            t += 25 + rng.random() * 70
    rows.sort(key=lambda r: r["t"])
    for i, r in enumerate(rows, 1):
        r["id"] = f"R{i:05d}"
    return rows


def make_eval(rng):
    """600 items, 300 per language; recorded outcomes of six variants with common random numbers."""
    items, outcomes = [], []
    skill = {"en": 2.05, "ar": 1.70}
    for lang in ("ar", "en"):
        for i in range(300):
            d = rng.gauss(0, 1.2)
            long_ctx = rng.random() < 0.3
            ctx = lognorm(rng, 9000, 0.25, 6000, 16000) if long_ctx else lognorm(rng, 1400, 0.4, 600, 3500)
            score = 1 / (1 + math.exp(-(1.1 * d + rng.gauss(0, 0.55))))
            task = rng.choice(["answer from event rules", "ticket change", "venue directions", "accessibility", "terms question"])
            items.append({"item": f"{lang.upper()}-{i + 1:03d}", "lang": lang, "task": task, "context": "long" if long_ctx else "short",
                          "context_tokens": ctx, "difficulty": round(d, 3), "classifier_score": round(score, 4)})
    for it in items:
        u = rng.random()
        d, lang, long_ctx = it["difficulty"], it["lang"], it["context"] == "long"
        base = skill[lang]
        pen = {
            "base_bf16": 0.0,
            "spec_draft": 0.0,
            "w_fp8": 0.04,
            "w_int4": 0.12 if lang == "en" else 0.55,
            "kv_fp8": (0.12 if lang == "en" else 0.42) if long_ctx else 0.02,
            "mini": 0.15 + 1.25 * max(0.0, d + 0.3) * (1.25 if lang == "ar" else 1.0),
        }
        for v, p in pen.items():
            prob = 1 / (1 + math.exp(-(base - p - d)))
            outcomes.append({"variant": v, "item": it["item"], "correct": int(u < prob)})
    return items, outcomes


EVENTS = [("Qamra Festival", "مهرجان قمرة"), ("Sawt Concerts", "حفلات صوت"), ("Athar Heritage Week", "أسبوع أثر للتراث")]
DAYS = [("Thursday", "الخميس"), ("Friday", "الجمعة"), ("Saturday", "السبت")]
VENUES = [("the North Gate", "البوابة الشمالية"), ("Lake Square", "ساحة البحيرة"), ("the Old Market", "السوق القديم")]
TICKETS = [("family", "العائلية"), ("VIP", "كبار الزوار"), ("day pass", "اليومية")]

INTENTS = {
    "opening": {
        "en": ["When do the gates open for {ev} on {day}?", "On {day}, when does {ev} open its gates?",
               "What is the gate opening for {ev} this {day}?", "Tell me when {ev} lets visitors in on {day}."],
        "ar": ["متى تفتح بوابات {ev} يوم {day}؟", "يوم {day}، متى يفتح {ev} أبوابه؟",
               "ما موعد فتح البوابات في {ev} هذا {day}؟", "أخبرني متى يبدأ دخول الزوار إلى {ev} يوم {day}."],
        "slots": ("ev", "day")},
    "parking": {
        "en": ["Is there parking near {venue} for {ev}?", "Where can I park close to {venue} during {ev}?",
               "Does {ev} have car parking by {venue}?", "Parking next to {venue} for {ev}: is it available?"],
        "ar": ["هل توجد مواقف سيارات قرب {venue} في {ev}؟", "أين أوقف سيارتي بالقرب من {venue} أثناء {ev}؟",
               "هل لدى {ev} مواقف بجانب {venue}؟", "مواقف بجوار {venue} في {ev}: هل هي متاحة؟"],
        "slots": ("venue", "ev")},
    "ticket_change": {
        "en": ["Can I move my {ticket} ticket for {ev} to {day}?", "Is it possible to change a {ticket} ticket of {ev} to {day}?",
               "I want to switch my {ev} {ticket} ticket to {day}. Can I?", "Could my {ticket} ticket for {ev} be used on {day} instead?"],
        "ar": ["هل يمكنني نقل تذكرتي {ticket} في {ev} إلى يوم {day}؟", "هل يمكن تغيير تذكرة {ev} {ticket} إلى يوم {day}؟",
               "أريد تحويل تذكرة {ev} {ticket} إلى يوم {day}. هل يمكن ذلك؟", "هل تصلح تذكرتي {ticket} في {ev} ليوم {day} بدلا من ذلك؟"],
        "slots": ("ticket", "ev", "day")},
    "access": {
        "en": ["Is {venue} wheelchair accessible during {ev}?", "Can a wheelchair user reach {venue} at {ev}?",
               "Does {ev} have step-free access at {venue}?", "Wheelchair access at {venue} for {ev}: is there any?"],
        "ar": ["هل {venue} مهيأ للكراسي المتحركة أثناء {ev}؟", "هل يستطيع مستخدم الكرسي المتحرك الوصول إلى {venue} في {ev}؟",
               "هل في {ev} مدخل بلا درج عند {venue}؟", "الوصول بالكرسي المتحرك إلى {venue} في {ev}: هل هو متاح؟"],
        "slots": ("venue", "ev")},
    "children": {
        "en": ["Are children allowed at {ev} on {day}?", "Can I bring my kids to {ev} on {day}?",
               "Is {ev} open to children this {day}?", "On {day}, may children attend {ev}?"],
        "ar": ["هل يسمح بدخول الأطفال إلى {ev} يوم {day}؟", "هل يمكنني إحضار أطفالي إلى {ev} يوم {day}؟",
               "هل {ev} مفتوح للأطفال هذا {day}؟", "يوم {day}، هل يمكن للأطفال حضور {ev}؟"],
        "slots": ("ev", "day")},
}
SLOT_VALUES = {"ev": EVENTS, "day": DAYS, "venue": VENUES, "ticket": TICKETS}


def fill(tpl, lang, vals):
    i = 0 if lang == "en" else 1
    return tpl.format(**{k: v[i] for k, v in vals.items()})


def personal(rng, lang):
    kind = rng.choice(["phone", "booking"])
    if kind == "phone":
        pd = "05" + "".join(str(rng.randrange(10)) for _ in range(8))
        return (f" My number is {pd}." if lang == "en" else f" رقمي {pd}."), "phone"
    pd = "LW-" + "".join(str(rng.randrange(10)) for _ in range(6))
    return (f" Booking {pd}." if lang == "en" else f" الحجز {pd}."), "booking"


def make_pairs(rng):
    """240 pairs: the cached question and a new one. same_answer says whether the cached answer is right."""
    pairs = []
    for lang in ("ar", "en"):
        for i in range(120):
            intent = rng.choice(list(INTENTS))
            spec = INTENTS[intent]
            vals = {k: rng.choice(SLOT_VALUES[k]) for k in spec["slots"]}
            a_tpl, b_tpl = rng.sample(spec[lang], 2)
            kind = rng.choices(["paraphrase", "slot_changed", "other_intent"], weights=[50, 35, 15])[0]
            vals_b = dict(vals)
            intent_b = intent
            if kind == "slot_changed":
                k = rng.choice(spec["slots"])
                vals_b[k] = rng.choice([v for v in SLOT_VALUES[k] if v != vals[k]])
                if rng.random() < 0.6:
                    b_tpl = a_tpl   # same wording, one slot changed: the dangerous case
            elif kind == "other_intent":
                intent_b = rng.choice([x for x in INTENTS if x != intent])
                specb = INTENTS[intent_b]
                vals_b = {k: vals.get(k, rng.choice(SLOT_VALUES[k])) for k in specb["slots"]}
                b_tpl = rng.choice(specb[lang])
            a = fill(a_tpl, lang, vals)
            b = fill(b_tpl, lang, vals_b)
            pii_a = pii_b = ""
            if rng.random() < 0.2:
                extra, pii_a = personal(rng, lang)
                a += extra
            if rng.random() < 0.2:
                extra, pii_b = personal(rng, lang)
                b += extra
            pairs.append({"pair": f"P{lang.upper()}{i + 1:03d}", "lang": lang, "intent": intent, "kind": kind,
                          "cached": a, "new": b, "same_answer": kind == "paraphrase",
                          "personal_data_cached": pii_a or None, "personal_data_new": pii_b or None,
                          "slots_cached": {k: v[0] for k, v in vals.items()},
                          "slots_new": {k: v[0] for k, v in vals_b.items()}})
    return pairs


def deployments():
    base = {"model": "qwen3-8b", "accelerator": "L4", "weight_dtype": "bf16", "kv_cache_dtype": "bf16",
            "gpu_memory_utilization": 0.90, "max_num_seqs": 48, "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": False, "enable_prefix_caching": True, "cache_salt": "none",
            "block_size": 16, "speculative_model": None, "num_speculative_tokens": 0, "role": "both",
            "layout": "deployed"}
    good = dict(base, max_num_seqs=16, max_num_batched_tokens=2048, enable_chunked_prefill=True)
    return {
        "baseline": {"note": "Musamir as deployed: the configuration the capstone starts from", "config": base},
        "capstone_start": {"note": "The provided under-performing deployment of the capstone", "config": base},
        "fault_queueing": {"note": "Seeded fault 1 (Lab 2.1)", "config": dict(good, max_num_seqs=4)},
        "fault_preemption": {"note": "Seeded fault 2 (Lab 2.1)", "config": dict(good, max_num_seqs=48, gpu_memory_utilization=0.86)},
        "fault_interference": {"note": "Seeded fault 3 (Lab 2.2)", "config": dict(good, enable_chunked_prefill=False, max_num_batched_tokens=8192)},
        "tuned_day2": {"note": "After Day 2: chunked prefill, a cap of 16, stable-first layout, salt per tenant", "config": dict(good, layout="stable_first", cache_salt="tenant")},
    }


def main():
    rng = random.Random(SEED)
    write_json("model_cards.json", MODEL_CARDS)
    write_json("accelerators.json", ACCELERATORS)
    write_json("objectives.json", {
        "ttft_s": 2.0, "tpot_s": 0.10, "attainment_target": 0.90,
        "note": "Per request: time to first token at most 2.0 s and time per output token at most 100 ms. The service objective: 90% of requests at the evening peak meet both."})
    write_json("spec_acceptance.json", {"qwen3-0.6b": {"en": 0.66, "ar": 0.46,
                                                         "note": "synthetic per-token acceptance of the draft model, by language"}})
    write_json("tenants.json", {"header_tokens": 24, "base_tokens": 640, "tenants": TENANTS,
                                "prompt_as_deployed": PROMPT_DEPLOYED,
                                "note": "The base instructions are the same text for every tenant; each tenant adds its own rules."})
    trace = make_trace(rng)
    cols = ["id", "t", "tenant", "lang", "kind", "session", "turn", "doc_id", "doc_tokens", "hist_tokens", "hist_parts",
            "q_tokens", "out_tokens", "difficulty"]
    write_csv("trace_day.csv", trace, cols)
    per_min = [0] * 1440
    for r in trace:
        per_min[int(r["t"] // 60)] += 1
    write_csv("daily_demand.csv", [{"minute": m, "requests": n} for m, n in enumerate(per_min)], ["minute", "requests"])
    peak_min = max(range(1440), key=lambda m: sum(per_min[m:m + 60]))
    write_json("traffic_profile.json", {
        "requests_per_day": len(trace), "peak_start_s": peak_min * 60, "peak_end_s": peak_min * 60 + 3600,
        "peak_requests": sum(per_min[peak_min:peak_min + 60]),
        "share_arabic": round(sum(r["lang"] == "ar" for r in trace) / len(trace), 3),
        "share_documents": round(sum(r["kind"] == "doc" for r in trace) / len(trace), 3),
        "note": "The evening peak is the busiest 60 minutes of the day."})
    items, outcomes = make_eval(rng)
    write_csv("eval/items.csv", items, ["item", "lang", "task", "context", "context_tokens", "difficulty", "classifier_score"])
    write_csv("eval/outcomes.csv", outcomes, ["variant", "item", "correct"])
    base = {lang: sum(o["correct"] for o in outcomes if o["variant"] == "base_bf16" and o["item"].startswith(lang.upper()))
            / 300 for lang in ("ar", "en")}
    write_json("eval/floors.json", {"ar": round(base["ar"] - 0.03, 2), "en": round(base["en"] - 0.03, 2),
                                    "note": "Quality floor per language: the base model's score on the set, less three points, rounded."})
    pairs = make_pairs(rng)
    (DATA / "paraphrase_pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs),
                                                 encoding="utf-8", newline="\n")
    for name, d in deployments().items():
        write_json(f"deployments/{name}.json", d)
    write_json("prices.json", {
        "currency": "SAR",
        "l4_hour": 3.20, "node_overhead_hour": 0.90,
        "hosted_input_per_million": 0.40, "hosted_output_per_million": 1.60,
        "pue": 1.4, "host_power_w": 55,
        "note": "Case figures (fictional): the authority's lease of one L4 accelerator per hour, the host's share, and a hosted-API quote for an 8B-class model. Not vendor prices."})
    print(f"trace {len(trace)} requests | peak {sum(per_min[peak_min:peak_min + 60])} in the busiest hour | "
          f"eval {len(items)} items | pairs {len(pairs)} | base ar {base['ar']:.3f} en {base['en']:.3f}")


if __name__ == "__main__":
    main()
