"""Writes labs/data/reference/: the supplied runs and the reference results of the safety nets.

    uv run python _tools/make_reference.py [name ...]

Every supplied run comes from the course simulator (servelab) with a fixed seed and is marked synthetic.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import servelab as S  # noqa: E402

REF = LABS / "data" / "reference"
REF.mkdir(parents=True, exist_ok=True)
MAKERS = {}


def maker(fn):
    MAKERS[fn.__name__] = fn
    return fn


def put(name, obj):
    (REF / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def put_csv(name, rows):
    with open(REF / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@maker
def d1_1():
    m = S.models()["qwen3-8b"]
    ridge = 121e12 * S.CAL["compute_eff"] / (300e9 * S.CAL["mem_eff"])
    ops, b = S.flops_per_token(m), S.weight_bytes(m)
    batch = 1
    while batch * ops / b < ridge:
        batch += 1
    put("d1_1", {"ridge": ridge, "batch_at_ridge": batch, "synthetic": False})


@maker
def d1_2():
    m = S.models()["qwen3-8b"]
    free = 24e9 * 0.90 - S.weight_bytes(m) - 1.5 * 2**30
    kvt = S.kv_bytes_per_token(m)
    put("d1_2", {"cache_bytes": free, "predicted": {L: free / (L * kvt) for L in (2000, 4000, 8000)}})


@maker
def d1_3():
    import pandas as pd
    cfg = S.Config.load("baseline")
    parts = []
    for users in (2, 8, 24):
        t = pd.DataFrame(S.run_closed(cfg, users, n_requests=200, seed=1).table())
        parts.append({"users": users, "queue_s": t.queue_s.mean(), "prefill_s": (t.ttft_s - t.queue_s).mean(),
                      "decode_s": (t.e2e_s - t.ttft_s).mean()})
    put("d1_3", {"parts": parts, "synthetic": True})


@maker
def curve_runs():
    """Lab 1.4: the baseline at six concurrency levels, three repeats each, raw rows (synthetic)."""
    cfg = S.Config.load("baseline")
    rows = []
    for users in (1, 2, 4, 8, 12, 16):
        for rep in (1, 2, 3):
            run = S.run_closed(cfg, users, n_requests=max(160, users * 14), seed=100 + rep)
            for x in run.table():
                rows.append({"users": users, "repeat": rep, **{k: x[k] for k in (
                    "id", "lang", "kind", "arrival", "prompt_tokens", "out_tokens", "queue_s", "ttft_s", "tpot_s", "e2e_s", "met")},
                    "end": round(x["arrival"] + x["e2e_s"], 3)})
    put_csv("curve_runs", rows)


@maker
def d1_4():
    cfg = S.Config.load("baseline")
    rows = []
    for rep in (1, 2, 3):
        run = S.run_closed(cfg, 6, n_requests=100, seed=200 + rep)
        for x in run.table():
            rows.append({"users": 6, "repeat": rep, **x, "end": x["arrival"] + x["e2e_s"]})
    put("d1_4", {"own_rows": rows, "synthetic": True})


@maker
def d2_1():
    q_cfg = S.Config.load("fault_queueing")
    rows = []
    for cap in (2, 4, 6, 8, 12):
        s = S.run_closed(q_cfg.but(max_num_seqs=cap), 6, n_requests=140, seed=1).summary()
        rows.append({"cap": cap, **{k: s[k] for k in ("throughput_rps", "goodput_rps", "attainment", "ttft_p90", "tpot_p90", "preemptions")}})
    put("d2_1", {"caps": rows, "synthetic": True})


@maker
def d2_2():
    import pandas as pd
    cfg = S.Config.load("fault_interference")
    pool = S.mixed_pool(share_docs=0.35, n=400, seed=1)
    rows = []
    for b in ("off", 512, 1024, 2048):
        c = cfg if b == "off" else cfg.but(enable_chunked_prefill=True, max_num_batched_tokens=b)
        run = S.run_closed(c, 5, pool=pool, n_requests=180, seed=1)
        t = pd.DataFrame(run.table())
        t["stall_s"] = [r.max_gap for r in run.rows]
        ch, d = t[t.kind == "chat"], t[t.kind == "doc"]
        rows.append({"budget": b, "attainment": run.summary()["attainment"], "chat_tpot_p90": round(ch.tpot_s.quantile(.9), 3),
                     "chat_stall_p90": round(ch.stall_s.quantile(.9), 3), "doc_ttft_p90": round(d.ttft_s.quantile(.9), 3)})
    put("d2_2", {"rows": rows, "synthetic": True})


@maker
def d2_3():
    put("d2_3", {"order": ["base", "tenant", "document", "history", "header", "question"]})


if __name__ == "__main__":
    names = sys.argv[1:] or list(MAKERS)
    for n in names:
        MAKERS[n]()
        print("wrote", n)
