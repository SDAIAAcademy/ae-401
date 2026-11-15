# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "matplotlib>=3.9"]
# ///
"""Capstone starter: measure the provided deployment and keep every raw row.

    cd labs && uv run python capstone/starter.py            # about two minutes on a laptop
    SERVE_QUICK=1 uv run python capstone/starter.py         # a smaller run to check the setup

It writes into capstone/results/ (create your own repository from labs/ and keep this folder in it):
  baseline_runs.csv   one row per request, three runs at each concurrency level
  baseline_curve.csv  goodput and attainment per level, mean and standard deviation
  quality.csv         Arabic and English scores of every recorded variant, with Wilson intervals
  isolation.txt       the tenant isolation test on the provided deployment

Then change one thing at a time with `cfg.but(...)`, rerun `curve(cfg, name)` and keep each run's rows.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import pandas as pd  # noqa: E402

import servelab as S  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
OBJ = S.objectives()
LEVELS = S.pick((2, 4, 8, 16, 32), (2, 8, 16))
RUNS = S.pick(3, 2)


def curve(cfg, name):
    """Goodput over concurrency, RUNS runs a level; returns the summary and writes the raw rows."""
    rows, summ = [], []
    for users in LEVELS:
        for seed in range(1, RUNS + 1):
            run = S.run_closed(cfg, users, n_requests=S.pick(users * 8, users * 4), seed=seed)
            s = run.summary()
            for r in run.table():
                rows.append({"config": name, "users": users, "seed": seed, **r})
            summ.append({"config": name, "users": users, "seed": seed, **{k: s[k] for k in
                         ("throughput_rps", "goodput_rps", "attainment", "ttft_p90", "tpot_p90", "stall_p90", "preemptions", "hit_rate")}})
    pd.DataFrame(rows).to_csv(OUT / f"{name}_runs.csv", index=False)
    per = pd.DataFrame(summ).groupby("users")[["goodput_rps", "attainment"]].agg(["mean", "std"]).round(3)
    per.to_csv(OUT / f"{name}_curve.csv")
    return per


def quality():
    items = {i["item"]: i for i in S.eval_items()}
    rows = []
    for variant, out in S.eval_outcomes().items():
        for lang in ("ar", "en"):
            ids = [i for i in out if items[i]["lang"] == lang]
            k = sum(out[i] for i in ids)
            lo, hi = S.wilson(k, len(ids))
            rows.append({"variant": variant, "lang": lang, "score": round(k / len(ids), 3), "low": lo, "high": hi,
                         "floor": S.floors()[lang], "passes": k / len(ids) >= S.floors()[lang]})
    pd.DataFrame(rows).to_csv(OUT / "quality.csv", index=False)
    return pd.DataFrame(rows)


def isolation(cfg):
    run = S.run_closed(cfg, 6, n_requests=S.pick(200, 100), seed=1)
    bad = [r for r in run.table() if r["cross_tenant_blocks"] > 0]
    text = (f"{len(bad)} of {len(run.table())} requests reused another tenant's cache blocks "
            f"(hit rate {run.summary()['hit_rate']}; a layout that reuses nothing passes trivially: rerun after each change)")
    (OUT / "isolation.txt").write_text(text + "\n", encoding="utf-8")
    return text


if __name__ == "__main__":
    start = S.Config.load("capstone_start")
    print("provided deployment:", start)
    print("cache blocks at start:", S.cache_blocks(start))
    print(curve(start, "baseline").to_string())
    print(quality().to_string(index=False))
    print("isolation:", isolation(start))
    print("written to", OUT.relative_to(LABS))
    if os.environ.get("SERVE_QUICK") != "1":
        print("Next: change one setting, run curve(cfg, name) again, and score the variant you serve.")
