# /// script
# requires-python = ">=3.11"
# ///
"""The course's scaling simulator: replicas over a day, one minute at a time. Standard library only.

    uv run scaling_sim.py --capacity 2.4 --min 1 --cold 4

Each minute, `demand[m]` requests arrive. A ready replica serves up to `capacity` requests per second within the
objective, up to `max_util` of that before latency fails. Requests beyond what the ready replicas can serve
wait in a backlog: they are served later and count as missed. A scaling policy sees the last minute's arrivals,
the backlog and the replicas, and returns how many replicas it wants. A replica asked for at minute m serves
from minute m + cold; a replica removed stops at once. You pay for every replica that is starting or ready.

This is a fluid model, not the engine simulator: it assumes the objective holds while load stays under
`max_util` of capacity, which is what Lab 1.4's curve and Lab 3.4's capacity runs measured.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load_demand(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent / "data" / "daily_demand.csv"
    with open(path, encoding="utf-8") as f:
        return [int(d["requests"]) for d in csv.DictReader(f)]


def rate_policy(target_util=0.7):
    """Scale on the request rate: enough replicas for last minute's arrivals at `target_util` of capacity."""
    def policy(s):
        return math.ceil(s["arrivals"] / 60 / (s["capacity"] * target_util)) if s["arrivals"] else 0
    return policy


def queue_policy(up_at=20, down_after=10):
    """Scale on the backlog: one more replica while requests wait; one fewer after `down_after` quiet minutes."""
    def policy(s):
        st = s["memory"]
        n = s["wanted"]
        if s["backlog"] > up_at:
            st["quiet"] = 0
            return n + 1
        st["quiet"] = st.get("quiet", 0) + 1 if s["util"] < 0.4 else 0
        if st["quiet"] >= down_after:
            st["quiet"] = 0
            return n - 1
        return n
    return policy


def simulate(demand, policy, capacity, min_replicas=1, max_replicas=8, cold=4, max_util=0.9, start=None):
    """Replay `demand` (requests per minute). Returns the per-minute rows and a summary."""
    per_min = capacity * 60 * max_util
    wanted = start if start is not None else max(min_replicas, 1)
    ready_at = [0] * wanted                     # minute at which each replica is ready
    backlog, rows, memory = 0.0, [], {}
    for m, arrivals in enumerate(demand):
        ready = sum(1 for t in ready_at if t <= m)
        cap = ready * per_min
        ok = min(arrivals, max(0.0, cap - backlog))
        backlog = max(0.0, backlog + arrivals - cap) if cap < backlog + arrivals else 0.0
        util = (arrivals) / cap if cap else (1.0 if arrivals else 0.0)
        rows.append({"minute": m, "arrivals": arrivals, "replicas_paid": len(ready_at), "replicas_ready": ready,
                     "met": ok, "backlog": round(backlog, 1)})
        s = {"minute": m, "arrivals": arrivals, "backlog": backlog, "util": util, "capacity": capacity,
             "wanted": len(ready_at), "ready": ready, "memory": memory}
        new = max(min_replicas, min(max_replicas, policy(s)))
        if new > len(ready_at):
            ready_at += [m + 1 + cold] * (new - len(ready_at))
        elif new < len(ready_at):
            ready_at = sorted(ready_at)[:new]   # stop the newest first
    total = sum(demand)
    met = sum(r["met"] for r in rows)
    replica_hours = sum(r["replicas_paid"] for r in rows) / 60
    hours = {}
    for r in rows:
        h = hours.setdefault(r["minute"] // 60, [0, 0.0])
        h[0] += r["arrivals"]
        h[1] += r["met"]
    worst = min((m / n for n, m in hours.values() if n), default=1.0)
    return rows, {"requests": total, "met": round(met), "attainment": round(met / total, 4) if total else 1.0,
                  "worst_hour": round(worst, 4), "replica_hours": round(replica_hours, 1),
                  "max_replicas": max(r["replicas_paid"] for r in rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=float, default=2.4, help="good requests per second per replica")
    ap.add_argument("--min", type=int, default=1)
    ap.add_argument("--cold", type=int, default=4, help="minutes from asking for a replica to serving")
    ap.add_argument("--signal", choices=("rate", "queue"), default="rate")
    a = ap.parse_args()
    pol = rate_policy() if a.signal == "rate" else queue_policy()
    _, summary = simulate(load_demand(), pol, a.capacity, a.min, cold=a.cold)
    print(summary)
