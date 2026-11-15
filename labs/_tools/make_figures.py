"""Draws the charts of the decks into assets/ from the supplied runs and the course simulator.

    uv run python _tools/make_figures.py [name ...]

Every chart is drawn from data in labs/data/ or from a simulator run with a fixed seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
ASSETS = LABS.parent / "assets"
sys.path.insert(0, str(LABS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import servelab as S  # noqa: E402

plt.rcParams.update({"figure.figsize": (8, 4.2), "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "font.size": 12, "svg.fonttype": "none",
                     "font.family": "sans-serif"})
TEAL, CORAL, NAVY, PURPLE = "#0f8b8d", "#e4572e", "#1d2b53", "#7d5ba6"
FIGS = {}


def fig(fn):
    FIGS[fn.__name__] = fn
    return fn


def save(f, name):
    ASSETS.mkdir(exist_ok=True)
    f.savefig(ASSETS / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(f)
    print("wrote", name)


def per_run(df):
    out = []
    for (u, rep), x in df.groupby(["users", "repeat"]):
        dur = x.end.max() - x.arrival.min()
        out.append({"users": u, "repeat": rep, "goodput": x.met.sum() / dur, "throughput": len(x) / dur})
    return pd.DataFrame(out).groupby("users").agg(["mean", "std"])


@fig
def d1_baseline_curve():
    sup = pd.read_csv(LABS / "data" / "reference" / "curve_runs.csv")
    own = pd.DataFrame(S.reference("d1_4")["own_rows"])
    allr = pd.concat([sup, own[sup.columns.intersection(own.columns)]])
    lv = per_run(allr)
    f, ax = plt.subplots()
    ax.errorbar(lv.index, lv[("throughput", "mean")], yerr=lv[("throughput", "std")], marker="o", c=NAVY, capsize=3, label="throughput")
    ax.errorbar(lv.index, lv[("goodput", "mean")], yerr=lv[("goodput", "std")], marker="o", c=CORAL, capsize=3, label="goodput")
    ax.set(xlabel="concurrent users", ylabel="requests per second")
    ax.legend(frameon=False)
    save(f, "d1-baseline-curve")


@fig
def d4_closing_day():
    import scaling_sim as Z
    d = [4 * x for x in Z.load_demand()]
    f, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot([m / 60 for m in range(len(d))], [x / 60 for x in d], lw=0.9, c=NAVY, label="requests per second, by minute")
    ax.axhline(2.5 * 0.9, c=CORAL, ls="--", lw=1.2, label="one replica at 90% of capacity")
    ax.set(xlabel="hour of the day", ylabel="requests per second", xlim=(0, 24), xticks=range(0, 25, 3))
    ax.legend(frameon=False, loc="upper left")
    save(f, "d4-closing-day")


@fig
def d4_break_even():
    import scaling_sim as Z
    p = S.load_json("prices.json")
    usual = Z.load_demand()
    trace = S.read_trace(limit=40_000)
    mean_in = sum(S.prompt_tokens(r, "stable_first") for r in trace) / len(trace)
    mean_out = sum(r.out_tokens for r in trace) / len(trace)
    hosted = mean_in * p["hosted_input_per_million"] / 1e6 + mean_out * p["hosted_output_per_million"] / 1e6
    xs, ys = [], []
    for k in [0.5 * i for i in range(1, 21)]:
        _, s = Z.simulate([round(k * x) for x in usual], Z.rate_policy(0.5), 2.5, 1, max_replicas=64, cold=8)
        xs.append(s["requests"] / 1000)
        ys.append(1000 * s["replica_hours"] * (p["l4_hour"] + p["node_overhead_hour"]) / s["met"])
    f, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(xs, ys, marker="o", c=TEAL, label="node, per 1,000 successful requests")
    ax.axhline(1000 * hosted, c=CORAL, ls="--", lw=1.2, label="hosted quote, per 1,000 requests")
    ax.set(xlabel="thousand requests a day", ylabel="SAR per 1,000 requests", ylim=(0, 6))
    ax.legend(frameon=False)
    save(f, "d4-break-even")


if __name__ == "__main__":
    for n in sys.argv[1:] or list(FIGS):
        FIGS[n.replace("-", "_")]()
