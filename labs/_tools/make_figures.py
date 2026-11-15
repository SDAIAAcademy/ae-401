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


if __name__ == "__main__":
    for n in sys.argv[1:] or list(FIGS):
        FIGS[n.replace("-", "_")]()
