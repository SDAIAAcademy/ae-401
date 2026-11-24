"""Stage timings of photo questions, from the client's clock: the tables of Day 1, Session 3.

    uv run python _tools/stage_timings.py

A request with a photo passes through stages the engine never sees: the phone prepares the photo,
the network uploads it, the server decodes and resizes it. Only then do queue and prefill start.
This script simulates three synthetic scenarios with a fixed seed and prints, for each, the median
seconds of every stage, its share of the time to first token, the 90th percentile of that time and
the share of requests that meet the 2.0 s objective. Every parameter below is a synthetic teaching
value, not a measurement of a phone, a network or a model. Prefill is timed with the course
simulator's step time on the L4 for the given number of prompt tokens.
"""
from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import servelab as S  # noqa: E402

N = 400
TTFT_OBJECTIVE = S.objectives()["ttft_s"] if "ttft_s" in S.objectives() else 2.0
MODEL_LONG_SIDE = 1024      # the pilot model resizes every photo to this long side, whatever it receives
SERVER_RESIZE_S_PER_MP = 0.012
ENCODE_S = 0.06             # the vision encoder of the pilot, per photo (synthetic)
RTT_S = 0.08


def lognorm(rng, median, sigma):
    return median * math.exp(rng.gauss(0, sigma))


def prefill_s(tokens):
    cfg = S.Config.load("baseline")
    t, left = 0.0, tokens
    while left > 0:
        c = min(left, cfg.max_num_batched_tokens)
        t += S.step_time(cfg, c, c * c / 2, 0, 0)
        left -= c
    return t


def run(name, seed, *, photo_mb, photo_mp, uplink_mbps, client_prep_s, queue_s, prompt_tokens, sigma_net=0.45):
    rng = random.Random(seed)
    pre = prefill_s(prompt_tokens)
    rows = []
    for _ in range(N):
        size = lognorm(rng, photo_mb, 0.25)
        bw = lognorm(rng, uplink_mbps, sigma_net)
        r = {
            "client prepares the photo": lognorm(rng, client_prep_s, 0.3) if client_prep_s else 0.0,
            "upload": RTT_S + size * 8 / bw,
            "server decodes and resizes": photo_mp * SERVER_RESIZE_S_PER_MP * math.exp(rng.gauss(0, 0.1)),
            "queue": lognorm(rng, queue_s, 0.5),
            "image encoding and prefill": (ENCODE_S + pre) * math.exp(rng.gauss(0, 0.05)),
        }
        r["ttft"] = sum(r.values())
        rows.append(r)
    stages = [k for k in rows[0] if k != "ttft"]
    med = {k: statistics.median(x[k] for x in rows) for k in stages}
    mean = {k: statistics.fmean(x[k] for x in rows) for k in stages}
    mean_ttft = statistics.fmean(x["ttft"] for x in rows)
    ttfts = sorted(x["ttft"] for x in rows)
    p90 = ttfts[int(0.9 * N) - 1]
    met = sum(t <= TTFT_OBJECTIVE for t in ttfts) / N
    print(f"\n== {name}  (seed {seed}, {N} photo requests, {photo_mb} MB median photo, {uplink_mbps} Mbit/s median uplink)")
    print(f"{'stage':32s} {'median s':>9s} {'share of mean TTFT':>19s}")
    for k in stages:
        print(f"{k:32s} {med[k]:9.2f} {mean[k] / mean_ttft:18.0%}")
    print(f"{'time to first token, median':32s} {statistics.median(ttfts):9.2f}")
    print(f"{'time to first token, p90':32s} {p90:9.2f}")
    print(f"{'meet TTFT <= ' + str(TTFT_OBJECTIVE) + ' s':32s} {met:9.0%}")
    return med, p90, met


def image_tokens(long_side, short_side, patch=28):
    """Tokens of the resized photo, for a model that cuts it into square patches (a synthetic rule)."""
    return math.ceil(long_side / patch) * math.ceil(short_side / patch)


if __name__ == "__main__":
    toks = image_tokens(MODEL_LONG_SIDE, 768) + 300
    print(f"photo prompt: {toks} tokens (image at {MODEL_LONG_SIDE} x 768 plus 300 text tokens);"
          f" prefill alone on the L4 {prefill_s(toks):.2f} s")
    # 1. Musamir's photo pilot: visitors photograph a sign or a printed programme at the venue, full camera size
    run("Musamir photo pilot, as deployed", 4011, photo_mb=3.4, photo_mp=12, uplink_mbps=7,
        client_prep_s=0.0, queue_s=0.12, prompt_tokens=toks)
    # 1b. the same requests after the app resizes to the model's long side and recompresses before upload
    run("Musamir photo pilot, resized on the phone", 4011, photo_mb=0.24, photo_mp=0.79, uplink_mbps=7,
        client_prep_s=0.07, queue_s=0.12, prompt_tokens=toks)
    # 2. Parallel case: Athar Heritage Week guides on venue Wi-Fi; their app already sends small photos
    run("Athar guides' artefact photos, evening peak", 4012, photo_mb=0.38, photo_mp=1.2, uplink_mbps=40,
        client_prep_s=0.08, queue_s=2.4, prompt_tokens=toks)
    # 3. Practice question: Qamra Festival wristband photos over a crowded mobile network
    run("Qamra wristband photos", 4013, photo_mb=2.9, photo_mp=12, uplink_mbps=6,
        client_prep_s=0.0, queue_s=0.09, prompt_tokens=toks)
