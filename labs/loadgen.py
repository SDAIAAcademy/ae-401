# /// script
# requires-python = ">=3.11"
# ///
"""A closed-loop load generator over streamed completions (OpenAI-compatible /v1/completions).

    uv run loadgen.py --url http://127.0.0.1:8401 --users 8 --requests 40

Each of `users` threads sends a request, reads the stream to the end and sends the next one. For each
request it records the time to the first token, the time per output token after it, the longest pause
between two tokens and the end-to-end time, measured by the client. Prompts are filler words, one word
per token on the stand-in (roughly so on a real tokenizer); `ignore_eos` asks a real engine to produce exactly `max_tokens`.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from urllib.parse import urlparse

FILLER = ("visitor asks about gates tickets parking venues children access and the festival programme "
          "in arabic and english ").split()


def prompt_of(n):
    return " ".join(FILLER[i % len(FILLER)] for i in range(n))


def one_request(url, prompt_tokens, max_tokens, model="musamir", timeout=600):
    body = json.dumps({"model": model, "prompt": prompt_of(prompt_tokens), "max_tokens": max_tokens,
                       "stream": True, "ignore_eos": True, "temperature": 0}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first, n_tok, t_last, gap = None, 0, None, 0.0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            text = chunk["choices"][0].get("text", "")
            k = max(1, len(text.split())) if text else 0
            if k:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                elif t_last is not None:
                    gap = max(gap, now - t_last)
                n_tok += k
                t_last = now
    t1 = time.perf_counter()
    ttft = (t_first - t0) if t_first else float("inf")
    tpot = ((t_last - t_first) / max(1, n_tok - 1)) if t_first and n_tok > 1 else 0.0
    return {"ttft_s": ttft, "tpot_s": tpot, "stall_s": gap, "e2e_s": t1 - t0, "out_tokens": n_tok, "start": t0, "end": t1}


def run_users(url, users, requests, prompt_tokens=1500, max_tokens=120, jobs=None):
    """Closed loop. `jobs` is an optional list of (prompt_tokens, max_tokens) drawn in turn."""
    lock = threading.Lock()
    rows, counter = [], [0]

    def worker(u):
        while True:
            with lock:
                if counter[0] >= requests:
                    return
                i = counter[0]
                counter[0] += 1
            p, m = jobs[i % len(jobs)] if jobs else (prompt_tokens, max_tokens)
            r = one_request(url, p, m)
            r.update({"user": u, "seq": i, "prompt_tokens": p, "max_tokens": m})
            with lock:
                rows.append(r)

    threads = [threading.Thread(target=worker, args=(u,)) for u in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(rows, key=lambda r: r["seq"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8401")
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--prompt-tokens", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=120)
    a = ap.parse_args()
    if not urlparse(a.url).scheme:
        raise SystemExit("--url needs http:// or https://")
    rows = run_users(a.url, a.users, a.requests, a.prompt_tokens, a.max_tokens)
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
