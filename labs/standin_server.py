# /// script
# requires-python = ">=3.11"
# ///
"""A stand-in for an inference engine: an OpenAI-shaped streaming completions endpoint driven by the
course simulator, in real time (optionally faster, with --scale).

    uv run standin_server.py --port 8401 --deployment baseline --scale 0.25

POST /v1/completions with {"prompt": "...", "max_tokens": n, "stream": true}: the prompt's tokens are
counted as its words (the load generator writes one word per token). The reply streams one event per
token in the format of OpenAI-compatible servers, then "data: [DONE]". GET /health says the time scale;
GET /metrics gives engine gauges. With --scale 0.25 every duration is a quarter of the simulated one:
divide what a client measures by the scale.

It generates no text: every token is the word "tok". It is a scripted stand-in, not a model.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent))
import servelab as S  # noqa: E402


class StandIn:
    def __init__(self, cfg: S.Config, scale=1.0, seed=0):
        self.engine = S.Engine(cfg, seed=seed, name="r0")
        self.scale = scale
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.t0 = time.perf_counter()
        self.track = {}
        self.n = 0
        self.stop_flag = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def sim_now(self):
        return (time.perf_counter() - self.t0) / self.scale

    def submit(self, prompt_tokens, max_tokens):
        q = Queue()
        with self.lock:
            self.n += 1
            r = S.Req(id=f"H{self.n:05d}", t=self.sim_now(), tenant="qamra", lang="en", kind="fixed",
                      session=f"H{self.n:05d}", turn=1, doc_id="", doc_tokens=0, hist_tokens=0, hist_parts="",
                      q_tokens=max(1, prompt_tokens), out_tokens=max(1, max_tokens))
            self.engine.add(r, now=r.t)
            self.track[r.id] = [r, 0, q]
        self.wake.set()
        return q

    def _loop(self):
        while not self.stop_flag:
            events = []
            with self.lock:
                busy = self.engine.has_work()
                if busy:
                    if self.engine.clock < self.sim_now() and not self.engine.running:
                        self.engine.clock = self.sim_now()
                    fin = self.engine.step()
                    target = self.engine.clock
                    for rid, ent in list(self.track.items()):
                        r, sent, q = ent
                        new = r.generated - sent if r.t_first is not None else 0
                        if new > 0:
                            events.append((q, ("tok", new)))
                            ent[1] = r.generated
                    for r in fin:
                        ent = self.track.pop(r.id, None)
                        if ent:
                            if r.generated > ent[1]:
                                events.append((ent[2], ("tok", r.generated - ent[1])))
                            events.append((ent[2], ("done", 0)))
            if not busy:
                self.wake.wait(0.005)
                self.wake.clear()
                continue
            # the tokens of a step leave when the step ends
            delay = target * self.scale - (time.perf_counter() - self.t0)
            if delay > 0:
                time.sleep(delay)
            for q, ev in events:
                q.put(ev)

    def metrics(self):
        e = self.engine
        with self.lock:
            return {"standin:num_requests_running": len(e.running), "standin:num_requests_waiting": len(e.waiting),
                    "standin:kv_cache_usage_perc": round(e.used_blocks() / max(1, e.n_blocks), 4),
                    "standin:num_preemptions_total": e.preemptions}

    def stop(self):
        self.stop_flag = True
        self.wake.set()
        self.thread.join(timeout=2)


def make_handler(app: StandIn):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok", "scale": app.scale, "deployment": app.engine.cfg.__dict__})
            if self.path == "/metrics":
                text = "".join(f"{k} {v}\n" for k, v in app.metrics().items()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                return self.wfile.write(text)
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/completions":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            prompt_tokens = len(str(req.get("prompt", "")).split())
            q = app.submit(prompt_tokens, int(req.get("max_tokens", 16)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    kind, k = q.get()
                    if kind == "done":
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        break
                    chunk = {"object": "text_completion", "choices": [{"index": 0, "text": "tok " * k}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

    return H


def serve(deployment="baseline", port=8401, scale=1.0, seed=0, cfg=None):
    """Start the stand-in in a background thread. Returns (server, app); call stop_server(server, app)."""
    cfg = cfg or S.Config.load(deployment)
    app = StandIn(cfg, scale=scale, seed=seed)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, app


def stop_server(httpd, app):
    httpd.shutdown()
    httpd.server_close()
    app.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8401)
    ap.add_argument("--deployment", default="baseline")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    httpd, app = serve(a.deployment, a.port, a.scale)
    print(f"stand-in engine on http://127.0.0.1:{a.port} (deployment {a.deployment}, scale {a.scale}); Ctrl+C stops it")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_server(httpd, app)
