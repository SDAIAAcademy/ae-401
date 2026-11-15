"""servelab: the helper module of the AE-401 labs (Liwan National Events Authority, assistant Musamir).

What is here
- Data: model cards, accelerator sheets, the day's trace, tenants, evaluation outcomes, reference results.
- The cost model: operations and bytes per token, key-value cache bytes, memory left for the cache.
- The course simulator of a serving engine (`Engine`, `Cluster`, `run_closed`, `run_open`): continuous
  batching with a sequence cap and a token budget, chunked or whole prefill, a paged key-value cache with
  prefix hashing, per-tenant salts and LRU eviction, preemption by recompute, weight and cache precision,
  speculative decoding, disaggregated prefill and decode pools and several replicas behind a router.
  The configuration keys follow the names of vLLM's engine arguments. The simulator stands in for an
  engine on a processor; it is calibrated from the model cards and spec sheets and is not a measurement
  of real hardware.
- Measurement: goodput against the objectives, percentiles, intervals, plots.

Quick mode: SERVE_QUICK=1 shrinks every run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path

try:  # a Windows console may not print the check marks otherwise
    __import__("sys").stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
QUICK = os.environ.get("SERVE_QUICK", "0") == "1"
ARTIFACTS = LABS / "artifacts"


def pick(full, quick):
    """The full value, or the small one in quick mode."""
    return quick if QUICK else full


def check(cond, msg):
    """A self-check: prints a tick, or raises with the message."""
    if not cond:
        print("❌", msg)
        raise AssertionError(msg)
    print("✅", msg)


_JSON = {}


def load_json(name):
    """A data file, read once and kept (do not change what it returns)."""
    if name not in _JSON:
        _JSON[name] = json.loads((DATA / name).read_text(encoding="utf-8"))
    return _JSON[name]


def reference(name):
    """A supplied reference result (labs/data/reference/<name>.json), for the safety nets."""
    return json.loads((DATA / "reference" / f"{name}.json").read_text(encoding="utf-8"))


def artifact_dir(lab):
    d = ARTIFACTS / lab
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- cards

def models():
    return load_json("model_cards.json")


def accelerators():
    return load_json("accelerators.json")


def objectives():
    return load_json("objectives.json")


BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.53}


def flops_per_token(m):
    """Matrix-multiply operations of one token through the model (2 per weight), without attention."""
    return 2 * (m["params_nonembedding"] + m["hidden_size"] * m["vocab_size"])


def attn_flops_per_token_per_ctx(m):
    """Attention operations per token for each token of context it attends to (scores and weighted sum)."""
    return 4 * m["num_hidden_layers"] * m["num_attention_heads"] * m["head_dim"]


def weight_bytes(m, dtype="bf16"):
    return m["params"] * BYTES[dtype]


def kv_bytes_per_token(m, dtype="bf16"):
    """Keys and values of one token in every layer: 2 x layers x KV heads x head size x bytes."""
    return 2 * m["num_hidden_layers"] * m["num_key_value_heads"] * m["head_dim"] * BYTES[dtype]


# ---------------------------------------------------------------- the engine configuration

@dataclass
class Config:
    model: str = "qwen3-8b"
    accelerator: str = "L4"
    weight_dtype: str = "bf16"
    kv_cache_dtype: str = "bf16"
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    cache_salt: str = "none"          # "none" or "tenant"
    block_size: int = 16
    speculative_model: str | None = None
    num_speculative_tokens: int = 0
    role: str = "both"                # "both", "prefill" or "decode"
    layout: str = "stable_first"      # prompt layout: "deployed", "stable_first" or a name in LAYOUTS
    scheduling_policy: str = "fcfs"   # "fcfs" or "chat_first" (chat requests go ahead of documents)

    @staticmethod
    def load(name):
        return Config(**load_json(f"deployments/{name}.json")["config"])

    def but(self, **kw):
        return replace(self, **kw)


# calibration of the simulator (documented in labs/README.md)
CAL = {
    "mem_eff": 0.80,          # share of peak memory bandwidth reached
    "compute_eff": 0.45,      # share of dense peak operations reached
    "reserve_gb": 1.6,        # activations, graphs and runtime outside the cache
    "step_overhead_s": 0.004, # scheduling, sampling, launch per step
    "per_seq_s": 0.00005,     # per sequence in the batch
    "int4_dequant": 1.15,     # extra compute of 4-bit weights unpacked to 16 bits
    "jitter": 0.03,           # run-to-run noise on step time (log-normal sigma)
    "link_gbs": 20.0,         # effective cache transfer between two accelerators of the node
    "admission": "chunk",     # admit on the blocks of the next chunk, as continuous-batching engines do
}


def peak_ops(acc, weight_dtype):
    """Dense peak operations per second for the format the matrix multiplies run in."""
    if weight_dtype in ("fp8",) and acc.get("fp8_dense_tflops"):
        return acc["fp8_dense_tflops"] * 1e12
    if weight_dtype == "int8" and acc.get("int8_dense_tops"):
        return acc["int8_dense_tops"] * 1e12
    return acc["fp16_dense_tflops"] * 1e12


def cache_blocks(cfg: Config):
    """Blocks the engine gets for the key-value cache, as it would report them at start."""
    m = models()[cfg.model]
    acc = accelerators()[cfg.accelerator]
    budget = acc["memory_gb"] * 1e9 * cfg.gpu_memory_utilization
    used = weight_bytes(m, cfg.weight_dtype) + CAL["reserve_gb"] * 1e9
    if cfg.speculative_model:
        used += weight_bytes(models()[cfg.speculative_model], "bf16")
    free = budget - used
    per_block = kv_bytes_per_token(m, cfg.kv_cache_dtype) * cfg.block_size
    if cfg.speculative_model:
        per_block += kv_bytes_per_token(models()[cfg.speculative_model], cfg.kv_cache_dtype) * cfg.block_size  # the cache format is engine-wide
    return max(0, int(free // per_block))


def step_time(cfg: Config, prefill_tokens, prefill_attn, decode_ctx, n_decode, spec_k=0, rng=None):
    """Seconds of one engine step: the larger of compute time and memory time, plus overheads."""
    m = models()[cfg.model]
    acc = accelerators()[cfg.accelerator]
    fpt = flops_per_token(m)
    att = attn_flops_per_token_per_ctx(m)
    verify = spec_k + 1 if spec_k else 1
    tokens = prefill_tokens + n_decode * verify
    flops = fpt * tokens + att * (prefill_attn + decode_ctx * verify)
    eff = CAL["compute_eff"]
    ops = peak_ops(acc, cfg.weight_dtype) * eff
    if cfg.weight_dtype == "int4":
        flops *= CAL["int4_dequant"]
    t_comp = flops / ops
    kvb = kv_bytes_per_token(m, cfg.kv_cache_dtype)
    bytes_ = weight_bytes(m, cfg.weight_dtype) + kvb * (decode_ctx + prefill_attn / max(1, prefill_tokens) * 0)
    bw = acc["bandwidth_gbs"] * 1e9 * CAL["mem_eff"]
    t_mem = bytes_ / bw
    t = max(t_comp, t_mem) + CAL["step_overhead_s"] + CAL["per_seq_s"] * (n_decode + (1 if prefill_tokens else 0))
    if spec_k and n_decode:
        d = models()[cfg.speculative_model]
        t_draft = max(weight_bytes(d, "bf16") / bw, flops_per_token(d) * n_decode / ops) + CAL["step_overhead_s"] / 2
        t += spec_k * t_draft
    if rng is not None and CAL["jitter"]:
        t *= math.exp(rng.gauss(0, CAL["jitter"]))
    return t


# ---------------------------------------------------------------- requests and the prompt layout

TENANTS = None


def tenants():
    global TENANTS
    if TENANTS is None:
        TENANTS = load_json("tenants.json")
    return TENANTS


@dataclass
class Req:
    id: str
    t: float
    tenant: str
    lang: str
    kind: str
    session: str
    turn: int
    doc_id: str
    doc_tokens: int
    hist_tokens: int
    hist_parts: str
    q_tokens: int
    out_tokens: int
    difficulty: float = 0.0
    # filled by the engine
    prompt_tokens: int = 0
    keys: list = field(default_factory=list)
    prefilled: int = 0
    to_prefill: int = 0
    generated: int = 0
    held: int = 0
    shared: list = field(default_factory=list)
    hit_tokens: int = 0
    cross_hits: int = 0
    t_sched: float | None = None
    t_first: float | None = None
    t_done: float | None = None
    preemptions: int = 0
    max_gap: float = 0.0
    t_last: float | None = None
    replica: int = -1
    prefill_time: float = 0.0
    failed: bool = False
    handoff: bool = False
    need_blocks: int = 0
    transfer: float = 0.0
    spec_accepted: int = 0
    spec_proposed: int = 0


LAYOUTS = {}


def prompt_parts(r: Req):
    """The named parts of a request's prompt: {name: [(segment id, tokens), ...]}."""
    t = tenants()
    parts = {"header": [(f"hdr:{r.id}", t["header_tokens"])], "base": [("base:common", t["base_tokens"])],
             "tenant": [(f"tenant:{r.tenant}", t["tenants"][r.tenant]["tokens"])], "document": [], "history": [],
             "question": [(f"q:{r.id}", r.q_tokens)]}
    if r.doc_id:
        parts["document"] = [(f"doc:{r.doc_id}", r.doc_tokens)]
    if r.hist_parts:
        parts["history"] = [(f"turn:{r.session}:{k}", int(n)) for k, n in enumerate(r.hist_parts.split("|"), start=1)]
    return parts


def segments(r: Req, layout: str):
    """The prompt as a list of (segment id, tokens). The deployed layout starts with a per-request header."""
    if layout in LAYOUTS:
        order = LAYOUTS[layout](r) if callable(LAYOUTS[layout]) else LAYOUTS[layout]
        parts = prompt_parts(r)
        return [seg for name in order for seg in parts[name]]
    t = tenants()
    header = (f"hdr:{r.id}", t["header_tokens"])
    base = ("base:common", t["base_tokens"])
    ten = (f"tenant:{r.tenant}", t["tenants"][r.tenant]["tokens"])
    parts = []
    if r.doc_id:
        parts.append((f"doc:{r.doc_id}", r.doc_tokens))
    if r.hist_parts:
        # the history of turn n is the text of turns 1 .. n-1 of the session, in order
        for k, n in enumerate(r.hist_parts.split("|"), start=1):
            parts.append((f"turn:{r.session}:{k}", int(n)))
    q = (f"q:{r.id}", r.q_tokens)
    if layout == "deployed":
        return [header, base, ten] + parts + [q]
    return [base, ten] + parts + [header, q]


def block_keys(segs, block, salt):
    """One hash per full block of the prompt; a block's hash covers everything before it."""
    keys = []
    h = hashlib.blake2b(salt.encode(), digest_size=8)
    pos, filled = 0, 0
    for sid, n in segs:
        off = 0
        while off < n:
            take = min(n - off, block - filled)
            h.update(f"{sid}:{off}:{take};".encode())
            off += take
            filled += take
            pos += take
            if filled == block:
                keys.append(h.hexdigest())
                h = hashlib.blake2b(h.digest(), digest_size=8)
                filled = 0
    return keys


def read_trace(name="trace_day.csv", start=None, end=None, limit=None):
    rows = []
    with open(DATA / name, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            t = float(d["t"])
            if start is not None and t < start:
                continue
            if end is not None and t >= end:
                break
            rows.append(Req(id=d["id"], t=t, tenant=d["tenant"], lang=d["lang"], kind=d["kind"],
                            session=d["session"], turn=int(d["turn"]), doc_id=d["doc_id"],
                            doc_tokens=int(d["doc_tokens"]), hist_tokens=int(d["hist_tokens"]), hist_parts=d["hist_parts"],
                            q_tokens=int(d["q_tokens"]), out_tokens=int(d["out_tokens"]),
                            difficulty=float(d["difficulty"])))
            if limit and len(rows) >= limit:
                break
    return rows


def peak_window():
    """Start and end, in seconds from the start of the day, of the evening peak used by most labs."""
    p = load_json("traffic_profile.json")
    return p["peak_start_s"], p["peak_end_s"]


def peak_requests(n=None, seed=0, kinds=None):
    """A sample of requests from the evening peak, as a pool for closed-loop runs."""
    s, e = peak_window()
    rows = read_trace(start=s, end=e)
    if kinds:
        rows = [r for r in rows if r.kind in kinds]
    rng = random.Random(seed)
    if n and n < len(rows):
        rows = rng.sample(rows, n)
    return rows


def fresh(r: Req) -> Req:
    """A copy of a request with the engine's fields cleared."""
    return Req(**{k: getattr(r, k) for k in ("id", "t", "tenant", "lang", "kind", "session", "turn", "doc_id",
                                             "doc_tokens", "hist_tokens", "hist_parts", "q_tokens", "out_tokens", "difficulty")})


# ---------------------------------------------------------------- the engine

class Engine:
    """One replica: a scheduler, a paged key-value cache and a step-time model."""

    def __init__(self, cfg: Config, seed=0, name="r0"):
        self.cfg = cfg
        self.name = name
        self.rng = random.Random(seed)
        self.n_blocks = cache_blocks(cfg)
        self.empty = self.n_blocks
        self.cached_free = OrderedDict()   # key -> owner tenant, evictable
        self.in_use = {}                   # key -> [refcount, owner tenant]
        self.waiting = deque()
        self.running = []
        self.clock = 0.0
        self.preemptions = 0
        self.steps = 0
        self.log = []
        self.lookups = 0
        self.hits = 0
        self.prefill_tokens_done = 0
        self.decode_tokens_done = 0
        self.busy_time = 0.0
        self.acceptance = load_json("spec_acceptance.json")[cfg.speculative_model] if cfg.speculative_model else None

    # -- cache
    def free_blocks(self):
        return self.empty + len(self.cached_free)

    def used_blocks(self):
        return self.n_blocks - self.free_blocks()

    def _take(self, n):
        if n > self.free_blocks():
            return False
        from_empty = min(n, self.empty)
        self.empty -= from_empty
        for _ in range(n - from_empty):
            self.cached_free.popitem(last=False)
        return True

    def _release(self, r: Req):
        anon = r.held - len(r.shared)
        for k in r.shared:
            ent = self.in_use.get(k)
            if ent is None:
                continue
            ent[0] -= 1
            if ent[0] <= 0:
                del self.in_use[k]
                self.cached_free[k] = ent[1]
        self.empty += max(0, anon)
        r.held = 0
        r.shared = []

    def _lookup(self, r: Req):
        """Prefix-cache hits at admission: whole leading blocks already computed."""
        if not self.cfg.enable_prefix_caching:
            return 0
        hits = 0
        limit = (r.prompt_tokens - 1) // self.cfg.block_size
        for k in r.keys[:limit]:
            if k in self.in_use:
                ent = self.in_use[k]
                ent[0] += 1
            elif k in self.cached_free:
                owner = self.cached_free.pop(k)
                self.in_use[k] = [1, owner]
                ent = self.in_use[k]
            else:
                break
            if ent[1] != r.tenant:
                r.cross_hits += 1
            r.shared.append(k)
            r.held += 1
            hits += 1
        self.lookups += limit
        self.hits += hits
        return hits * self.cfg.block_size

    def _register(self, r: Req):
        if not self.cfg.enable_prefix_caching:
            return
        n_full = min(len(r.keys), r.prefilled // self.cfg.block_size)
        for k in r.keys[len(r.shared):n_full]:
            if k in self.in_use or k in self.cached_free:
                break
            self.in_use[k] = [1, r.tenant]
            r.shared.append(k)

    # -- requests
    def add(self, r: Req, now=None):
        if now is not None and not self.running and not self.waiting:
            self.clock = max(self.clock, now)
        if r.handoff:
            # handed over from a prefill pool with its cache: nothing to recompute
            r.handoff = False
            r.to_prefill = r.prefilled = r.prompt_tokens + r.generated
            r.need_blocks = math.ceil((r.to_prefill + 1) / self.cfg.block_size)
            r.held, r.shared = 0, []
            r.t_last = now
            self.waiting.append(r)
            return
        if r.prompt_tokens == 0:
            segs = segments(r, self.cfg.layout)
            r.prompt_tokens = sum(n for _, n in segs)
            salt = r.tenant if self.cfg.cache_salt == "tenant" else ""
            r.keys = block_keys(segs, self.cfg.block_size, salt)
        r.to_prefill = r.prompt_tokens + r.generated
        r.prefilled = 0
        r.replica = int(self.name[1:]) if self.name[1:].isdigit() else -1
        if self.cfg.scheduling_policy == "chat_first" and r.kind != "doc":
            for i, w in enumerate(self.waiting):
                if w.kind == "doc" and w.prefilled == 0:
                    self.waiting.insert(i, r)
                    return
        self.waiting.append(r)

    def has_work(self):
        return bool(self.running or self.waiting)

    def _preempt(self, victim: Req):
        self.running.remove(victim)
        self._release(victim)
        victim.preemptions += 1
        self.preemptions += 1
        victim.to_prefill = victim.prompt_tokens + victim.generated
        victim.prefilled = 0
        self.waiting.appendleft(victim)

    def _admit_handoffs(self):
        # the decode pool takes handed-over requests without recomputing their prompts
        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            r = self.waiting[0]
            if not self._take(r.need_blocks):
                break
            r.held = r.need_blocks
            self.waiting.popleft()
            self.running.append(r)

    def step(self):
        """One engine step. Returns the list of requests that finished in it."""
        if self.cfg.role == "decode":
            self._admit_handoffs()
        decodes, chunks = self._plan(prefill_only=self._prefill_only())
        if not decodes and not chunks and self._prefill_only():
            decodes, chunks = self._plan(prefill_only=False)
        if not decodes and not chunks:
            if self.running:
                self._preempt(self.running[-1])     # unblock: free the newest sequence's blocks
                return []
            if self.waiting:
                r = self.waiting.popleft()           # larger than the whole cache: refused
                self._release(r)
                r.failed = True
                r.t_done = self.clock
                return [r]
            return []
        return self._execute(decodes, chunks)

    def _prefill_only(self):
        # without chunked prefill, whole-prompt prefill steps run alone: every decode waits for them
        cfg = self.cfg
        return (not cfg.enable_chunked_prefill and cfg.role != "decode" and bool(self.waiting)
                and len(self.running) < cfg.max_num_seqs and self.free_blocks() > 0)

    def _plan(self, prefill_only):
        cfg = self.cfg
        bs = cfg.block_size
        spec_k = cfg.num_speculative_tokens if cfg.speculative_model else 0
        budget = cfg.max_num_batched_tokens
        decodes, chunks = [], []
        if not prefill_only:
            # 1 · decodes first
            for r in list(self.running):
                if r not in self.running or r.prefilled < r.to_prefill:
                    continue
                need_tok = (spec_k + 1) if spec_k else 1
                need = math.ceil((r.prompt_tokens + r.generated + need_tok) / bs) - r.held
                ok = True
                while need > 0 and not self._take(need):
                    victim = self.running[-1]
                    self._preempt(victim)
                    if victim in decodes:
                        decodes.remove(victim)
                        budget += need_tok
                    if victim is r:
                        ok = False
                        break
                if ok:
                    if need > 0:
                        r.held += need
                    decodes.append(r)
                    budget -= need_tok
        # 2 · prefills: chunks of running sequences, then new admissions
        if cfg.role != "decode":
            for r in self.running:
                if budget <= 0:
                    break
                if r.prefilled < r.to_prefill and r not in decodes:
                    c = min(r.to_prefill - r.prefilled, budget)
                    need = math.ceil((r.prefilled + c) / bs) - r.held
                    if need > 0 and not self._take(need):
                        continue
                    r.held += max(0, need)
                    chunks.append((r, c))
                    budget -= c
            while self.waiting and len(self.running) < cfg.max_num_seqs and budget > 0:
                r = self.waiting[0]
                if r.t_sched is None:
                    r.t_sched = self.clock
                hit_now = 0
                if r.prefilled == 0 and not r.shared:
                    hit_now = self._lookup(r)
                    r.hit_tokens += hit_now
                    r.prefilled = min(hit_now, r.to_prefill - 1)
                remaining = r.to_prefill - r.prefilled
                if not cfg.enable_chunked_prefill and remaining > budget and chunks:
                    break
                c = min(remaining, budget) if cfg.enable_chunked_prefill else remaining
                need = math.ceil((r.prefilled + c) / bs) - r.held
                # admit when this step's blocks fit above a small watermark of free blocks (or, with
                # CAL["admission"] == "prompt", only when the whole prompt fits)
                need_adm = math.ceil(r.to_prefill / bs) - r.held if CAL["admission"] == "prompt" else need
                if (self.running and need_adm + max(1, self.n_blocks // 100) > self.free_blocks()) or                         (need > 0 and not self._take(need)):
                    # not admitted: a waiting request holds no blocks
                    self._release(r)
                    r.prefilled = 0
                    r.hit_tokens -= hit_now
                    break
                r.held += max(0, need)
                self.waiting.popleft()
                self.running.append(r)
                chunks.append((r, c))
                budget -= c
                if prefill_only and not cfg.enable_chunked_prefill and budget <= 0:
                    break
        return decodes, chunks

    def _execute(self, decodes, chunks):
        cfg = self.cfg
        spec_k = cfg.num_speculative_tokens if cfg.speculative_model else 0
        # time of the step
        pf = sum(c for _, c in chunks)
        pf_attn = sum(c * (r.prefilled + c / 2) for r, c in chunks)
        ctx = sum(r.prompt_tokens + r.generated for r in decodes)
        dt = step_time(cfg, pf, pf_attn, ctx, len(decodes), spec_k, self.rng)
        t0 = self.clock
        self.clock += dt
        self.busy_time += dt
        self.steps += 1
        now = self.clock
        finished = []
        for r, c in chunks:
            r.prefilled += c
            r.prefill_time += dt
            self.prefill_tokens_done += c
            if r.prefilled >= r.to_prefill:
                self._register(r)
                if r.t_first is None:
                    r.t_first = now
                    r.t_last = now
                    r.generated = max(r.generated, 1)
                    if cfg.role == "prefill" or r.generated >= r.out_tokens:
                        finished.append(r)
        for r in decodes:
            n = 1
            if spec_k:
                a = self.acceptance[r.lang]
                acc = 0
                while acc < spec_k and self.rng.random() < a:
                    acc += 1
                n = acc + 1
                r.spec_accepted += acc
                r.spec_proposed += spec_k
            n = min(n, r.out_tokens - r.generated)
            r.generated += n
            self.decode_tokens_done += n
            if r.t_last is not None:
                r.max_gap = max(r.max_gap, now - r.t_last)
            r.t_last = now
            if r.generated >= r.out_tokens:
                finished.append(r)
        for r in finished:
            r.t_done = now
            if r in self.running:
                self.running.remove(r)
            self._release(r)
        if self.steps % 4 == 0 or finished:
            self.log.append((round(t0, 4), len(self.running), len(self.waiting),
                             round(self.used_blocks() / max(1, self.n_blocks), 4), self.preemptions, pf, len(decodes)))
        return finished


# ---------------------------------------------------------------- drivers

def _route_rr(engines, r, state):
    i = state.setdefault("i", 0)
    state["i"] = (i + 1) % len(engines)
    return engines[i]


ROUTERS = {"round_robin": _route_rr}


def simulate(cfg: Config, requests=None, *, users=None, pool=None, n_requests=None, replicas=1,
             router="round_robin", seed=0, warmup=0.1, disagg=None, horizon=None):
    """Run the simulator.

    Open loop: `requests` with arrival times `t`. Closed loop: `users` concurrent users draw requests
    from `pool` until `n_requests` were sent; each sends the next as soon as the last one finished.
    `replicas` engines behind `router` (a name in ROUTERS or a function(engines, request, state)).
    `disagg=(n_prefill, n_decode)` splits the engines into a prefill pool and a decode pool.
    Returns a Result.
    """
    rng = random.Random(seed)
    route = ROUTERS[router] if isinstance(router, str) else router
    state = {}
    if disagg:
        npf, ndc = disagg
        pf_eng = [Engine(cfg.but(role="prefill"), seed=seed * 101 + i, name=f"p{i}") for i in range(npf)]
        dc_eng = [Engine(cfg.but(role="decode"), seed=seed * 103 + i, name=f"d{i}") for i in range(ndc)]
        engines = pf_eng + dc_eng
        front = pf_eng
    else:
        engines = [Engine(cfg, seed=seed * 97 + i, name=f"r{i}") for i in range(replicas)]
        front = engines
    arrivals = []   # (time, seq, request, target pool)
    seq = 0
    if users:
        pool = list(pool)
        order = list(range(len(pool)))
        rng.shuffle(order)
        sent = 0
        for u in range(users):
            if sent >= n_requests:
                break
            base = pool[order[sent % len(order)]]
            r = fresh(base)
            r.id = f"{base.id}-{sent}"
            r.t = 0.0
            arrivals.append((0.0, seq, r, "front"))
            seq += 1
            sent += 1
    else:
        for r in requests:
            arrivals.append((r.t, seq, fresh(r), "front"))
            seq += 1
        sent = len(arrivals)
    import heapq
    heapq.heapify(arrivals)
    done = []
    link = CAL["link_gbs"] * 1e9
    kvb = kv_bytes_per_token(models()[cfg.model], cfg.kv_cache_dtype)
    while True:
        busy = [e for e in engines if e.has_work()]
        e = min(busy, key=lambda x: x.clock) if busy else None
        if arrivals and (e is None or arrivals[0][0] <= e.clock):
            t, _, r, where = heapq.heappop(arrivals)
            if horizon and t > horizon:
                break
            if where == "front":
                target = route(front, r, state)
            else:
                target = min(dc_eng, key=lambda x: len(x.running) + len(x.waiting))
            target.add(r, now=t)
            continue
        if e is None:
            break
        fin = e.step()
        for r in fin:
            if e.cfg.role == "prefill" and r.generated < r.out_tokens and not r.failed:
                r.transfer = r.prompt_tokens * kvb / link
                heapq.heappush(arrivals, (r.t_done + r.transfer, seq, r, "decode"))
                seq += 1
                r.t_done = None
                r.handoff = True
                continue
            done.append(r)
            if users and sent < n_requests:
                base = pool[order[sent % len(order)]]
                nr = fresh(base)
                nr.id = f"{base.id}-{sent}"
                nr.t = r.t_done
                heapq.heappush(arrivals, (nr.t, seq, nr, "front"))
                seq += 1
                sent += 1
    return Result(cfg, done, engines, warmup=warmup, users=users)

# ---------------------------------------------------------------- results

def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


class Result:
    def __init__(self, cfg, done, engines, warmup=0.1, users=None):
        self.cfg = cfg
        self.engines = engines
        self.users = users
        obj = objectives()
        self.ttft_slo, self.tpot_slo, self.stall_slo = obj["ttft_s"], obj["tpot_s"], obj["stall_s"]
        done = sorted(done, key=lambda r: r.t_done)
        cut = int(len(done) * warmup)
        self.all = done
        self.rows = done[cut:]
        self.t0 = min((r.t for r in self.rows), default=0.0)
        self.t1 = max((r.t_done for r in self.rows), default=0.0)

    def table(self):
        out = []
        for r in self.rows:
            if r.failed or r.t_first is None:
                ttft, tpot = float("inf"), float("inf")
            else:
                ttft = r.t_first - r.t
                tpot = (r.t_done - r.t_first) / max(1, r.out_tokens - 1) if r.out_tokens > 1 else 0.0
            out.append({"id": r.id, "tenant": r.tenant, "lang": r.lang, "kind": r.kind, "arrival": round(r.t, 3),
                        "prompt_tokens": r.prompt_tokens, "out_tokens": r.out_tokens, "hit_tokens": r.hit_tokens,
                        "cross_tenant_blocks": r.cross_hits, "queue_s": round((r.t_sched or r.t) - r.t, 4),
                        "ttft_s": round(ttft, 4), "tpot_s": round(tpot, 5), "stall_s": round(r.max_gap, 4),
                        "e2e_s": round(r.t_done - r.t, 4), "preemptions": r.preemptions, "replica": r.replica,
                        "met": bool(ttft <= self.ttft_slo and tpot <= self.tpot_slo and r.max_gap <= self.stall_slo)})
        return out

    def summary(self):
        tab = self.table()
        dur = max(1e-9, self.t1 - self.t0)
        ttft = [x["ttft_s"] for x in tab]
        tpot = [x["tpot_s"] for x in tab if x["out_tokens"] > 1]
        met = sum(x["met"] for x in tab)
        toks = sum(x["out_tokens"] for x in tab)
        e = self.engines
        lookups = sum(x.lookups for x in e)
        return {
            "requests": len(tab), "duration_s": round(dur, 2),
            "throughput_rps": round(len(tab) / dur, 4), "goodput_rps": round(met / dur, 4),
            "attainment": round(met / max(1, len(tab)), 4), "output_tok_s": round(toks / dur, 1),
            "ttft_p50": round(pct(ttft, .5), 3), "ttft_p90": round(pct(ttft, .9), 3), "ttft_p99": round(pct(ttft, .99), 3),
            "tpot_p50": round(pct(tpot, .5), 4), "tpot_p90": round(pct(tpot, .9), 4), "tpot_p99": round(pct(tpot, .99), 4),
            "stall_p90": round(pct([x["stall_s"] for x in tab], .9), 3),
            "preemptions": sum(x.preemptions for x in e),
            "hit_rate": round(sum(x.hits for x in e) / lookups, 4) if lookups else 0.0,
            "cache_blocks": e[0].n_blocks, "busy_s": round(sum(x.busy_time for x in e), 2),
        }

    def engine_log(self, i=0):
        """Sampled engine metrics: time, running, waiting, cache usage, preemptions so far, prefill tokens, decodes."""
        cols = ("t", "running", "waiting", "kv_cache_usage", "preemptions_total", "prefill_tokens", "decode_seqs")
        return [dict(zip(cols, row)) for row in self.engines[i].log]


def run_closed(cfg, users, n_requests=None, pool=None, seed=0, **kw):
    """Closed loop: `users` concurrent users at the evening peak."""
    n_requests = n_requests or pick(max(400, users * 12), max(120, users * 6))
    pool = pool if pool is not None else peak_requests(seed=0)
    return simulate(cfg, users=users, pool=pool, n_requests=n_requests, seed=seed, **kw)


def run_open(cfg, requests, seed=0, **kw):
    """Open loop: requests arrive at their own times."""
    return simulate(cfg, requests, seed=seed, **kw)


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def paired_diff(a, b, n_boot=2000, seed=0):
    """Mean of b - a over paired items, with a bootstrap 95% interval."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return round(sum(d) / n, 4), (round(means[int(.025 * n_boot)], 4), round(means[int(.975 * n_boot)], 4))


def mean_sd(xs):
    xs = list(xs)
    return (round(statistics.mean(xs), 4), round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0)


# ---------------------------------------------------------------- evaluation

def eval_items():
    with open(DATA / "eval" / "items.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eval_outcomes():
    """{variant: {item id: 0 or 1}} from the recorded outcomes (synthetic)."""
    out = {}
    with open(DATA / "eval" / "outcomes.csv", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            out.setdefault(d["variant"], {})[d["item"]] = int(d["correct"])
    return out


def floors():
    return load_json("eval/floors.json")


# ---------------------------------------------------------------- plots

def plt():
    import matplotlib
    if not os.environ.get("DISPLAY") and "ipykernel" not in __import__("sys").modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as p
    p.rcParams.update({"figure.figsize": (7, 3.6), "axes.spines.top": False, "axes.spines.right": False,
                       "axes.grid": True, "grid.alpha": .3})
    return p


def show(fig, name=None, lab=None):
    if name and lab:
        fig.savefig(artifact_dir(lab) / f"{name}.png", dpi=110, bbox_inches="tight")
    p = plt()
    if "ipykernel" in __import__("sys").modules:
        p.show()
    else:
        p.close(fig)


def save_rows(rows, lab, name):
    """Raw rows to artifacts/<lab>/<name>.csv: the raw data a claim rests on."""
    path = artifact_dir(lab) / f"{name}.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return path


# ---------------------------------------------------------------- helpers for the labs

def synthetic_requests(n, prompt_tokens, out_tokens, lang="en", tenant="qamra", start=0.0, gap=0.0):
    """n requests of a fixed length with no shared prefix (for sizing checks)."""
    return [Req(id=f"X{i:04d}", t=start + i * gap, tenant=tenant, lang=lang, kind="fixed", session=f"X{i:04d}",
                turn=1, doc_id="", doc_tokens=0, hist_tokens=0, hist_parts="",
                q_tokens=prompt_tokens, out_tokens=out_tokens) for i in range(n)]


def fixed_layout_segments(r):
    return [(f"q:{r.id}", r.q_tokens)]


_orig_segments = segments


def segments(r, layout):  # noqa: F811  fixed-length requests are one segment, no header or base
    if r.kind == "fixed":
        return fixed_layout_segments(r)
    return _orig_segments(r, layout)


def start_log(cfg: Config):
    """What the engine reports when it starts: memory taken and the cache it got (stand-in wording)."""
    m = models()[cfg.model]
    acc = accelerators()[cfg.accelerator]
    w = weight_bytes(m, cfg.weight_dtype) / 2**30
    blocks = cache_blocks(cfg)
    tokens = blocks * cfg.block_size
    lines = [f"engine: {cfg.model} {cfg.weight_dtype} on {cfg.accelerator} ({acc['memory_gb']} GB), "
             f"gpu_memory_utilization={cfg.gpu_memory_utilization}",
             f"model weights took {w:.2f} GiB; activations, graphs and runtime reserve {CAL['reserve_gb'] * 1e9 / 2**30:.2f} GiB",
             f"key-value cache: {blocks} blocks of {cfg.block_size} tokens = {tokens} tokens ({cfg.kv_cache_dtype})"]
    for L in (2000, 4000, 8000):
        lines.append(f"maximum concurrency for {L} tokens per request: {tokens / L:.2f}x")
    return "\n".join(lines)


def max_running(result, i=0):
    """The largest number of sequences the engine held at once, from its log."""
    return max((row[1] for row in result.engines[i].log), default=0)


def prompt_tokens(r, layout="deployed"):
    """Tokens of a request's prompt under a layout."""
    return sum(n for _, n in segments(r, layout))


def peak_jobs(n, seed=0, layout="deployed"):
    """(prompt tokens, answer tokens) of n requests from the evening peak, for the load generator."""
    return [(prompt_tokens(r, layout), r.out_tokens) for r in peak_requests(n=n, seed=seed)]


def engine_signature(run, i=0):
    """Mean running, waiting and cache usage over the engine log, and preemptions per 100 requests."""
    log = run.engine_log(i)
    n = max(1, len(log))
    return {"running": round(sum(x["running"] for x in log) / n, 2),
            "waiting": round(sum(x["waiting"] for x in log) / n, 2),
            "kv_cache_usage": round(sum(x["kv_cache_usage"] for x in log) / n, 3),
            "preemptions_per_100": round(100 * run.engines[i].preemptions / max(1, len(run.all)), 1),
            "cap": run.cfg.max_num_seqs}


def mixed_pool(share_docs=0.35, n=400, seed=1):
    """Peak requests with a chosen share of long-document requests (Lab 2.2)."""
    pk = peak_requests()
    docs = [r for r in pk if r.kind == "doc"]
    chat = [r for r in pk if r.kind == "chat"]
    rng = random.Random(seed)
    k = int(n * share_docs)
    return rng.sample(docs, k) + rng.sample(chat, n - k)
