#!/usr/bin/env python3
"""Load a deployed endpoint and report what it holds up to.

    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY"
    python3 scripts/benchmark.py https://<endpoint> --key "$API_KEY" --concurrency 64,128,256,512 \
        --input-tokens 1000 --output-tokens 190 --seconds 120

One row per concurrency level: requests/sec, input and output tokens/sec, latency p50/p95/p99, and the
decode speed a single request saw. Size a fleet on the aggregate numbers; check your latency budget on
the percentiles. Both come from the same run, which is the point: peak throughput from one level and a
passing p95 from a lower one overstated capacity by 21% once.

Prompts are unique by default (a UUID up front defeats prefix caching) because that is the shape a fleet
has to be sized for; --shared-prefix measures the cached case instead. Load is spread across processes,
not threads: one Python process driving 768 connections measured a third of the true throughput.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
import uuid

import requests

FILLER = ("The customer placed an order containing several items and asked about delivery timing, "
          "refunds, and the status of a previous return. ")


def prompt(n_tokens: int, shared: bool) -> str:
    body = FILLER * max(1, n_tokens // 25)
    # Unique prompts lead with the nonce. A trailing nonce would leave every earlier cache block
    # identical and still cacheable.
    return (body if shared else f"Request {uuid.uuid4()}. {body}")[: n_tokens * 5]


def worker(url: str, key: str, model: str, conc: int, in_tok: int, out_tok: int, seconds: float,
           shared: bool, q: mp.Queue) -> None:
    import signal
    import threading

    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent handles Ctrl-C and terminates us

    stop = threading.Event()
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "in": 0, "out": 0, "lat": []}
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    def one() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                r = sess.post(f"{url}/v1/responses",
                              json={"model": model, "input": prompt(in_tok, shared),
                                    "max_output_tokens": out_tok, "temperature": 0.0},
                              timeout=300)
                dt = time.perf_counter() - t0
                if r.status_code == 200:
                    u = r.json().get("usage") or {}
                    with lock:
                        stats["ok"] += 1
                        stats["in"] += int(u.get("input_tokens") or 0)
                        stats["out"] += int(u.get("output_tokens") or 0)
                        stats["lat"].append(dt)
                else:
                    with lock:
                        stats["fail"] += 1
            # A body that is not the promised shape is a failure too. Left uncaught, it killed the
            # thread silently and the row reported a plausible number at lower concurrency.
            except (requests.RequestException, ValueError, AttributeError, TypeError):
                with lock:
                    stats["fail"] += 1

    threads = [threading.Thread(target=one, daemon=True) for _ in range(conc)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    with lock:
        q.put({"wall": time.perf_counter() - t_start, **stats})


def run_level(a: argparse.Namespace, model: str, total: int) -> dict:
    procs_n = min(a.processes, total)
    per_proc = [total // procs_n + (1 if i < total % procs_n else 0) for i in range(procs_n)]
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=worker, args=(a.url, a.key, model, n, a.input_tokens, a.output_tokens,
                                             a.seconds, a.shared_prefix, q)) for n in per_proc if n]
    for p in procs:
        p.start()
    # A worker killed by the OS (OOM) would leave a bare q.get() waiting forever; the level plus the
    # per-request timeout bounds how long a healthy worker can take.
    results = []
    for _ in procs:
        try:
            results.append(q.get(timeout=a.seconds + 330))
        except Exception:                                   # noqa: BLE001 - queue.Empty
            print("  a load process did not report; its share is missing from this level",
                  file=sys.stderr)
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    if not results:
        sys.exit("no load process reported; is the endpoint reachable?")

    wall = max(r["wall"] for r in results)
    ok = sum(r["ok"] for r in results)
    fail = sum(r["fail"] for r in results)
    lat = sorted(x for r in results for x in r["lat"])
    pct = lambda p: lat[min(int(len(lat) * p), len(lat) - 1)] if lat else 0.0  # noqa: E731
    out_tok = sum(r["out"] for r in results)
    return {
        "concurrency": total, "rps": ok / wall, "input_tok_s": sum(r["in"] for r in results) / wall,
        "output_tok_s": out_tok / wall, "p50": statistics.median(lat) if lat else 0.0,
        "p95": pct(0.95), "p99": pct(0.99), "failed": fail,
        # Decode speed one request saw: its output tokens over its own wall time, averaged.
        "decode_tok_s_per_request": (out_tok / ok) / statistics.mean(lat) if ok and lat else 0.0,
    }


def served_model(url: str, key: str) -> str:
    r = None
    try:
        r = requests.get(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        body = r.json() if r.ok else {}
        data = body.get("data") if isinstance(body, dict) else None
        models = [m["id"] for m in data or [] if isinstance(m, dict) and m.get("id")]
    except (requests.RequestException, ValueError):
        models = []
    if not models:
        status = r.status_code if r is not None else "unreachable"
        body = r.text[:200] if r is not None else ""
        sys.exit(f"could not read /v1/models ({status}); check the endpoint and the key. Response: {body!r}")
    return models[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="the Endpoint stack output")
    ap.add_argument("--key", required=True, help="the ApiKeyValue stack output")
    ap.add_argument("--concurrency", default="64,128,256",
                    help="comma-separated levels to sweep; keep going until throughput stops rising")
    ap.add_argument("--input-tokens", type=int, default=1000)
    ap.add_argument("--output-tokens", type=int, default=190)
    ap.add_argument("--seconds", type=float, default=90, help="per level, after warm-up")
    ap.add_argument("--warmup-seconds", type=float, default=20)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--shared-prefix", action="store_true",
                    help="identical prompts, so the prefix cache hits; default is unique prompts")
    ap.add_argument("--json", action="store_true", help="also print one JSON line per level")
    a = ap.parse_args()
    a.url = a.url.rstrip("/")

    try:
        levels = [int(x) for x in a.concurrency.split(",") if x.strip()]
    except ValueError:
        sys.exit(f"--concurrency must be comma-separated integers (got {a.concurrency!r})")
    if (not levels or min(levels) < 1 or a.processes < 1 or a.seconds <= 0 or a.warmup_seconds < 0
            or a.input_tokens < 1 or a.output_tokens < 1):
        sys.exit("--concurrency needs one or more levels of at least 1, --processes at least 1, "
                 "--seconds above 0, --warmup-seconds at least 0, token counts at least 1")
    model = served_model(a.url, a.key)
    print(f"model {model}\n{a.input_tokens} input / {a.output_tokens} output tokens, "
          f"{'shared-prefix' if a.shared_prefix else 'unique'} prompts, {a.seconds:g}s per level "
          f"after {a.warmup_seconds:g}s warm-up, {a.processes} client processes\n")

    # Warm-up: the first requests pay for CUDA graph capture and kernel autotuning. Measuring from the
    # instant a target reports healthy read 21% low here.
    saved = a.seconds
    a.seconds = a.warmup_seconds
    run_level(a, model, levels[0])
    a.seconds = saved

    print(f"{'conc':>6} {'req/s':>7} {'in tok/s':>9} {'out tok/s':>10} {'p50':>7} {'p95':>7} "
          f"{'p99':>7} {'tok/s/req':>10} {'failed':>7}")
    prev = None
    for level in levels:
        r = run_level(a, model, level)
        print(f"{level:>6} {r['rps']:>7.1f} {r['input_tok_s']:>9,.0f} {r['output_tok_s']:>10,.0f} "
              f"{r['p50']:>6.2f}s {r['p95']:>6.2f}s {r['p99']:>6.2f}s "
              f"{r['decode_tok_s_per_request']:>10.1f} {r['failed']:>7}")
        if a.json:
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))
        if prev and r["input_tok_s"] < 0.85 * prev["input_tok_s"]:
            print("        throughput fell by more than 15% with more concurrency. A saturated server "
                  "plateaus; a fall usually means the client is the bottleneck. Add --processes.",
                  file=sys.stderr)
        prev = r

    print("\nSize on the aggregate columns at the highest level whose p95 (or p99) is inside your budget."
          "\nUnique-prompt numbers; with prefix cache hits the same hardware goes roughly 2x further.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Without this, Ctrl-C printed one traceback per worker and waited for the level to finish.
        for p in mp.active_children():
            p.terminate()
        print("\ninterrupted.", file=sys.stderr)
        sys.exit(130)
