#!/usr/bin/env python3
"""Verify a deployed endpoint: both API shapes, streaming, and a concurrency check.

    python3 scripts/test_endpoint.py https://llm.example.com --key "$API_KEY"
    python3 scripts/test_endpoint.py https://llm.example.com --key "$API_KEY" --concurrency 64

Exercises the Responses API and Chat Completions separately, because a deployment can serve one and
not the other depending on the engine version.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def check_health(url: str, key: str) -> bool:
    try:
        r = requests.get(f"{url}/health", headers=headers(key), timeout=20)
        print(f"  /health                 HTTP {r.status_code}")
        return r.status_code == 200
    except requests.RequestException as e:
        print(f"  /health                 unreachable: {type(e).__name__}")
        return False


def _json(r: requests.Response) -> dict:
    """Response body as a dict, or empty. A 200 is not a promise of JSON: a proxy or an error page can
    return one, and `r.json()` then raises where the caller only wanted to report what came back."""
    try:
        d = r.json()
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def check_models(url: str, key: str) -> str:
    """Returns the served model name, which every later request needs."""
    r = requests.get(f"{url}/v1/models", headers=headers(key), timeout=30)
    names = [m.get("id") for m in (_json(r).get("data") or [])] if r.ok else []
    print(f"  /v1/models              HTTP {r.status_code}  {names}")
    return names[0] if names else "model"


def check_responses(url: str, key: str, model: str) -> bool:
    """The Responses API. Newer than Chat Completions, so absent on older engine builds."""
    r = requests.post(f"{url}/v1/responses", headers=headers(key),
                      json={"model": model, "input": "Reply with exactly: ok"}, timeout=120)
    if r.status_code == 404:
        print("  /v1/responses           HTTP 404 - not available on this engine build")
        return False
    text = ""
    if r.ok:
        d = _json(r)
        # The Responses API nests output differently from Chat Completions.
        for item in d.get("output") or []:
            for c in item.get("content") or []:
                text += c.get("text") or ""
        text = text or d.get("output_text") or ""
    print(f"  /v1/responses           HTTP {r.status_code}  {text.strip()[:60]!r}")
    return r.ok


def check_chat(url: str, key: str, model: str) -> bool:
    """Chat Completions, kept as the compatibility path for existing clients."""
    r = requests.post(f"{url}/v1/chat/completions", headers=headers(key),
                      json={"model": model,
                            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                            "max_tokens": 16}, timeout=120)
    # `content` is None on reasoning builds, which put the text in `reasoning_content` instead - and
    # the shipped model family has such builds, so `.strip()` on the raw value crashed the smoke test
    # against a perfectly healthy endpoint. `choices` is also absent when a 200 carries an error body.
    text = ""
    if r.ok:
        choices = _json(r).get("choices") or [{}]
        message = choices[0].get("message") or {}
        text = message.get("content") or message.get("reasoning_content") or ""
    print(f"  /v1/chat/completions    HTTP {r.status_code}  {text.strip()[:60]!r}")
    return r.ok


def check_streaming(url: str, key: str, model: str) -> None:
    """Streaming on the Responses API, which is the primary interface here.

    Parses both event shapes, so the same check works against Chat Completions too: the Responses API
    emits `{"type": "response.output_text.delta", "delta": "..."}` while Chat Completions nests the
    text under `choices[].delta.content`.
    """
    t0 = time.perf_counter()
    first = None
    chunks = 0
    with requests.post(f"{url}/v1/responses", headers=headers(key),
                       json={"model": model, "input": "Count to twenty.",
                             "max_output_tokens": 80, "stream": True},
                       stream=True, timeout=120) as r:
        if not r.ok:
            print(f"  streaming               HTTP {r.status_code}")
            return
        for raw in r.iter_lines():
            if raw and raw.startswith(b"data: ") and raw[6:] != b"[DONE]":
                try:
                    d = json.loads(raw[6:])
                except json.JSONDecodeError:
                    continue
                piece = d.get("delta") if isinstance(d.get("delta"), str) else None
                if piece is None:
                    for ch in d.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            break
                if piece:
                    if first is None:
                        first = time.perf_counter() - t0
                    chunks += 1
    total = time.perf_counter() - t0
    # A 200 that yields no text deltas is a real outcome, not an impossible one: an engine build
    # without the Responses API streams a different event shape, and a content filter can close the
    # stream empty. Formatting `first` (still None) into a `:.2f` field raised TypeError and turned a
    # diagnosable result into a crash in the middle of the smoke test.
    if not chunks:
        print("  streaming               HTTP 200 but no text deltas - "
              "unexpected event shape or an empty response")
        return
    rate = chunks / max(total - (first or 0), 1e-6)
    print(f"  streaming               {chunks} chunks, first at {first:.2f}s, "
          f"{rate:.0f} tok/s decode")


def check_concurrency(url: str, key: str, model: str, n: int) -> None:
    """Aggregate throughput rises with concurrency while each request gets slower - both are shown,
    because tuning on one alone leads to the wrong conclusion."""
    prompt = "Summarise the benefits of horizontal scaling in two sentences."

    def one() -> float | None:
        # Exceptions are counted, not raised. `pool.map` re-raises the first one, so a single timeout
        # or reset aborted the entire check with a traceback - and under load some failures are the
        # expected outcome, which is exactly what the "n/64 ok" count is there to report.
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{url}/v1/responses", headers=headers(key),
                              json={"model": model, "input": prompt,
                                    "max_output_tokens": 64}, timeout=180)
        except requests.RequestException:
            return None
        return time.perf_counter() - t0 if r.ok else None

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        times = [t for t in pool.map(lambda _: one(), range(n)) if t]
    wall = time.perf_counter() - t0
    if not times:
        print(f"  concurrency {n:<3}         all requests failed")
        return
    print(f"  concurrency {n:<3}         {len(times)}/{n} ok, {len(times) / wall:.2f} rps, "
          f"p50 {statistics.median(times):.2f}s, max {max(times):.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="the Endpoint stack output, e.g. https://llm.your.domain")
    ap.add_argument("--key", required=True)
    # 64, not 8: a current-generation GPU is not meaningfully loaded below ~64 concurrent requests,
    # and a throughput number taken at low concurrency understates the hardware by roughly 2x.
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    url = a.url.rstrip("/")

    print(f"Testing {url}\n")
    if not check_health(url, a.key):
        print("\nEndpoint is not healthy. See docs/troubleshooting.md.")
        return 1

    model = check_models(url, a.key)
    has_responses = check_responses(url, a.key, model)
    ok_chat = check_chat(url, a.key, model)
    check_streaming(url, a.key, model)
    if a.concurrency > 1:
        check_concurrency(url, a.key, model, a.concurrency)

    print()
    if has_responses:
        print("Endpoint is serving. /v1/responses is the primary interface; "
              + ("/v1/chat/completions also works." if ok_chat
                 else "/v1/chat/completions FAILED, which is unexpected."))
        return 0
    if ok_chat:
        print("Endpoint is serving, but /v1/responses is unavailable on this engine build - "
              "use /v1/chat/completions.")
        return 0
    print("Endpoint is NOT serving. See docs/troubleshooting.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
