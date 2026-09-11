#!/usr/bin/env python3
"""Score the served model on entity extraction into a fixed JSON structure, through the endpoint.

    pip install requests datasets
    python3 scripts/extraction.py https://<endpoint> --key "$API_KEY"                    # schema and free modes, 1,000 sentences
    python3 scripts/extraction.py https://<endpoint> --key "$API_KEY" --modes schema --csv results.csv --tag fp8
    python3 scripts/extraction.py https://<endpoint> --key "$API_KEY" --compare <output dir of the bf16 run>

The task is CoNLL-2003 named entity recognition: for each sentence, list the persons, organisations,
locations and miscellaneous names it mentions, as a JSON object with those four arrays. This is what
"extract the fields from this document" workloads do, scored against a standard gold set with the standard
entity-level precision, recall and F1 (a predicted mention counts when its text and its type both match).

Three ways to ask for the JSON, because they are different features of the engine:

- schema: `response_format` with a JSON schema and strict on. The engine compiles a grammar and constrains
  every token, so the answer always parses and always has the four keys. This is structured output.
- json: `response_format` json_object. Valid JSON is guaranteed, the keys are not.
- free: no constraint; the prompt asks for JSON only and the script parses whatever came back, leniently.

Run two modes on the same deployment and the difference is what constrained decoding costs or gains in
answers; run one mode on two deployments and the difference is the precision's cost, like scripts/quality.py.
--compare reports the fraction of sentences whose extracted set changed between two runs. Greedy decoding,
three fixed few-shot examples from the validation split, first N test sentences in dataset order, so two runs
see identical requests. About five minutes for 1,000 sentences on one engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import statistics
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

KEYS = {"PER": "persons", "ORG": "organizations", "LOC": "locations", "MISC": "misc"}
MODES = ("schema", "json", "free")
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(KEYS.values()),
          "properties": {k: {"type": "array", "items": {"type": "string"}} for k in KEYS.values()}}
SYSTEM = ("Extract the named entities from the sentence. Answer with one JSON object and nothing else, with the "
          "keys persons, organizations, locations and misc, each a list of the entity mentions exactly as they "
          "appear in the sentence (same words, same order, same spelling). misc is for nationalities, events, "
          "products, works and other names that are not a person, an organisation or a place. Use an empty list "
          "for a key with no mentions.")


def load(split: str):
    from datasets import load_dataset   # imported here so --help works without it
    return load_dataset("tomaarsen/conll2003", split=split)


def entities(tokens: list[str], tags: list[int], names: list[str]) -> dict[str, list[str]]:
    """BIO tags -> {key: [mention, ...]} in sentence order."""
    out: dict[str, list[str]] = {k: [] for k in KEYS.values()}
    cur, cur_type = [], ""
    for tok, tag in list(zip(tokens, (names[t] for t in tags))) + [("", "O")]:
        if tag.startswith("I-") and cur and tag[2:] == cur_type:
            cur.append(tok)
            continue
        if cur:
            out[KEYS[cur_type]].append(" ".join(cur))
        cur, cur_type = ([tok], tag[2:]) if tag != "O" else ([], "")
    return out


def parse(text: str) -> dict[str, list[str]] | None:
    """The first JSON object in the answer, or None. Tolerates a code fence, prose before or after, and
    a second object: a greedy first-brace-to-last-brace match failed all three."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            d, _ = dec.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(d, dict):
            out = {}
            for k in KEYS.values():
                v = d.get(k, [])
                out[k] = [re.sub(r"\s+", " ", str(x)).strip() for x in v if str(x).strip()] if isinstance(v, list) else []
            return out
    return None


def score(gold: dict, pred: dict | None) -> tuple[int, int, int]:
    """Entity-level (type, mention) matches: true positives, false positives, false negatives."""
    g = Counter((k, m) for k, ms in gold.items() for m in ms)
    p = Counter((k, m) for k, ms in (pred or {}).items() for m in ms)
    tp = sum((g & p).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def served_model(url: str, key: str) -> str:
    r = requests.get(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data") or []]
    if not models:
        sys.exit("the endpoint lists no model")
    return models[0]


def body(model: str, sentence: str, shots: list[tuple[str, dict]], mode: str, max_tokens: int = 300) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}]
    for s, e in shots:
        msgs += [{"role": "user", "content": s}, {"role": "assistant", "content": json.dumps(e)}]
    msgs.append({"role": "user", "content": sentence})
    b = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens}
    if mode == "schema":
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": "entities", "strict": True, "schema": SCHEMA}}
    elif mode == "json":
        b["response_format"] = {"type": "json_object"}
    return b


def run(a: argparse.Namespace, model: str, mode: str, docs: list, shots: list, out: str) -> dict:
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {a.key}", "Content-Type": "application/json"})

    def one(i_doc):
        i, (sentence, gold) = i_doc
        t0 = time.perf_counter()
        try:
            r = sess.post(f"{a.url}/v1/chat/completions", json=body(model, sentence, shots, mode, a.max_tokens), timeout=120)
            text = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""
            err = "" if r.status_code == 200 else f"status {r.status_code}"
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
            text, err = "", str(e)[:100]
        pred = parse(text)
        tp, fp, fn = score(gold, pred)
        return {"doc_id": i, "sentence": sentence, "gold": gold, "pred": pred, "parsed": pred is not None,
                "exact": pred == gold, "tp": tp, "fp": fp, "fn": fn, "seconds": round(time.perf_counter() - t0, 3), "error": err}

    with ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(one, enumerate(docs)))
    with open(f"{out}/samples_{mode}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"mode": mode, "n": len(rows), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "exact": round(sum(r["exact"] for r in rows) / len(rows), 4),
            "unparsed": sum(not r["parsed"] for r in rows), "errors": sum(bool(r["error"]) for r in rows),
            "seconds_p50": round(statistics.median(r["seconds"] for r in rows), 2)}


def compare(out: str, other: str) -> None:
    """Sentences whose extracted set changed between two runs of the same mode."""
    for mode in MODES:
        a, b = f"{out}/samples_{mode}.jsonl", f"{other}/samples_{mode}.jsonl"
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        ra = {json.loads(l)["doc_id"]: json.loads(l) for l in open(a)}
        rb = {json.loads(l)["doc_id"]: json.loads(l) for l in open(b)}
        shared = sorted(set(ra) & set(rb))
        if not shared:
            continue
        changed = sum(ra[i]["pred"] != rb[i]["pred"] for i in shared)
        better = sum(ra[i]["exact"] and not rb[i]["exact"] for i in shared)
        worse = sum(rb[i]["exact"] and not ra[i]["exact"] for i in shared)
        print(f"{mode:8} shared {len(shared):5}  changed {100 * changed / len(shared):5.1f}%  "
              f"this exact, other not {better:4}   other exact, this not {worse:4}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="the Endpoint stack output")
    ap.add_argument("--key", required=True, help="the ApiKeyValue stack output")
    ap.add_argument("--modes", default="schema,free", help=f"comma-separated: {', '.join(MODES)}")
    ap.add_argument("--limit", type=int, default=1000, help="sentences, in dataset order")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=300,
                    help="answer cap; a reasoning model spends this on its thinking first, so give it 2,000 or more")
    ap.add_argument("--output", default="", help="directory for per-sentence results; default a temp dir")
    ap.add_argument("--tag", default="", help="label written to --csv rows")
    ap.add_argument("--csv", default="", help="append one row per mode here")
    ap.add_argument("--compare", default="", help="output directory of another run: report changed sentences")
    a = ap.parse_args()
    a.url = a.url.rstrip("/")
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    if not modes or set(modes) - set(MODES) or a.limit < 1:
        sys.exit(f"--modes takes any of {', '.join(MODES)}; --limit at least 1")
    model = served_model(a.url, a.key)
    out = a.output or tempfile.mkdtemp(prefix="extraction-")
    os.makedirs(out, exist_ok=True)

    test, val = load("test"), load("validation")
    names = test.features["ner_tags"].feature.names
    docs = [(" ".join(d["tokens"]), entities(d["tokens"], d["ner_tags"], names))
            for d in test if len(d["tokens"]) >= 3 and d["tokens"][0] != "-DOCSTART-"][: a.limit]
    shots = [(" ".join(d["tokens"]), entities(d["tokens"], d["ner_tags"], names)) for d in
             (val[i] for i in (4, 24, 60))]   # three sentences with several entity types between them
    print(f"model {model}, {len(docs)} sentences, modes {', '.join(modes)}, output {out}")
    print(f"\n{'mode':8} {'n':>5} {'precision':>9} {'recall':>7} {'f1':>7} {'exact':>7} {'unparsed':>8} {'errors':>6} {'p50 s':>6}")
    rows = []
    for mode in modes:
        s = run(a, model, mode, docs, shots, out)
        rows.append({"tag": a.tag or model, "model": model, **s})
        print(f"{mode:8} {s['n']:>5} {100 * s['precision']:>8.1f}% {100 * s['recall']:>6.1f}% {100 * s['f1']:>6.1f}% "
              f"{100 * s['exact']:>6.1f}% {s['unparsed']:>8} {s['errors']:>6} {s['seconds_p50']:>6}")
    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
    if a.compare:
        print()
        compare(out, a.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
