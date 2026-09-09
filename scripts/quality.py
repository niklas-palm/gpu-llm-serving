#!/usr/bin/env python3
"""Score the served model on standard benchmarks through the endpoint, so a cheaper precision can be
checked for what it costs in answers, not only what it saves in GPUs.

    pip install "lm-eval[api]==0.4.13" transformers langdetect immutabledict nltk
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY"                      # the standard suite
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY" --suite quick         # 10 minutes
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY" --csv results.csv --tag fp8
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY" --compare <dir of the bf16 run>

Runs lm-evaluation-harness against the deployment's OpenAI-compatible API, so what is scored is the
served configuration: weights, KV cache precision, kernels and all. Two kinds of task, two API paths:

- log-likelihood tasks (multiple choice scored by the probability of each answer, and perplexity) go
  through /v1/completions with echo and logprobs; they need the model's tokenizer, fetched from the
  Hub by model id, and measure the model's raw predictions with no generation involved;
- generative tasks (the model writes an answer that is checked) go through /v1/chat/completions with
  the chat template, greedy, a 1,024-token cap.

The standard suite is the set quantised checkpoints are usually published with: arc_challenge (25-shot),
hellaswag (10-shot), mmlu (5-shot), truthfulqa_mc2, winogrande (5-shot), gsm8k (5-shot), wikitext
perplexity, plus ifeval (instruction following), humaneval (code, executed) and gpqa_diamond (hard
reasoning, chain of thought). Limits keep it to about 40 minutes on one engine; the interval at these
sizes is 1 to 2 points, enough to see a precision that answers worse, not enough to certify one that
does not. Run the same command against two deployments and compare rows, or pass --compare to get the
fraction of questions whose answer changed: a precision can keep the aggregate and still flip one
answer in ten.

A thinking model must be served with thinking off for these settings (see docs/tuning.md), or the
chain of thought eats the cap and every generative score collapses. Do not compare with published
numbers, which use other prompts and settings; compare deployments with each other.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

# (kind, tasks, extra harness arguments, limit). Log-likelihood passes carry their own few-shot counts.
SUITES = {
    "standard": [
        ("loglik", "arc_challenge", ["--num_fewshot", "25"], 500),
        ("loglik", "hellaswag", ["--num_fewshot", "10"], 500),
        ("loglik", "winogrande", ["--num_fewshot", "5"], 500),
        ("loglik", "truthfulqa_mc2", [], 500),
        ("loglik", "wikitext", [], 60),
        ("loglik", "mmlu", ["--num_fewshot", "5"], 50),       # per subject: 57 x 50
        ("gen", "gsm8k", ["--num_fewshot", "5"], 500),
        ("gen", "ifeval,humaneval_instruct,gpqa_diamond_cot_zeroshot", ["--confirm_run_unsafe_code"], 600),
    ],
    "quick": [
        ("loglik", "arc_challenge,winogrande,wikitext", [], 200),
        ("gen", "gsm8k", ["--num_fewshot", "5"], 200),
    ],
}
METRICS = {   # the one number to report per task, and the key of its per-sample score for --compare
    "arc_challenge": "acc_norm,none", "hellaswag": "acc_norm,none", "winogrande": "acc,none",
    "truthfulqa_mc2": "acc,none", "mmlu": "acc,none", "wikitext": "word_perplexity,none",
    "gsm8k": "exact_match,strict-match", "ifeval": "prompt_level_strict_acc,none",
    "humaneval_instruct": "pass@1,create_test", "gpqa_diamond_cot_zeroshot": "exact_match,flexible-extract",
}
SAMPLE_SCORE = {"gsm8k": "exact_match", "mmlu": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm"}


def served_model(url: str, key: str) -> str:
    r = requests.get(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def harness(kind: str, tasks: str, extra: list[str], limit: int, a: argparse.Namespace, model: str, out: str) -> None:
    if kind == "loglik":
        model_args = (f"model={model},base_url={a.url}/v1/completions,num_concurrent={a.concurrency},"
                      f"max_retries=3,tokenized_requests=True,tokenizer={a.tokenizer or model},max_length=8192")
        cmd = ["lm_eval", "--model", "local-completions", "--model_args", model_args]
    else:
        model_args = (f"model={model},base_url={a.url}/v1/chat/completions,num_concurrent={a.concurrency},"
                      f"max_retries=3,tokenized_requests=False")
        cmd = ["lm_eval", "--model", "local-chat-completions", "--model_args", model_args,
               "--apply_chat_template", "--gen_kwargs", "temperature=0,max_gen_toks=1024"]
    cmd += ["--tasks", tasks, "--limit", str(limit), "--log_samples", "--output_path", f"{out}/{kind}-{tasks.split(',')[0]}",
            "--seed", "1234"] + extra
    print(f"\n[{time.strftime('%H:%M:%S')}] {kind}: {tasks} (limit {limit})", flush=True)
    r = subprocess.run(cmd, env={**os.environ, "OPENAI_API_KEY": a.key, "HF_ALLOW_CODE_EVAL": "1"})
    if r.returncode:
        print(f"  harness failed for {tasks} (exit {r.returncode}); continuing", file=sys.stderr)


def results(out: str) -> dict:
    """Merge every results file the harness wrote under `out` into one task -> metrics dict."""
    merged: dict = {}
    for path in sorted(glob.glob(f"{out}/**/results_*.json", recursive=True)):
        merged.update(json.load(open(path)).get("results", {}))
    return merged


def summarise(out: str, tag: str, model: str, csv_path: str = "") -> list[dict]:
    res = results(out)
    if not res:
        sys.exit("the harness wrote no results")
    rows = []
    print(f"\n{'task':28} {'metric':30} {'score':>8} {'+-':>6} {'n':>6}")
    for task, m in res.items():
        key = METRICS.get(task)
        if key not in m:
            continue
        err = m.get(key.replace(",", "_stderr,", 1), 0.0)
        err = err if isinstance(err, (int, float)) else 0.0
        n = sample_count(out, task)
        score = m[key]
        shown = f"{score:8.2f}" if task == "wikitext" else f"{100 * score:7.1f}%"
        print(f"{task:28} {key:30} {shown:>8} {100 * err:>5.1f} {n:>6}")
        rows.append({"tag": tag, "model": model, "task": task, "metric": key, "score": round(score, 4),
                     "stderr": round(err, 4), "n": n})
    if csv_path:
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {csv_path}")
    return rows


def samples(out: str, task: str) -> dict:
    """doc_id -> per-sample score for one task, from the harness's sample logs."""
    key = SAMPLE_SCORE.get(task)
    scores: dict = {}
    for path in glob.glob(f"{out}/**/samples_{task}_*.jsonl", recursive=True):
        for line in open(path):
            d = json.loads(line)
            if key and d.get("filter", "strict-match") in ("strict-match", "none") and key in d:
                scores[d["doc_id"]] = float(d[key])
    return scores


def sample_count(out: str, task: str) -> int:
    n = len(samples(out, task))
    if n:
        return n
    return sum(1 for p in glob.glob(f"{out}/**/samples_{task}_*.jsonl", recursive=True) for _ in open(p))


def compare(out: str, other: str) -> None:
    """How many answers changed between two runs, question by question. The aggregate can stay while a
    tenth of the answers flip; that is what a cheaper precision usually does first."""
    print(f"\n{'task':14} {'shared':>7} {'flipped':>8} {'this right, other wrong':>24} {'other right, this wrong':>24}")
    for task in SAMPLE_SCORE:
        a, b = samples(out, task), samples(other, task)
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        gained = sum(1 for i in shared if a[i] > b[i])
        lost = sum(1 for i in shared if a[i] < b[i])
        print(f"{task:14} {len(shared):>7} {100 * (gained + lost) / len(shared):>7.1f}% {gained:>24} {lost:>24}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="the Endpoint stack output")
    ap.add_argument("--key", required=True, help="the ApiKeyValue stack output")
    ap.add_argument("--suite", default="standard", choices=sorted(SUITES), help="which task set")
    ap.add_argument("--tokenizer", default="", help="tokenizer id for log-likelihood tasks; default the served model id")
    ap.add_argument("--concurrency", type=int, default=16, help="requests in flight")
    ap.add_argument("--output", default="", help="directory for the harness output; default a temp dir")
    ap.add_argument("--tag", default="", help="label written to --csv rows")
    ap.add_argument("--csv", default="", help="append one row per task here")
    ap.add_argument("--compare", default="", help="output directory of another run: report flipped answers")
    a = ap.parse_args()
    if shutil.which("lm_eval") is None:
        sys.exit('lm_eval not found: pip install "lm-eval[api]==0.4.13" transformers')
    a.url = a.url.rstrip("/")
    out = a.output or tempfile.mkdtemp(prefix="quality-")
    model = served_model(a.url, a.key)
    print(f"model {model}, suite {a.suite}, output {out}")
    for kind, tasks, extra, limit in SUITES[a.suite]:
        harness(kind, tasks, extra, limit, a, model, out)
    summarise(out, a.tag or model, model, a.csv)
    if a.compare:
        compare(out, a.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
