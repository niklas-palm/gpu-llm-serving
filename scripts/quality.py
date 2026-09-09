#!/usr/bin/env python3
"""Score the deployed model on standard tasks through the endpoint, so a cheaper precision can be
checked for what it costs in answers, not only what it saves in GPUs.

    pip install "lm-eval[api]"
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY"
    python3 scripts/quality.py https://<endpoint> --key "$API_KEY" --tasks gsm8k,ifeval --limit 500

Runs lm-evaluation-harness against the OpenAI-compatible Chat Completions API of the deployment, so
what is scored is the served configuration: weights, KV cache precision, kernels and all. Prints one
row per task and, for tasks with per-sample logs, accuracy by prompt-length quartile: a small aggregate
drop can hide a consistent loss on one kind of input, and the quartiles show it.

Defaults: gsm8k 5-shot (arithmetic reasoning, exact match) and ifeval 0-shot (instruction following),
greedy decoding, 500 gsm8k questions, chat template applied. At 500 questions the interval is about
+-2 points, enough to see a precision that answers worse, not enough to certify one that does not.
Run the same command against two deployments and compare rows; do not compare with published numbers,
which use other prompts and settings.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

import requests

TASK_METRICS = {
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none", "prompt_level_loose_acc,none"],
}


def served_model(url: str, key: str) -> str:
    r = requests.get(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def run(url: str, key: str, model: str, tasks: list[str], limit: int, concurrency: int, out: str) -> None:
    cmd = ["lm_eval", "--model", "local-chat-completions",
           "--model_args", f"model={model},base_url={url}/v1/chat/completions,num_concurrent={concurrency},"
                           f"max_retries=3,tokenized_requests=False",
           "--tasks", ",".join(tasks), "--apply_chat_template", "--log_samples", "--output_path", out,
           "--gen_kwargs", "temperature=0,max_gen_toks=1024", "--seed", "1234"]
    if limit:
        cmd += ["--limit", str(limit)]
    env = {**os.environ, "OPENAI_API_KEY": key}
    print("running:", " ".join(cmd[:6]), "...", flush=True)
    subprocess.run(cmd, check=True, env=env)


def summarise(out: str) -> None:
    results = glob.glob(f"{out}/**/results_*.json", recursive=True)
    if not results:
        sys.exit("lm_eval wrote no results file")
    res = json.load(open(sorted(results)[-1]))["results"]
    print(f"\n{'task':8} {'metric':32} {'score':>7} {'+-':>6}")
    for task, metrics in res.items():
        for m in TASK_METRICS.get(task, [k for k in metrics if not k.endswith("_stderr") and k != "alias"]):
            if m in metrics:
                err = metrics.get(m.replace(",", "_stderr,", 1), 0.0)
                err = err if isinstance(err, (int, float)) else 0.0   # the harness writes "N/A" for some
                print(f"{task:8} {m:32} {100 * metrics[m]:>6.1f}% {100 * err:>5.1f}")
    for task in res:
        samples = glob.glob(f"{out}/**/samples_{task}_*.jsonl", recursive=True)
        if samples and task == "gsm8k":
            buckets(sorted(samples)[-1], task)


def buckets(path: str, task: str) -> None:
    """Exact match by prompt-length quartile: is the loss concentrated on long inputs?"""
    rows = []
    for line in open(path):
        d = json.loads(line)
        if d.get("filter") not in (None, "strict-match"):
            continue                       # one row per filter; score the strict one
        score = d.get("exact_match,strict-match", d.get("exact_match"))
        if score is None:
            return
        args = d.get("arguments")
        if isinstance(args, dict):         # {"gen_args_0": {"arg_0": prompt, ...}}
            first = next(iter(args.values()), {})
            text = first.get("arg_0", "") if isinstance(first, dict) else ""
        else:                              # [[prompt, gen_kwargs], ...]
            text = args[0][0] if args and args[0] else ""
        rows.append((len(str(text)), float(score)))
    if len(rows) < 8:
        return
    rows.sort()
    q = len(rows) // 4
    print(f"\n{task} exact match by prompt length quartile (n={len(rows)}):")
    for i in range(4):
        part = rows[i * q:(i + 1) * q if i < 3 else len(rows)]
        print(f"  {part[0][0]:>6} to {part[-1][0]:>6} chars   {100 * statistics.mean(s for _, s in part):5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="the Endpoint stack output")
    ap.add_argument("--key", required=True, help="the ApiKeyValue stack output")
    ap.add_argument("--tasks", default="gsm8k,ifeval", help="comma-separated lm-eval tasks")
    ap.add_argument("--limit", type=int, default=500, help="questions per task; 0 = all")
    ap.add_argument("--concurrency", type=int, default=16, help="requests in flight")
    ap.add_argument("--output", default="", help="directory for the harness output; default a temp dir")
    a = ap.parse_args()
    if shutil.which("lm_eval") is None:
        sys.exit('lm_eval not found: pip install "lm-eval[api]"')
    a.url = a.url.rstrip("/")
    out = a.output or tempfile.mkdtemp(prefix="quality-")
    model = served_model(a.url, a.key)
    print(f"model {model}, tasks {a.tasks}, limit {a.limit or 'all'}, output {out}")
    run(a.url, a.key, model, [t.strip() for t in a.tasks.split(",") if t.strip()], a.limit, a.concurrency, out)
    summarise(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
