"""The scripts are not imported by the stack tests, so a NameError in one of them reached a user:
`local_config_path()` was called in two places and defined nowhere."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)   # endpoint_info imports build_image from the same directory


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_script_runs_help():
    """--help runs the module body and the argparse setup; neither needs credentials."""
    import subprocess
    for name in ("build_image", "endpoint_info", "test_endpoint", "benchmark", "size_fleet", "quality"):
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, f"{name}.py"), "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_region_and_image_follow_the_config_in_use(tmp_path, monkeypatch):
    """With $CONFIG pointing elsewhere, the region must come from that file and the image URI must be
    written next to it, or the deploy reads a file the build never wrote."""
    bi = _load("build_image")
    cfg = tmp_path / "other.yaml"
    cfg.write_text("region: eu-west-2\n")
    monkeypatch.setenv("CONFIG", str(cfg))
    assert bi.config_region() == "eu-west-2"
    (tmp_path / "config.local.yaml").write_text("region: eu-north-1\napiKey: keep-me-1234567890\n")
    assert bi.config_region() == "eu-north-1", "config.local.yaml wins, like in app.py"
    uri = "111122223333.dkr.ecr.eu-west-2.amazonaws.com/gpu-llm-serving:vllm-0.28.0"
    bi.write_image_uri(uri)
    assert (tmp_path / "config.local.yaml").read_text() == f"region: eu-north-1\napiKey: keep-me-1234567890\nimage: {uri}\n"


def test_codebuild_may_assume_the_role_only_from_this_account_and_project_in_any_region():
    bi = _load("build_image")
    cond = bi.trust_policy("111122223333", "aws")["Statement"][0]["Condition"]
    assert cond["StringEquals"] == {"aws:SourceAccount": "111122223333"}
    assert cond["ArnLike"] == {"aws:SourceArn": "arn:aws:codebuild:*:111122223333:project/gpu-llm-serving-build"}, \
        "any region: one account-wide role serves every region this account builds in"


def test_a_malformed_usage_body_is_one_failed_request_not_an_ok_and_a_failure(monkeypatch):
    """`ok` was incremented before the usage fields were read, so a body with a bad usage shape raised
    inside the lock and counted as both ok and failed, with no latency recorded: inflated rps, p50 0.0."""
    import multiprocessing as mp
    bench = _load("benchmark")

    class Resp:
        status_code = 200
        def __init__(self, body): self._b = body
        def json(self): return self._b

    class Session:
        def __init__(self): self.headers = {}
        def post(self, *a, **k): return Resp({"usage": {"input_tokens": [1], "output_tokens": 7}})

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)   # worker ignores SIGINT; not in pytest
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=2, in_tok=10, out_tok=5, seconds=0.05, shared=True, q=q)
    r = q.get(timeout=5)
    assert r["ok"] == 0 and r["lat"] == [] and r["fail"] > 0


def test_a_size_list_is_a_per_request_mix(monkeypatch):
    """--input-tokens 1000,8000,16000 must produce prompts of each size, not one size; a bad entry or a
    zero is a one-line exit."""
    import multiprocessing as mp
    bench = _load("benchmark")
    seen = []

    class Resp:
        status_code = 200
        def json(self): return {"usage": {"input_tokens": 1, "output_tokens": 1}}

    class Session:
        def __init__(self): self.headers = {}
        def post(self, url, json, timeout):
            seen.append((len(json["input"]) // 5, json["max_output_tokens"])); return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=2, in_tok=[100, 1000], out_tok=[50, 800], seconds=0.1, shared=True, q=q)
    q.get(timeout=5)
    assert {round(n, -2) for n, _ in seen} == {100, 1000} and {o for _, o in seen} == {50, 800}


def test_a_level_drains_its_in_flight_requests_and_counts_only_the_window(monkeypatch):
    """Exiting with requests open left the engine generating them behind CloudFront, and the next level
    started behind that backlog. The worker now waits for them and does not count the late ones."""
    import multiprocessing as mp
    import time
    bench = _load("benchmark")

    class Resp:
        status_code = 200
        def json(self): return {"usage": {"input_tokens": 1, "output_tokens": 1}}

    class Session:
        def __init__(self): self.headers = {}
        def post(self, *a, **k): time.sleep(0.4); return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue(); t0 = time.perf_counter()
    bench.worker("http://u", "k", "m", conc=3, in_tok=[10], out_tok=[5], seconds=0.5, shared=True, q=q)
    r = q.get(timeout=5); elapsed = time.perf_counter() - t0
    assert r["ok"] == 3, "one request per thread completed inside the 0.5 s window"
    assert elapsed >= 0.8, "the second request of each thread was drained, not abandoned"
    assert r["wall"] < 0.6, "the reported wall is the window, not the drain"


def test_turns_resend_the_growing_conversation_on_one_session(monkeypatch):
    """--turns N: every turn after the first carries the previous prompt plus the answer, on the same
    session (cookie jar), so a sticky load balancer can keep it on one engine."""
    import multiprocessing as mp
    bench = _load("benchmark")
    seen = []

    class Resp:
        status_code = 200
        def json(self): return {"usage": {"input_tokens": 1, "output_tokens": 1},
                                "output": [{"content": [{"text": "ANSWER"}]}]}

    class Session:
        def __init__(self): self.headers = {}
        def post(self, url, json, timeout): seen.append((id(self), json["input"])); return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=1, in_tok=[50], out_tok=[5], seconds=0.05, shared=True, q=q, turns=3)
    q.get(timeout=5)
    first = seen[:3]
    assert len({s for s, _ in first}) == 1, "one session per conversation"
    assert first[1][1].startswith(first[0][1]) and "ANSWER" in first[1][1], "turn 2 = turn 1 + answer + follow-up"
    assert first[2][1].startswith(first[1][1])


def test_streaming_measures_time_to_first_token_and_reads_usage_from_the_completed_event(monkeypatch):
    import multiprocessing as mp
    import json as js
    bench = _load("benchmark")

    class Resp:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_lines(self):
            yield b"event: response.reasoning_text.delta"
            yield b"data: " + js.dumps({"type": "response.reasoning_text.delta", "delta": "hmm"}).encode()
            yield b"data: " + js.dumps({"type": "response.output_text.delta", "delta": "hi"}).encode()
            yield b""
            yield b"data: " + js.dumps({"type": "response.completed",
                                        "response": {"usage": {"input_tokens": 12, "output_tokens": 7}}}).encode()

    class Session:
        def __init__(self): self.headers = {}
        def post(self, url, json, timeout, stream): assert json["stream"] is True; return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=1, in_tok=[10], out_tok=[5], seconds=0.05, shared=True, q=q, stream=True)
    r = q.get(timeout=5)
    assert r["ok"] >= 1 and r["in"] == 12 * r["ok"] and r["out"] == 7 * r["ok"]
    assert len(r["ttft"]) == r["ok"] and all(t > 0 for t in r["ttft"])


def test_schema_and_reasoning_effort_land_in_the_request_body(monkeypatch):
    import multiprocessing as mp
    bench = _load("benchmark")
    seen = []

    class Resp:
        status_code = 200
        def json(self): return {"usage": {"input_tokens": 1, "output_tokens": 1}, "output": []}

    class Session:
        def __init__(self): self.headers = {}
        def post(self, url, json, timeout): seen.append(json); return Resp()

    monkeypatch.setattr(bench.requests, "Session", Session)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=1, in_tok=[10], out_tok=[5], seconds=0.05, shared=True, q=q,
                 schema=True, effort="low")
    q.get(timeout=5)
    body = seen[0]
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["schema"] == bench.SCHEMA
    assert body["reasoning"] == {"effort": "low"}


def test_size_fleet_rounds_up_and_prices_per_million_tokens():
    sf = _load("size_fleet")
    p = sf.plan(engine_rps=12.8, demand_rps=70, price_per_hour=5.85, gpus_per_instance=1,
                input_tokens=1000, output_tokens=190, headroom=0.15)
    assert p["instances"] == 7, "70 / (12.8 * 0.85) = 6.4 -> 7"
    assert p["fleet_price_per_hour"] == 7 * 5.85
    # demand 70 rps * 1190 tokens * 3600 s = 299.9M tokens/h at $40.95/h
    assert abs(p["price_per_million_tokens_at_demand"] - 40.95 / 299.88) < 1e-3
    assert p["utilisation_at_demand"] < 0.85
    p8 = sf.plan(12.8, 70, 33.14, 8, 1000, 190, 0.15)
    assert p8["instances"] == 1, "one eight-GPU instance holds 102 rps"


def test_quality_summary_reads_harness_output_and_buckets_by_prompt_length(tmp_path, capsys):
    import json as js
    q = _load("quality")
    d = tmp_path / "m" / "x"; d.mkdir(parents=True)
    (d / "results_1.json").write_text(js.dumps({"results": {
        "gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.96, "exact_match_stderr,strict-match": 0.009,
                  "exact_match,flexible-extract": 0.962, "exact_match_stderr,flexible-extract": 0.0086},
        "ifeval": {"prompt_level_strict_acc,none": 0.8, "prompt_level_strict_acc_stderr,none": 0.02,
                   "inst_level_strict_acc,none": 0.85, "inst_level_strict_acc_stderr,none": "N/A",
                   "prompt_level_loose_acc,none": 0.83, "prompt_level_loose_acc_stderr,none": 0.02}}}))
    rows = []
    for i in range(16):
        prompt = "q" * (100 + 50 * i)
        rows.append({"filter": "strict-match", "exact_match": 1.0 if i < 12 else 0.0,
                     "arguments": {"gen_args_0": {"arg_0": prompt, "arg_1": {}}}})
        rows.append({"filter": "flexible-extract", "exact_match": 1.0, "arguments": {"gen_args_0": {"arg_0": prompt}}})
    (d / "samples_gsm8k_1.jsonl").write_text("\n".join(js.dumps(r) for r in rows) + "\n")
    q.summarise(str(tmp_path))
    out = capsys.readouterr().out
    assert "exact_match,strict-match" in out and " 96.0%" in out and "ifeval" in out
    assert "by prompt length quartile (n=16)" in out
    assert out.strip().endswith("0.0%"), "the longest quartile scored 0, the loss is on long prompts"
