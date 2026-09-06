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
    for name in ("build_image", "endpoint_info", "test_endpoint", "benchmark"):
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
    q = mp.Queue()
    bench.worker("http://u", "k", "m", conc=2, in_tok=10, out_tok=5, seconds=0.05, shared=True, q=q)
    r = q.get(timeout=5)
    assert r["ok"] == 0 and r["lat"] == [] and r["fail"] > 0
