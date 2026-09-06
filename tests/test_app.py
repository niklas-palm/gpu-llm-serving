"""app.py's file side effect: the generated API key must land in config.local.yaml exactly once."""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "infra"))
_spec = importlib.util.spec_from_file_location("app", os.path.join(HERE, "..", "infra", "app.py"))
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


def _persist(tmp_path, monkeypatch, existing: str | None) -> str:
    path = tmp_path / "config.local.yaml"
    if existing is not None:
        path.write_text(existing)
    monkeypatch.setattr(app, "LOCAL_CONFIG_PATH", str(path))
    key = app._persist_generated_api_key()
    return path.read_text(), key


def test_a_missing_file_is_created_with_the_key(tmp_path, monkeypatch):
    body, key = _persist(tmp_path, monkeypatch, None)
    assert body.count("apiKey:") == 1 and key in body


def test_a_file_without_a_trailing_newline_gets_one_before_the_key(tmp_path, monkeypatch):
    body, key = _persist(tmp_path, monkeypatch, "region: us-east-2")
    assert body == f"region: us-east-2\napiKey: {key}\n"


def test_a_blank_apikey_line_is_replaced_not_duplicated(tmp_path, monkeypatch):
    """Appending gave two apiKey lines; the file worked only because PyYAML keeps the last one."""
    body, key = _persist(tmp_path, monkeypatch, "region: us-east-2\napiKey:\ninstanceCount: 2\n")
    assert body.count("apiKey:") == 1
    assert f"apiKey: {key}\n" in body and "instanceCount: 2" in body


def test_an_existing_key_is_left_alone_by_load_config(tmp_path, monkeypatch):
    """load_config only generates when no key is configured."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("region: us-east-2\ninstanceType: g7e.2xlarge\nmodelId: m\napiKey: keep-this-key-1234\nimage: x/y:z\n")
    monkeypatch.setattr(app, "CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(app, "LOCAL_CONFIG_PATH", str(tmp_path / "config.local.yaml"))
    assert app.load_config()["apiKey"] == "keep-this-key-1234"


@pytest.mark.parametrize("blank", ["apiKey:", 'apiKey: ""', "apiKey: ''", "apiKey: null", "apiKey: ~",
                                   'apiKey: "   "', "apiKey:   # set on first deploy"])
def test_every_spelling_of_a_blank_key_is_replaced_not_duplicated(tmp_path, monkeypatch, blank):
    body, key = _persist(tmp_path, monkeypatch, f"region: us-east-2\n{blank}\n")
    assert body.count("apiKey:") == 1 and f"apiKey: {key}" in body


def test_the_local_config_is_not_world_readable(tmp_path, monkeypatch):
    _persist(tmp_path, monkeypatch, None)
    assert oct(os.stat(tmp_path / "config.local.yaml").st_mode & 0o777) == "0o600"


def test_missing_serving_image_is_rejected_at_synth(tmp_path):
    """The upstream engine image would deploy and then serve nothing: this project's Dockerfile
    replaces the entrypoint with `serve`, which is what turns these env vars into
    engine flags. A default would trade a one-second failure for a 20-minute one that looks like a
    broken model.

    Driven through the real entry point as a subprocess, because app.py synthesises on import.
    """
    import subprocess
    import textwrap
    root = os.path.join(HERE, "..")

    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent("""
        region: us-west-2
        instanceType: g7e.2xlarge
        modelId: some-org/some-model
    """))

    env = {**os.environ, "CONFIG": str(cfg), "CDK_DEFAULT_ACCOUNT": "111122223333"}
    env.pop("SERVING_IMAGE", None)
    r = subprocess.run([sys.executable, os.path.join(root, "infra", "app.py")],
                       capture_output=True, text=True, env=env)

    assert r.returncode != 0, "a config with no image must not synthesise"
    out = r.stdout + r.stderr
    assert "image" in out.lower(), f"the error must say what is missing:\n{out}"
    assert "build_image.py" in out, f"and how to produce one:\n{out}"




def test_a_whitespace_only_required_value_is_missing(tmp_path, monkeypatch):
    """`modelId: "  "` passed the required-values check, synthesised MODEL_ID="" and the task died
    after a full deploy with 'MODEL_ID is required'."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text('region: us-east-2\ninstanceType: g7e.2xlarge\nmodelId: "   "\napiKey: some-stable-key-1234\n')
    monkeypatch.setattr(app, "CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(app, "LOCAL_CONFIG_PATH", str(tmp_path / "config.local.yaml"))
    with pytest.raises(app.ConfigError, match="missing required values: modelId"):
        app.load_config()


def test_a_non_string_apikey_is_not_treated_as_blank(tmp_path, monkeypatch):
    """load_config called `apiKey: 0` blank and generated a key, but the line-replacer did not, so the
    file got a second apiKey line. One predicate now: 0 is a value, and synth rejects it."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("region: us-east-2\ninstanceType: g7e.2xlarge\nmodelId: org/m\napiKey: 0\nimage: x\n")
    monkeypatch.setattr(app, "CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(app, "LOCAL_CONFIG_PATH", str(tmp_path / "config.local.yaml"))
    assert app.load_config()["apiKey"] == 0
    assert not (tmp_path / "config.local.yaml").exists()
