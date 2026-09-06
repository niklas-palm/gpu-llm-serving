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
