#!/usr/bin/env python3
"""CDK entry point. Reads config.yaml, validates it, and synthesises one stack.

    cd infra && cdk deploy
"""

from __future__ import annotations

import os
import secrets
import sys

import aws_cdk as cdk
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware import ConfigError  # noqa: E402
from serving_stack import ServingStack  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.environ.get("CONFIG", os.path.join(ROOT, "config.yaml"))
# Deep-merged OVER config.yaml when present, and gitignored. This is where deployment-specific values
# belong - the API key, a region, an instance count for an experiment - so the tracked file stays
# generic. Resolved NEXT TO whichever config is in use, not from the repo root: deriving it from ROOT
# meant a developer's real config.local.yaml was merged into tests that point CONFIG at a temporary
# file, which breaks isolation.
LOCAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(CONFIG_PATH)),
                                 "config.local.yaml")

# Only these. Everything else has a default.
REQUIRED = ("region", "instanceType", "modelId")

# Config keys settable from the environment, for CI. The image URI is the case: build_image.py can
# write it into the config, but a pipeline that builds and deploys in one run has nowhere to write.
#
# `region` is NOT here. $AWS_REGION is commonly set to
# something unrelated to this project, so honouring it would let a shell variable silently retarget the
# whole deployment - which is exactly what happened while testing this: a config saying us-east-2
# synthesised against eu-north-1 and failed on an unrelated error about availability zones. The region
# lives in config.yaml, or in config.local.yaml if it must stay uncommitted.
# An EMPTY value counts as unset and overrides nothing (see _apply_env), because an empty environment
# variable is indistinguishable from an unset one often enough that "empty means default" is the safer
# reading.
ENV_OVERRIDES = {
    "SERVING_IMAGE": "image",
    # For CI, where config.local.yaml does not exist: without this every run would generate a fresh
    # key and rotate it for every client.
    "API_KEY": "apiKey",
}


def _merge(base: dict, overlay: dict) -> dict:
    """Recursive merge, so a local override can set one tuning key without restating the block."""
    out = dict(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env(cfg: dict) -> dict:
    for env_name, key in ENV_OVERRIDES.items():
        if os.environ.get(env_name):
            cfg[key] = os.environ[env_name]
    return cfg


def _persist_generated_api_key() -> str:
    """Generate an API key once and append it to the local config, so redeploys keep it.

    Writing a file during synth is a side effect, and normally I would avoid one. It is the right
    trade here: the alternative is either a key that rotates on every deploy (breaking clients) or an
    extra mandatory step before a first deploy can succeed. It happens exactly once, only when no key
    is configured, and it says so loudly.
    """
    key = secrets.token_urlsafe(30)
    try:
        exists = os.path.exists(LOCAL_CONFIG_PATH)
        # Append a leading newline if the file does not already end with one. Without this, a
        # hand-written config.local.yaml whose last line has no trailing newline gets `apiKey:` glued
        # onto it - and because the key is only generated once, every later deploy then fails on a YAML
        # scanner error until someone repairs the file by hand.
        needs_newline = False
        body = ""
        if exists:
            with open(LOCAL_CONFIG_PATH) as fh:
                body = fh.read()
            needs_newline = bool(body) and not body.endswith("\n")
        # A blank `apiKey:` line already in the file is replaced, not appended to. Appending produced
        # two keys, and the file only worked because PyYAML takes the last.
        if exists and any(line.split("#", 1)[0].strip() in ("apiKey:", 'apiKey: ""', "apiKey: ''")
                          for line in body.splitlines()):
            lines = [f"apiKey: {key}" if line.split("#", 1)[0].strip().startswith("apiKey:") else line
                     for line in body.splitlines()]
            with open(LOCAL_CONFIG_PATH, "w") as fh:
                fh.write("\n".join(lines) + "\n")
        else:
            with open(LOCAL_CONFIG_PATH, "a") as fh:
                if not exists:
                    fh.write("# Local overrides, deep-merged over config.yaml. Gitignored.\n")
                elif needs_newline:
                    fh.write("\n")
                fh.write(f"apiKey: {key}\n")
    except OSError as e:
        raise ConfigError(
            f"no apiKey configured, and {LOCAL_CONFIG_PATH} could not be written ({e}).\n"
            "  Set apiKey in your config instead - it must be stable, because a value generated on\n"
            "  each synth would rotate the key on every deploy."
        ) from e
    print(f"generated an API key and saved it to {os.path.basename(LOCAL_CONFIG_PATH)} "
          f"(gitignored). Redeploys will reuse it.", file=sys.stderr)
    return key


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh) or {}

    if os.path.exists(LOCAL_CONFIG_PATH):
        with open(LOCAL_CONFIG_PATH) as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
        print(f"merged local overrides from {os.path.basename(LOCAL_CONFIG_PATH)}",
              file=sys.stderr)

    cfg = _apply_env(cfg)

    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        raise ConfigError(
            "config.yaml is missing required values: " + ", ".join(missing)
            + "\n  See README.md 'Configure' for what each one is."
        )

    # A STABLE api key. Generated once and persisted, never regenerated.
    #
    # Synth runs on every `cdk deploy`, so generating a value here without saving it would mint a new
    # key each time - silently breaking every existing client, and making `cdk diff` always dirty. It
    # goes to config.local.yaml because that file is gitignored: the key is not confidential from
    # anyone who can read the deployed stack (an ALB rule needs a literal), but it should still not be
    # committed to a shared repository.
    if not str(cfg.get("apiKey") or "").strip():
        cfg["apiKey"] = _persist_generated_api_key()

    # The serving image, and it must be THIS project's image rather than the upstream engine image.
    #
    # No default. The obvious fallback - the public vllm/vllm-openai image - would
    # synthesise and deploy happily and then not work: container/Dockerfile replaces the entrypoint
    # with `serve`, which translates these env vars into engine flags. The upstream image expects
    # vLLM's own command-line arguments and is passed none, so the task would start and serve nothing.
    # Failing here costs a second; failing that way costs a 20-minute deploy and looks like a broken
    # model.
    if not cfg.get("image"):
        raise ConfigError(
            "no serving image configured.\n"
            "  Build and push it first; --write-config saves the URI to config.local.yaml:\n"
            f"    python3 scripts/build_image.py --region {cfg['region']} --write-config\n"
            "  Or set SERVING_IMAGE to an image built from container/Dockerfile."
        )
    return cfg


def main() -> None:
    # Wraps the stack as well as the config load. ServingStack re-validates everything itself and
    # raises a few checks that only it can make (the availability-zone list, the api key), so a
    # handler around load_config() alone printed those as tracebacks. Validating in exactly one place
    # also means app.py cannot drift from the stack.
    try:
        cfg = load_config()
        app = cdk.App()
        ServingStack(
            app, "GpuLlmServing",
            cfg=cfg,
            env=cdk.Environment(
                account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                region=cfg["region"],
            ),
            description=f"vLLM on {cfg['instanceType']} behind an ALB",
        )
    except ConfigError as e:
        sys.exit(f"\nConfiguration error:\n  {e}\n")
    app.synth()


if __name__ == "__main__":
    main()
