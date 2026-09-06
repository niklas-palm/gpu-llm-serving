#!/usr/bin/env python3
"""Print everything needed to call a deployed endpoint, ready to paste.

    python3 scripts/endpoint_info.py

Exists because the stack outputs are correct but not directly usable: a reader had to find the
endpoint, find the key, and assemble a request from a table in the README. This is the "how do I call
it" answer, and it should not require reading anything.
"""

from __future__ import annotations

import argparse
import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_STACK = "GpuLlmServing"


def config_region(default: str = "") -> str:
    """Region from config.yaml (and config.local.yaml), matching the other scripts."""
    import yaml
    # Honours $CONFIG, the same as infra/app.py, so this and the deploy read the same files.
    config_path = os.environ.get("CONFIG", os.path.join(ROOT, "config.yaml"))
    local_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), "config.local.yaml")
    region = default
    for path in (config_path, local_path):
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            region = ((yaml.safe_load(fh) or {}).get("region") or region)
    # No $AWS_REGION fallback: app.py refuses it on purpose, and the two tools must agree.
    return region


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None, help="defaults to `region` in config.yaml")
    ap.add_argument("--stack", default=DEFAULT_STACK)
    a = ap.parse_args()

    region = a.region or config_region()
    if not region:
        sys.exit("no region: set `region` in config.yaml, or pass --region")

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        stacks = cfn.describe_stacks(StackName=a.stack)["Stacks"]
    except Exception as e:                                    # noqa: BLE001 - reported, not raised
        sys.exit(f"could not read stack {a.stack} in {region}: "
                 f"{getattr(e, 'response', {}).get('Error', {}).get('Message', e)}")

    out = {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}
    endpoint = out.get("Endpoint", "")
    key = out.get("ApiKeyValue", "")
    model = out.get("ModelName", "model")
    if not endpoint or not key:
        sys.exit(f"stack {a.stack} has no Endpoint/ApiKeyValue outputs; is it fully deployed?")

    print(f"Endpoint     {endpoint}   (HTTPS via CloudFront; the load balancer behind it is internal)")
    print(f"API key      {key}")
    print(f"Model name   {model}")
    for label, k in (("GPUs/engine", "ResolvedTensorParallel"),
                     ("Purchase", "PurchaseModel")):
        if out.get(k):
            print(f"{label:<12} {out[k]}")
    # Printed with the endpoint rather than in a separate command, because the moment anyone needs the
    # dashboard is the moment the endpoint is misbehaving, and hunting for it in the console is the
    # thing that makes people give up and read logs instead.
    if out.get("DashboardUrl"):
        print(f"Dashboard    {out['DashboardUrl']}")

    print(f"""
--- curl, Responses API (the primary interface) ---

curl -sS -X POST "{endpoint}/v1/responses" \\
  -H "Authorization: Bearer {key}" \\
  -H 'Content-Type: application/json' \\
  -d '{{"model": "{model}", "input": "Say hello in five words."}}'

--- Python, OpenAI SDK ---

from openai import OpenAI

client = OpenAI(base_url="{endpoint}/v1", api_key="{key}")

print(client.responses.create(model="{model}", input="Say hello in five words.").output_text)

--- Smoke test the whole endpoint ---

python3 scripts/test_endpoint.py "{endpoint}" --key "{key}"

--- Find what the fleet holds up to (sweep concurrency; ~2 min per level) ---

python3 scripts/benchmark.py "{endpoint}" --key "{key}" --concurrency 64,128,256
""".rstrip())
    return 0


def _run() -> int:
    """Turn an AWS API failure into one actionable line instead of a botocore traceback.

    Worth doing because the most common failure by far is an expired SSO session, and unhandled it
    surfaced as `ClientError: An error occurred (ExpiredToken)` with a stack trace at whichever API
    call happened to come first - which tells the reader nothing about what to do.
    """
    try:
        return main()
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except (ClientError, BotoCoreError) as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
        msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
        print(f"\nAWS error: {code} - {msg}", file=sys.stderr)
        if code in ("ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId",
                    "UnrecognizedClientException", "AccessDenied", "AccessDeniedException",
                    "CredentialsError", "NoCredentialsError"):
            print("  Check your credentials (and that they are for the right account), then retry.",
                  file=sys.stderr)
        else:
            print("  Check that `region` in your config is correct and that these credentials can\n"
                  "  reach it.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_run())
