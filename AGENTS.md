# AGENTS.md

Instructions for coding agents working in this repository. Read it fully before changing anything.
Humans: start with `README.md`.

## Project

One CDK stack that serves an open-weight LLM with vLLM on ECS GPU instances (g7e), behind an internal
ALB and CloudFront, with an OpenAI-compatible API and a Bearer key. One config file. The docs carry as
much value as the code: every number in them was measured on this hardware, and the point of the sample
is that a reader can deploy it, understand it, and tune it without reading the source.

Python 3.11, CDK v2 (Python), pytest. No other build system.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && npm install -g aws-cdk
```

## Commands

| command | when |
|---|---|
| `python3 -m pytest tests/ -q` | before every commit; must pass; needs no AWS credentials |
| `cd infra && npx cdk synth` | after any stack change; needs credentials (one prefix-list lookup) |
| `cd infra && npx cdk deploy` | ~15 min for infrastructure, then minutes while the model loads |
| `python3 scripts/build_image.py` | after ANY change under `container/`; a deploy alone runs the old entrypoint |
| `python3 scripts/endpoint_info.py` | prints endpoint, key, model name, dashboard, and the two commands below |
| `python3 scripts/test_endpoint.py <url> --key <key>` | smoke test: health, both APIs, streaming, 64 concurrent |
| `python3 scripts/benchmark.py <url> --key <key>` | concurrency sweep; the numbers to size a fleet from |

## Layout

```
config.yaml              the only file a user edits; each key has a comment and a docs/tuning.md section
config.local.yaml        gitignored overrides: API key, image URI, anything account-specific
infra/app.py             loads and validates config, generates the API key once, synthesises
infra/hardware.py        instance catalog and derived values; pure functions, no AWS calls
infra/serving_stack.py   the stack: VPC, ECS, ALB, CloudFront, service, metrics sidecar, dashboard, alarms
container/serve          entrypoint: environment variables to vLLM flags; baked into the image
container/Dockerfile     vLLM base image plus the entrypoint
scripts/                 build_image, endpoint_info, test_endpoint, benchmark
tests/                   test_hardware (catalog, validation), test_template (synthesised template)
docs/tuning.md           measurements per config key, sizing, autoscaling, reading the dashboard
docs/troubleshooting.md  symptom, cause, fix
```

## Rules

1. **Do not add machinery for problems that have not happened.** Every construct, script and config key
   here earned its place by a failure or a measurement. A new one needs the same. Removing code is
   usually the better change.
2. **A new config key needs all of:** a comment in `config.yaml` saying what it does and the one trap,
   a default, validation in `hardware.py` or `app.py` raising a one-line `ConfigError`, a test, and a
   section in `docs/tuning.md` that the comment names. `0` means "let the engine decide".
3. **Nothing account-specific in tracked files.** Account ids (`111122223333` in tests only), hostnames,
   distribution ids, tokens, keys, company or customer names. If it came from a real deployment, it
   belongs in `config.local.yaml` or nowhere.
4. **Tests stay credential-free.** Context lookups are seeded in `AZ_CONTEXT` in `tests/test_template.py`.
   Test names are sentences saying what would break; docstrings say what broke before.
5. **Assert values, not presence.** Synth cannot catch a wrong IAM action, header name, alarm threshold
   or an apostrophe in a security-group description. All of those reached a live deploy once.
6. **Keep headings and anchors.** `config.yaml` comments and the README link to sections in `docs/` by
   title.
7. **Never delete a measurement or a documented failure to shorten a file.** Shorten the prose around it.

## Writing

Engineer to engineer. Short sentences. State the fact and the action; one sentence of justification per
default at most. Numbers in tables. No em-dashes. No "deliberately", "worth knowing", "genuinely",
"verified", "world-class", "battle-tested", "seamless", "robust", "leverage". A statement is a statement;
when something is estimated or not measured, say so instead of labelling the rest as verified.

Code comments explain why, not what, and record the failure that motivated a non-obvious choice.

## Common changes

- **Engine flag or startup behaviour**: edit `container/serve`, rebuild the image, deploy, confirm the
  `[serve] vllm serve ...` line in the task log shows the change.
- **Stack resource**: edit `serving_stack.py`, add or update a template test, synth with the shipped
  `config.yaml` in isolation (not only your local overrides), deploy once.
- **Instance type**: add to the catalog in `hardware.py` with vCPU and host RAM from
  `ec2 describe-instance-types`; the tests check derived values.
- **Dashboard or alarm**: widget titles are questions in plain language; alarm descriptions are written
  for someone on a phone who did not deploy this. Test the threshold value.

## Helping someone deploy and use this

Most people arriving with an agent want a running endpoint, not a code change. Walk them through this
order and check each step before the next.

1. **Preflight, before anything is created.** Region offers `g7e.2xlarge`
   (`aws ec2 describe-instance-type-offerings ... --location-type availability-zone`); if only some
   zones do, put them in `availabilityZones`. Quota for the purchase model they will use: on-demand
   `L-DB2E81BA`, spot `L-3819A6DF`, in vCPU, 8 per instance. `cdk bootstrap` once per account and
   region. CloudFront VPC origins must be supported in the region (they are in all commercial regions
   that offer g7e as of this writing).
2. **Configure.** `region`, `instanceType`, `modelId` in `config.yaml`. The default fleet is 16 and
   needs 128 vCPU of quota; with less, set `instanceCount` and `maxInstanceCount` in `config.local.yaml`.
   For a first deployment suggest `useSpot: true`: on-demand g7e has had no capacity in several regions
   at once, and spot in the same regions launched within a minute.
3. **Gated model?** Create the `gpu-llm-serving/hf-token` secret in the deployment region and set
   `hfTokenSecretName`. A 401 while pulling means no token; a 403 means the token's account has not
   accepted that model's licence.
4. **Build, then deploy.** `build_image.py --write-config`, then `cdk deploy`. Three to five minutes
   into the deploy, read the ASG scaling activities. `InsufficientInstanceCapacity` or
   `UnfulfillableCapacity` will not resolve soon: `cdk destroy`, switch purchase model, deploy again.
   Do not let CloudFormation wait; it will, for up to an hour.
5. **Confirm.** `endpoint_info.py`, then the smoke-test command it prints. Then `benchmark.py` if they
   need capacity numbers; set `maxInstanceCount` equal to `instanceCount` first.
6. **Read the dashboard with them.** `DashboardUrl` output. Top rows are the load balancer's view,
   bottom row is the engine's: waiting requests mean saturation, KV cache near 100% means preemption
   next, any preemptions mean lost work. Alarms fire only on sustained conditions.
7. **Leaving.** Park with both counts at 0 and deploy; instances are gone in about five minutes. Destroy
   only after they are gone. A destroy started while the CloudFront VPC origin is still `Deploying`
   fails; wait for `Deployed` and retry.

Names that are account-wide, not regional, and already carry the region so two stacks can coexist:
the CloudWatch dashboard and the CloudFront VPC origin. IAM roles are CDK-generated and unique.

## Operating a test deployment

- Use `instanceCount: 6`, `maxInstanceCount: 8`, `useSpot: true` in `config.local.yaml`. The shipped
  default is 16 and needs a quota increase.
- `UPDATE_IN_PROGRESS` blocks further deploys. `aws cloudformation cancel-update-stack` rolls back to
  the previous task definition.
- GPU instances are the entire cost. Never leave them running after a test unless asked to.

## Commits

Imperative subject under 70 characters. The body says why, names the failure or measurement behind the
change, and states what was deployed or tested. One logical change per commit.

## Done means

- Tests pass.
- Stack changes were synthesised from the shipped config and deployed once.
- `container/` changes were rebuilt into the image and seen in a task log.
- Docs changes: no em-dash character in the repo, every number that was there is still there.
- The fleet is parked.
