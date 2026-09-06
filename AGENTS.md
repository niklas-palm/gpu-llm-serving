# AGENTS.md

Instructions for coding agents working in this repository. Humans: the README is for you; this file is
short on purpose.

## What this is

A CDK stack that serves an open-weight LLM with vLLM on ECS GPU instances (g7e), behind an internal ALB
and CloudFront, with an OpenAI-compatible API. One config file. The docs are the product as much as the
code: every number in them was measured on this hardware.

## Layout

```
config.yaml              the only file a user edits; every key has a comment and a docs/tuning.md section
config.local.yaml        gitignored overrides (API key, image URI, account-specific values); never commit
infra/app.py             loads and validates config, generates the API key once, synthesises
infra/hardware.py        instance catalog and derived values; pure functions, no AWS calls
infra/serving_stack.py   the stack: VPC, ECS, ALB, CloudFront, service, metrics sidecar, dashboard
container/serve          entrypoint: environment to vLLM flags; baked into the image
container/Dockerfile     vLLM base image plus the entrypoint
scripts/build_image.py   builds and pushes the image in CodeBuild
scripts/endpoint_info.py prints endpoint, key, model name, dashboard, ready-to-paste requests
scripts/test_endpoint.py smoke test against a deployed endpoint
tests/                   template and catalog tests; no AWS credentials needed
docs/tuning.md           measurements and how to read your own
docs/troubleshooting.md  symptom, cause, fix
```

## Commands

```bash
python3 -m pytest tests/ -q                 # before every commit; must pass, no credentials needed
cd infra && npx cdk synth                   # needs credentials (one prefix-list lookup at synth)
cd infra && npx cdk deploy                  # ~15 min infrastructure, then minutes of model load
python3 scripts/build_image.py              # required after ANY change under container/
python3 scripts/endpoint_info.py            # then run the smoke-test command it prints
```

## Rules

1. **Do not add machinery for problems that have not happened.** Every construct, script and config key
   here earned its place by a failure or a measurement. A new one needs the same.
2. **Config lives in `config.yaml`.** A new key needs: a comment saying what it does and the one trap,
   a default, validation in `hardware.py` or `app.py` with a one-line `ConfigError`, a test, and a
   section in `docs/tuning.md` that the comment names. Zero is the convention for "let the engine
   decide".
3. **Nothing account-specific in tracked files.** No account ids (use `111122223333` in tests), no
   hostnames, no tokens, no company or customer names. `config.local.yaml` and `cdk.context.json` are
   gitignored for this reason.
4. **Changes under `container/` need an image rebuild before they exist.** A deploy alone runs the old
   entrypoint. The smoke test reads the served model name from `/v1/models` so it cannot tell you; check
   the entrypoint line in the task log.
5. **Tests must stay credential-free.** Lookups are seeded in `AZ_CONTEXT` in `tests/test_template.py`.
6. **Synth cannot catch wrong values in valid templates.** IAM actions, header names, EC2 rule
   description characters, alarm thresholds: write a test that asserts the value, not just the presence.
7. **Do not remove a documented failure or measurement to shorten a file.** Shorten the prose around it.

## Writing

Engineer to engineer. Short sentences. State the fact and the action. No em-dashes. No "deliberately",
"worth knowing", "genuinely", "verified", "world-class", "battle-tested". A statement in these docs is a
statement; when something is estimated or not measured, say so, do not label the rest as verified.
Numbers go in tables. Keep every heading other files link to.

## Operating a test deployment

- Test with `instanceCount: 6`, `maxInstanceCount: 8` in `config.local.yaml`; the shipped default is 16
  and needs a quota increase. Use spot when on-demand has no capacity.
- Check the ASG activity a few minutes into a deploy. CloudFormation waits up to an hour on a service
  that can never place a task; the ASG says why in seconds.
- Park before you leave: `instanceCount: 0`, `maxInstanceCount: 0`, deploy. Instances are gone in about
  five minutes. Destroy only after they are gone, or the destroy stalls for 25 minutes on the service.
- A stack in `UPDATE_IN_PROGRESS` cannot be updated. `cancel-update-stack` rolls back to the previous
  task definition.

## Before you finish

- `python3 -m pytest tests/ -q` passes.
- If the stack changed: synthesised with the shipped `config.yaml` in isolation, not only with your local
  overrides, and deployed once.
- If `container/` changed: image rebuilt and the entrypoint line in a task log shows the change.
- If docs changed: no em-dash character anywhere in the repo, every number that was there is still there,
  the fleet is parked.
