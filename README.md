# GPU LLM Serving

Deploy an open-weight LLM on AWS GPU instances behind a load balancer, with an OpenAI-compatible API.

One CDK stack, one config file. You get an endpoint serving the **Responses API**
(`/v1/responses`) and **Chat Completions** (`/v1/chat/completions`), authenticated with an API key.

```
  HTTPS + Bearer key                   VPC origin      internal      ECS service (1..N engines)
  ────────────────►  CloudFront  ───────────────────►  ALB  ──────►  on g7e GPU instances, running vLLM
```

The defaults come from measurements on this hardware; [docs/tuning.md](docs/tuning.md) has the numbers.
[docs/troubleshooting.md](docs/troubleshooting.md) is organised by symptom.

**What it costs:** one `g7e.2xlarge` is about $3.30/hour on-demand, the shipped 16 about $53/hour;
idle at zero instances about $50/month. Details and how to stop paying: [Operating](#operating).

---

## Quickstart

The default fleet is **16 `g7e.2xlarge`**, autoscaling to 24: 128 vCPU at the minimum, 192 at the
ceiling. Request the G-instance quota for that before deploying (see [Check your quota](#check-your-quota));
a fresh account rarely has it. With less quota, lower `instanceCount` and `maxInstanceCount` in
`config.local.yaml` to what fits, and raise them later. For a first deployment, set `useSpot: true` there
too: on-demand g7e.2xlarge had no capacity in two regions on the same day while spot launched in a minute.

```bash
pip install -r requirements.txt && npm install -g aws-cdk

# Edit config.yaml: region, instanceType, modelId.
# Pick a region that has g7e and check the quota matching your `useSpot` setting; see "Before you start".

# Build the serving image. Runs in CodeBuild, not on your machine; ~10-15 min.
python3 scripts/build_image.py --write-config

# Deploy. ~15 min for infrastructure (CloudFront is the slow part), then several more while the model loads.
cd infra && cdk bootstrap && cdk deploy && cd ..

# Print the endpoint, the key, a ready-to-paste curl and Python snippet,
# and the exact smoke-test command with your values already filled in.
python3 scripts/endpoint_info.py

# Then run the smoke test it printed: health, both APIs, streaming, 64 concurrent requests.
```

[Deploy](#deploy) has the detail behind each step.

**The endpoint is HTTPS out of the box**: a CloudFront URL with its own certificate; no domain,
certificate, or hosted zone needed. See [Access](#access) for the one limit.

---

## Before you start

- An AWS account and credentials
- The **AWS CLI**, configured
- Permissions to create: CloudFormation, EC2, ECS, ELB, ECR, S3, CodeBuild, Secrets Manager, CloudFront,
  and IAM, including `iam:CreateRole`, `iam:PutRolePolicy` **and `iam:PassRole`** (the scripts create a
  CodeBuild service role and pass it)
- Node.js 20 or 22 LTS (for the CDK CLI; newer majors work but jsii warns) and Python 3.11 or 3.12.
  `requirements.txt` pins exact versions; loosen them if you must, but they are what was tested
- A virtualenv; system and Homebrew Python refuse `pip install` without one:
  `python3 -m venv .venv && source .venv/bin/activate`

**Docker is not required.** The image is built in CodeBuild, so Docker Hub → build → ECR stays inside
AWS, with no cross-architecture risk from an ARM laptop building x86_64.

**There is no AMI to build.** The stack resolves the **ECS-optimised Amazon Linux 2023 GPU AMI** from
its SSM public parameter at deploy time; the same config works in any region, and driver updates arrive
by redeploying. Do not use the Amazon Linux **2** GPU AMI: its driver is too old, the instance registers
with **zero** GPUs, and tasks hang forever. See [docs/troubleshooting.md](docs/troubleshooting.md).

### Check your quota

The most common reason a deployment never gets an instance. Increases can take hours. **Spot and
on-demand have separate quotas.**

| Quota | Code | Typical default |
|---|---|---|
| All G and VT Spot Instance Requests | `L-3819A6DF` | **64 vCPU** |
| Running On-Demand G and VT instances | `L-DB2E81BA` | 64–768 vCPU (varies) |

| Instance | GPUs | VRAM | vCPU | Host RAM | Fits a 64 vCPU quota? |
|---|---|---|---|---|---|
| `g7e.2xlarge` | 1 | 96 GiB | 8 | 64 GiB | yes, **8 of them, exactly** |
| `g7e.4xlarge` | 1 | 96 GiB | 16 | 128 GiB | yes, 4 |
| `g7e.8xlarge` | 1 | 96 GiB | 32 | 256 GiB | yes, 2 |
| `g7e.12xlarge` | 2 | 192 GiB | 48 | 512 GiB | yes, 1 |
| `g7e.24xlarge` | 4 | 384 GiB | 96 | 1 TiB | **no**, needs 96 |
| `g7e.48xlarge` | 8 | 768 GiB | 192 | 2 TiB | **no**, needs 192 |

The default fits 64 vCPU. Larger needs an increase a new account will not have.

**Check the quota that matches your `useSpot` setting**, or the stack succeeds and no instance ever
launches:

```bash
# useSpot: false (the shipped default) -> on-demand quota
aws service-quotas get-service-quota --region <region> \
  --service-code ec2 --quota-code L-DB2E81BA

# useSpot: true -> spot quota, which is the one people hit
aws service-quotas get-service-quota --region <region> \
  --service-code ec2 --quota-code L-3819A6DF
```

A new account's on-demand G quota is often lower than the table suggests, sometimes 0. To request an
increase:

```bash
aws service-quotas request-service-quota-increase --region <region> \
  --service-code ec2 --quota-code L-DB2E81BA --desired-value 96
```

Requests of roughly +50% over the current value are most likely to be approved automatically.

### Check for capacity

Quota and offerings say what you are allowed to launch, not what exists right now. On-demand g7e has had
no capacity in whole regions for hours at a time, and nothing tells you in advance. Spot has a signal:

```bash
aws ec2 get-spot-placement-scores --region <region> --region-names <region> \
  --instance-types g7e.2xlarge --target-capacity 6 --single-availability-zone \
  --query 'SpotPlacementScores[].[AvailabilityZoneId,Score]' --output text     # 10 is best, 1 is none
```

For on-demand there is no such API; the only test is a launch. Deploy step 3 shows how to read the
answer within minutes rather than after CloudFormation's hour-long wait.

### Check the instance type exists where you are deploying

g7e is not in every region:

```bash
aws ec2 describe-instance-type-offerings --region <region> \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g7e.2xlarge \
  --query 'InstanceTypeOfferings[].Location'
```

Empty means pick another region. Fewer zones than the region has (`us-east-2`: three zones, g7e in two)
means launches into the missing zone fail with `Unsupported` and the group retries elsewhere; list the
zones in `availabilityZones` if you want to avoid the noise.

---

## Configure

Everything lives in [`config.yaml`](config.yaml), which documents each setting. The minimum config:

```yaml
region: us-east-2
instanceType: g7e.2xlarge
modelId: Qwen/Qwen3-30B-A3B-Instruct-2507
```

**The weights are a choice.** The shipped config quantises the bf16 checkpoint to fp8 at load time.
Smaller weights of the same model run faster and need fewer instances; measured on this GPU with unique
1,000-token prompts, per instance at the same load:

| `modelId` | `quantization` | input tok/s per instance, 96 requests each on an 8-instance fleet | note |
|---|---|---|---|
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | `"fp8"` | 14,900 | shipped default |
| `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | `""` | same | publisher's fp8; half the download |
| `nvidia/Qwen3-30B-A3B-NVFP4` | `""` | 19,100 (+28%) | 4-bit; output quality not measured |
| any of the above + EAGLE-3 speculator in `extraArgs` | | +24% to +41% | one line; see [docs/tuning.md](docs/tuning.md) |

bf16 with no quantisation is half the throughput of fp8. Details, and the caveats, in *Quantisation is two
independent decisions* and *Speculative decoding with EAGLE-3* in [docs/tuning.md](docs/tuning.md).

The shipped fleet defaults are sized for a production workload; `config.yaml` shows the arithmetic:

```yaml
instanceCount: 16          # 128 vCPU: request the quota first, or lower it in config.local.yaml
maxInstanceCount: 24       # autoscaling ceiling; equal to instanceCount = fixed fleet
useSpot: false             # on-demand: the safer default. Spot is often the only way to GET these GPUs
quantization: "fp8"
```

`useSpot: false` is the safer default, but **spot is frequently the only pool with capacity** for this
GPU class, with its own, smaller quota. If on-demand cannot find instances, set `useSpot: true`. See
[Spot and instance protection](#spot-and-instance-protection).

One engine per GPU is **derived, not configured**: `replicas: 0` means "GPUs ÷ tensorParallel", so a
`g7e.48xlarge` gets eight engines.

### Values you would rather not commit

`config.local.yaml` is **gitignored** and deep-merged over `config.yaml`, so it only needs the keys you
change:

```yaml
# config.local.yaml
region: eu-west-1
instanceCount: 2
maxInstanceCount: 2
```

The API key lands here too, generated on first deploy (see [Notes](#notes-and-limitations)).
`SERVING_IMAGE` and `API_KEY` are the two environment variables that override both files, for a build-and-deploy
pipeline; an **empty** variable counts as unset. `region` is **not** settable this way: `$AWS_REGION` is
often set to something unrelated.

### Access

The endpoint is `https://<id>.cloudfront.net`, printed by `python3 scripts/endpoint_info.py` and in the
`Endpoint` stack output. CloudFront reaches the load balancer through a **VPC origin**, a network
interface in the private subnets. The load balancer is internal and admits only CloudFront's addresses,
so the hop carrying your API key never crosses the public internet. Plain `http://` gets a 403 from
CloudFront, not a redirect (a redirect would turn POST into GET).

CloudFront is a pass-through, not a cache: nothing cached, every header forwarded, compression off,
**streaming works**. One limit:

**A non-streamed answer must finish within 120 seconds.** The read timeout resets on every byte, so a
streamed response can run as long as it likes; a non-streamed one gets a 504 after 120 s while the
engine finishes anyway. 120 s is the most the default quota allows. At ~30 tokens/s per request on a
busy engine, that is roughly 3,000 tokens. For anything longer, stream:

```bash
curl -sN -X POST "$ENDPOINT/v1/responses" -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model": "'$MODEL'", "input": "Write a long story.", "max_output_tokens": 4000, "stream": true}'
```

**Why there is no secret "origin header".** That header defends a *public* origin. This one is internal,
reachable only through the VPC origin, and the load balancer already validates a header on every
request: your API key. If you make the load balancer internet-facing, add the header the same day.

**TLS.** The default `*.cloudfront.net` certificate accepts TLS 1.0 and 1.1 as well as 1.2, and that
cannot be raised without a custom domain and certificate. If your compliance bar requires TLS 1.2 only,
that is the one reason to add a domain: `domain_names`, `certificate` and `minimum_protocol_version` on
the distribution.

**Anyone with the key can call it, from anywhere.** To restrict by network, attach an AWS WAF web ACL
with an IP set to the distribution (`web_acl_id` on the `Distribution` in `infra/serving_stack.py`).
Not included: it is a cost and a policy decision.

**What it costs**: roughly $0.01 per 10,000 requests plus $0.085/GB out. At 100 requests/second
sustained, about $280/month, around 2% of the GPU fleet. VPC origins are free.

### Before you change the tuning block

Leave it alone unless you have a reason; [docs/tuning.md](docs/tuning.md) has a section per setting.
Two decisions to make *before* deploying:

**Quantise the weights.** The largest effect available, most of all when prompts do *not* share a
prefix. At each precision's latency-passing knee, bf16 sustained **7,903 input tok/s per GPU against
15,941 for FP8**: roughly **twice the fleet** for the same work, from lower throughput at matched load
and a lower usable load (64 concurrent per GPU against 128). Use a publisher's official FP8 build as
`modelId` if one exists. If not, `quantization: "fp8"` measured level with an official build.

**If you leave the weights unquantised, set `kvCacheDtype: auto`.** fp8 cache with bf16 weights runs out
of VRAM under load, and the lower utilisation that survives it is slower than not doing it at all.
`cdk synth` warns on the combination.

Instance choice, topology, operating concurrency, and what *not* to bother with:
[docs/tuning.md](docs/tuning.md).

### Spot and instance protection

Spot instances are reclaimed with two minutes' notice; normally the ASG launches a replacement. You can
protect a hard-won instance by suspending the ASG processes that can take it away:

```bash
REGION=eu-west-2        # the region in your config
ASG=$(aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`AsgName`].OutputValue' --output text)

aws autoscaling suspend-processes --auto-scaling-group-name "$ASG" --region "$REGION" \
  --scaling-processes ReplaceUnhealthy Terminate AZRebalance
```

**Understand what this trades.** It also stops the ASG replacing that instance when spot reclaims it.
The phantom lingers as `InService`/`Healthy` while `describe-instances` returns
`InvalidInstanceID.NotFound`, counts toward desired capacity, and the service stays down until you
`resume-processes`. Suspend only for an experiment you are watching; never for anything unattended.

---

## Deploy

### 1. Build and push the serving image

```bash
python3 scripts/build_image.py --write-config
```

The region comes from `config.yaml`, so it cannot drift from the deployment.

**The build runs in CodeBuild, not on your machine.** The base image is ~9 GB. Roughly 10–15 minutes;
each phase is printed. It creates what it needs, only if missing, so re-running is safe:

| Created | Purpose |
|---|---|
| ECR repository `gpu-llm-serving` | holds the image; untagged images expire after 7 days |
| S3 bucket `gpu-llm-serving-build-<account>-<region>` | the build context; private and encrypted |
| IAM role `GpuLlmServingCodeBuildRole` | lets CodeBuild read the context and push to ECR |
| CodeBuild project `gpu-llm-serving-build` | the build itself |

`--write-config` writes the URI into **`config.local.yaml`** (gitignored), because the URI contains your
account id. Without the flag it prints the line to paste.

A failed build prints the failing phase and the log location. To check later:
`python3 scripts/build_image.py --status <build-id>`.

### 2. Gated model? Add your Hugging Face token (optional)

Public models need nothing here. For a gated one (Llama, Gemma, anything you had to click "agree" for),
put your Hugging Face token in Secrets Manager once and name the secret in your config. The engine reads
it as `HF_TOKEN`; the token never enters the template, the outputs or the logs.

```bash
aws secretsmanager create-secret --region <region> --name gpu-llm-serving/hf-token --secret-string "$HF_TOKEN"
```
```yaml
# config.local.yaml
hfTokenSecretName: gpu-llm-serving/hf-token
```

Without it, a gated `modelId` fails while pulling weights with a **401** in the task's log. A **403**
"not in the authorized list" means the token works but its account has not accepted that model's licence
on Hugging Face; accept it there and redeploy.

Weights are pulled from Hugging Face by the first task on each instance into a shared cache on the host,
so restarts and further tasks on that instance download nothing.

### 3. Deploy

```bash
cd infra
cdk bootstrap        # first time in this account+region only
cdk deploy
cd ..                # the commands in steps 4 and 5 run from the repo root
```

One value is **looked up at synth time**: the id of CloudFront's managed prefix list, which differs per
region and is what the load balancer's security group admits. So `cdk deploy` needs credentials and a
resolvable account id (`CDK_DEFAULT_ACCOUNT`, set by the CLI from your profile). The result is cached in
`infra/cdk.context.json`, gitignored because it holds your account id.

Roughly 15 minutes for the infrastructure, then several more while the container pulls the image and
loads the model.

**Check the instance launched, a few minutes in.** This is the step that fails, and it fails quietly:
the ASG reports the reason within seconds, but CloudFormation keeps waiting on the ECS service for up to
an hour before it gives up.

```bash
REGION=eu-west-2        # the region in your config
ASG=$(aws autoscaling describe-auto-scaling-groups --region "$REGION" \
  --query 'AutoScalingGroups[?starts_with(AutoScalingGroupName,`GpuLlmServing`)].AutoScalingGroupName|[0]' --output text)
aws autoscaling describe-scaling-activities --region "$REGION" --auto-scaling-group-name "$ASG" \
  --max-items 3 --query 'Activities[].[StatusCode,StatusMessage]' --output text
```

`Successful` means an instance is up and the model is loading. `We currently do not have sufficient
g7e.2xlarge capacity` means on-demand has none right now, and it will not get any soon: run
`cdk destroy`, set `useSpot: true`, and deploy again. A quota error means the quota check above was
skipped. More cases: [docs/troubleshooting.md](docs/troubleshooting.md), *the ASG never launches an
instance*.

Once an instance is up, watch the model load:

```bash
CLUSTER=$(aws cloudformation describe-stacks --stack-name GpuLlmServing --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterName`].OutputValue' --output text)
SERVICE=$(aws ecs list-services --cluster "$CLUSTER" --region "$REGION" \
  --query 'serviceArns[0]' --output text | awk -F/ '{print $NF}')
aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].{running:runningCount,desired:desiredCount,pending:pendingCount}'
```

### Changing the model, tuning or image later

Edit the config and `cdk deploy` again. A change to the task definition replaces **every engine at
once**: with one GPU per task the new task cannot start until the old one has released the GPU, so the
endpoint returns 503 for the reload, five to fifteen minutes. Plan it, or run a second stack and move
your gateway to it. Changes that touch only the fleet size, alarms or dashboard do not restart engines.

### 4. Call it

```bash
python3 scripts/endpoint_info.py
```

Prints the endpoint, API key, dashboard URL, and a working `curl` and Python snippet. Everything it shows
is a stack output:

| Output | What it is |
|---|---|
| `Endpoint` | base URL: `https://<id>.cloudfront.net` |
| `ResponsesApi` | full URL of `/v1/responses`, the primary interface |
| `ChatCompletionsApi` | full URL of `/v1/chat/completions` |
| `DistributionId` | the CloudFront distribution, for `aws cloudfront` commands |
| `PurchaseModel` | `spot` or `on-demand`, as deployed |
| `ApiKeyValue` | the key to send as `Authorization: Bearer <key>` |
| `ModelName` | the model id to put in requests |
| `DashboardUrl`, `DashboardName` | the CloudWatch dashboard |
| `ClusterName`, `AsgName`, `LogGroup` | for the operational commands below |
| `ResolvedTensorParallel`, `ContainerMemoryMib` | what the derivation chose |

### 5. Test it

Step 4 prints this command with your values substituted. In full:

```bash
python3 scripts/test_endpoint.py "<Endpoint output>" --key "<ApiKeyValue output>"
```

Checks health, both API shapes, streaming, and 64 concurrent requests. If anything fails, see
[docs/troubleshooting.md](docs/troubleshooting.md).

That is a smoke test. To find what the fleet holds up to, sweep concurrency:

```bash
python3 scripts/benchmark.py "<Endpoint output>" --key "<ApiKeyValue output>" \
  --concurrency 64,128,256,512 --input-tokens 1000 --output-tokens 190 --seconds 90
```

One row per level: requests/sec, tokens/sec, p50/p95/p99, and the decode speed one request saw. Size on
the aggregate columns at the highest level whose p95 is inside your budget. Use your own prompt shape;
whether prompts share a prefix changes the answer by about 2×, and set `maxInstanceCount` equal to
`instanceCount` first so autoscaling does not move under the measurement. How to read the results:
*Choosing an operating concurrency* in [docs/tuning.md](docs/tuning.md).

---

## Calling the endpoint

Authentication is `Authorization: Bearer <key>`, the header every OpenAI-compatible client and gateway
sends. Requests without a valid key get a `403` from the load balancer and never reach the model. The
model name is the `modelId` you configured (`ModelName` output); `/v1/models` returns it. Read
[what this key is and is not](#notes-and-limitations) before putting anything sensitive behind it.

```bash
ENDPOINT=...; API_KEY=...; MODEL=...      # the three outputs, or copy from scripts/endpoint_info.py
```

### Responses API: the primary interface

```bash
curl -X POST "$ENDPOINT/v1/responses" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "'$MODEL'", "input": "Explain load balancing in two sentences."}'
```

### Chat Completions: the compatibility option

For clients already written against Chat Completions. New code should use the Responses API.

```bash
curl -X POST "$ENDPOINT/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "'$MODEL'", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

### With the OpenAI SDK

Not a dependency of this project; `pip install openai`. No custom headers: the key is the API key.

```python
from openai import OpenAI

client = OpenAI(base_url=f"{ENDPOINT}/v1", api_key=API_KEY)

print(client.responses.create(model=MODEL, input="Hello").output_text)              # prefer this
print(client.chat.completions.create(                                             # existing clients
    model=MODEL, messages=[{"role": "user", "content": "Hello"}]).choices[0].message.content)
```

### Behind an AI gateway

Register it as an OpenAI-compatible provider:

| setting | value |
|---|---|
| base URL | `<Endpoint>/v1` |
| API key | `ApiKeyValue`, sent as `Authorization: Bearer` |
| model | `ModelName` |
| timeout | streamed requests have no limit; non-streamed must finish within 120 s, see [Access](#access) |

Nothing else is needed. The gateway's own auth, rate limits and logging sit in front; this endpoint sees
one shared key from the gateway.

### Available endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/responses` | **Responses API, the primary interface** |
| `GET /v1/responses/{id}` | retrieve a response |
| `POST /v1/responses/{id}/cancel` | cancel |
| `POST /v1/chat/completions` | Chat Completions, for existing clients |
| `POST /v1/completions` | legacy text completion |
| `GET /v1/models` | list served models |
| `GET /health` | health check (used by the load balancer) |
| `GET /metrics` | Prometheus metrics |

Streaming works on all generation endpoints via `"stream": true`. For token counts while streaming, add
`"stream_options": {"include_usage": true}`; otherwise a streaming response omits usage.

### If you need auth that holds a real secret

The key above is a gate, not a secret (see [Notes](#notes-and-limitations)). To authenticate with a
value that never enters the template, let the **engine** check it: vLLM accepts `--api-key`, injected
from Secrets Manager as an ECS task secret, the same way `HF_TOKEN` is. Clients change nothing; they
already send `Authorization: Bearer`. Trade-off: rejection moves from the load balancer to the engine,
so unauthenticated requests reach the container. `/health` stays open, since vLLM authenticates only
`/v1*`.

**API Gateway is a poor fit here:** its integration timeout caps at **30 seconds**, and long or streamed
generations routinely exceed that. Measured here, a 760-token completion took ~17 s.

---

## Operating

### Watching it run

Every deployment creates a CloudWatch dashboard and two alarms. The URL is a stack output;
`endpoint_info.py` prints it:

```bash
python3 scripts/endpoint_info.py     # Dashboard  https://<region>.console.aws.amazon.com/...
```

Widgets are titled as questions (*Is it slow?*, *Are the engines up?*, *Did AWS give us the
instances?*). Alarms:
`<stack>-engines-unhealthy` and `<stack>-load-balancer-erroring`; set `alarmTopicArn` in config.yaml to
notify an SNS topic. Set `latencyAlarmSeconds` for a third, `<stack>-too-slow`, on p95; it has no
default because "too slow" depends on the caller.

Top rows come from the load balancer and Auto Scaling group. The bottom row comes from inside the
engines: a sidecar in every task scrapes vLLM's metrics and publishes queue depth, KV cache usage and
preemptions, which say *why* a fleet is slow. See *Watching a running fleet* and *Engine metrics* in
[docs/tuning.md](docs/tuning.md), which also covers overload (the container queues indefinitely; it
never returns "busy").

### What it costs

**While serving**, GPU instances dominate. Spot is typically 40–70% cheaper than on-demand.

**While idle**, with the GPU count at zero: load balancer (~$16–20/month), NAT gateway (~$32/month plus
data), storage for the image, and about $1.20/month for the four engine metrics
(custom metrics are billed per name, not per task).

**Autoscaling does not reliably reduce this**: a fleet that grew during a spike shrinks over 45–60
minutes (see [Notes](#notes-and-limitations)). To stop paying, scale to zero.

### Scale to zero without tearing down

Removes all GPU cost while keeping the stack, endpoint and configuration:

```yaml
# config.local.yaml
instanceCount: 0
maxInstanceCount: 0
```

Deploy. The empty fleet is then what the template says, so it survives later deploys. Instances are gone
about **5 minutes** after the deploy finishes (the termination hook drains each one first). Both values
must be zero; zeroing only one is rejected at synth, because autoscaling would otherwise hold the floor.

To come back, restore the counts and deploy again; allow several minutes for the model to load.

Do not scale to zero with the CLI instead. Anything set that way is drift: the next `cdk deploy`, even a
doc-only one, restores the template's counts and the fleet comes back. If you did, and it came back, see
*you scaled the service to zero and it came back* in [docs/troubleshooting.md](docs/troubleshooting.md).

### Full teardown

```bash
cd infra && cdk destroy
```

**Park the fleet first** (`instanceCount: 0`, `maxInstanceCount: 0`, deploy), then **wait until the
GPU instances are gone**, about five minutes (the check at the end of this section). Then destroy, about
ten minutes. Destroying while the instances were still draining left the ECS service in `DRAINING` for
25 minutes, paying for idle GPUs throughout.

If the destroy hangs on the subnets or the VPC, something outside the stack still holds a network
interface or security group in it. GuardDuty Runtime Monitoring is the usual cause; delete its VPC
endpoint and security group.

`cdk destroy` does not touch what `build_image.py` created: the ECR repository `gpu-llm-serving` (image
storage, the one that costs), the build bucket `gpu-llm-serving-build-<account>-<region>`, the CodeBuild
project and the role `GpuLlmServingCodeBuildRole`. Nor the `gpu-llm-serving/hf-token` secret if you made
one.

```bash
REGION=eu-west-2        # the region in your config
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws ecr delete-repository --repository-name gpu-llm-serving --force --region "$REGION"
aws s3 rb "s3://gpu-llm-serving-build-$ACCOUNT-$REGION" --force --region "$REGION"
aws codebuild delete-project --name gpu-llm-serving-build --region "$REGION"
aws logs delete-log-group --log-group-name /aws/codebuild/gpu-llm-serving-build --region "$REGION" 2>/dev/null
for pol in $(aws iam list-role-policies --role-name GpuLlmServingCodeBuildRole --query 'PolicyNames' --output text); do
  aws iam delete-role-policy --role-name GpuLlmServingCodeBuildRole --policy-name "$pol"; done
aws iam delete-role --role-name GpuLlmServingCodeBuildRole
aws secretsmanager delete-secret --secret-id gpu-llm-serving/hf-token --force-delete-without-recovery --region "$REGION" 2>/dev/null
```

**Always confirm no GPU instances survive**:

```bash
aws ec2 describe-instances --region "$REGION" \
  --filters Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[?starts_with(InstanceType, `g7e`)].[InstanceId,InstanceType]'
```

---

## Measurements

The shipped default, deployed into an empty region and loaded with 1,000-token **unique** prompts and
190 output tokens over `/v1/responses`:

| Instances | Concurrency | Input tok/s | Requests/sec | p95 | Errors |
|---|---|---|---|---|---|
| 1 | 128 | 16,075 | 17.2 | 7.42 s | not recorded |
| 6 | 768 | 95,645 | 102.1 | 8.38 s | 0.01% |
| 8 | 1024 | 132,772 | 141.8 | 7.60 s | 0.00% |
| 8, via CloudFront | 768 | 114,956 | 122.8 | 6.50 s | 0.01% |
| 8, direct to ALB | 768 | 113,926 | 121.7 | 6.67 s | 0.04% |
| 8 | 1920 | 178,681 | 190.8 | 11.96 s | 0.01% |

**CloudFront adds nothing measurable**; the two 768-concurrency rows are the same fleet on the same day.
Its one limit: a 40,000-token answer requested *without* streaming got CloudFront's 504 at 120 s; *with*
streaming it completed all 40,000 tokens in 252 s.

**At 1,920 concurrent, throughput stopped at the batch ceiling, not memory**: 244 of 256 batch slots in
use, KV cache at 20%, zero preemptions. See *Engine metrics* in [docs/tuning.md](docs/tuning.md).

**Fleet capacity is linear in instance count.** 6 instances delivered 99.2% of 6 × one; 8 delivered
103% of 8 ×. Measure one instance, divide your demand by it, round up.

**Autoscaling recovered the latency budget.** A 15-minute run at 768 concurrent scaled 6 → 8 mid-run
and averaged 111.7 rps at p95 7.11 s with 2 failures in 100,520 requests. A fixed 6 instances sat at p95
8.38 s.

Two caveats: **p99 leaves the budget long before p95** (13.10 s at 1024 concurrent while p95 read
7.60 s), and these are **unique-prompt** figures; with prefix cache hits the same hardware goes roughly
2× further.

**Two options measured after the table above, both large.** On the same fleet shape (eight
`g7e.2xlarge`, unique 1,000-token prompts, 768 concurrent): an EAGLE-3 speculator, one line in
`extraArgs`, raised throughput 24% and cut p95 from 6.33 s to 5.62 s; the NVFP4 checkpoint of the same
model raised it 28% and cut p95 to 5.06 s. Neither is the default: the speculator is tied to the model,
and NVFP4's output quality is unmeasured. *Speculative decoding* and *NVFP4* in
[docs/tuning.md](docs/tuning.md).

**Not measured**, so treat as estimates: tensor parallelism above 2 (four- and eight-GPU instances were
not obtainable); autoscaling timings on fleets other than 6 → 8; output quality of any quantisation
(every figure here is throughput); prompts longer than 4,000 tokens, and mixing long and short requests
on one engine; any model other than `Qwen/Qwen3-30B-A3B-Instruct-2507`, on which all of the above was
run.

---

## Layout

```
config.yaml               the only file you need to edit
config.local.yaml         optional, gitignored: overrides merged over config.yaml
AGENTS.md                 instructions for coding agents; CLAUDE.md points here
infra/
  app.py                  CDK entry point; validates config before synthesising
  cdk.json                CDK app command and feature flags
  hardware.py             instance catalog and all derived values (pure, tested)
  serving_stack.py        VPC, ECS, GPU capacity provider, ALB, CloudFront, service, dashboard
container/
  Dockerfile              vLLM base image plus the entrypoint
  serve                   entrypoint: translate config to engine flags
scripts/
  build_image.py          build and push to ECR, via CodeBuild
  endpoint_info.py        print the endpoint, key and a ready-to-paste request
  test_endpoint.py        smoke-test a deployed endpoint
  benchmark.py            concurrency sweep: req/s, tok/s, p50/p95/p99 per level
LICENSE                   MIT-0
docs/
  tuning.md               choosing an instance type and tuning the engine
  troubleshooting.md      symptom → cause → fix
tests/
  test_hardware.py        instance catalog, derived values, config validation
  test_template.py        properties of the synthesised template a deployment depends on
  test_app.py             the API key file side effect in app.py
```

Run the tests with `python3 -m pytest tests/ -q`. They need no AWS credentials and cover failures that
would otherwise take a 20-minute deployment to surface.

---

## Notes and limitations

- **The endpoint is public HTTPS, gated by the API key alone.** CloudFront fronts an internal load
  balancer, so there is no plaintext hop. Anyone with the key can call it from anywhere; see
  [Access](#access) for WAF restriction and the 120 s non-streamed limit.
- **The API key is a gate, not an authorization layer, and it is not confidential.** An ALB listener
  rule cannot resolve a Secrets Manager reference, so it holds a literal value. Anyone who can read the
  stack sees the key in the CloudFormation template and the stack outputs (`ApiKeyValue`). One shared secret, no
  per-caller identity, rotation or revocation. Keeps unauthenticated traffic off the model; not
  sufficient for sensitive data. See [that section](#if-you-need-auth-that-holds-a-real-secret).
- **The key is generated once and persisted to `config.local.yaml`** on first deploy, so redeploys
  reuse it. Set `apiKey` there to control it. It is *not* generated during synth, which runs on every
  deploy and would rotate the key each time.
- **Autoscaling is ON in the shipped config** (`instanceCount: 16`, `maxInstanceCount: 24`) and is
  **slower than you expect** (measured on a 6 → 8 fleet). Set `maxInstanceCount` equal to `instanceCount` for a
  fixed-size fleet. Scale-out took **~11 minutes** to usable capacity (metric lag, a 3-datapoint alarm,
  then weight loading); scale-in runs **one step at a time**, so 8 → 6 took 45–60 minutes. The instance
  **outlives its task by ~15 minutes**, because the capacity provider evaluates scale-in separately.
  Size `instanceCount` for steady state; scale-out is insurance, not savings. See *Sizing a fleet* in
  [docs/tuning.md](docs/tuning.md).
- **Spot instances can be reclaimed** with two minutes' notice. Fine for evaluation, not for production
  without extra handling. See [Spot and instance protection](#spot-and-instance-protection).
- **The engine version is pinned** in `container/Dockerfile`. Flag names and API surface change between
  releases; upgrade on purpose and re-run `scripts/test_endpoint.py`.
- **Gated models need `hfTokenSecretName`.** Without it the task fails with a 401 while pulling; with a
  token whose account has not accepted the licence, a 403.
- **g7e only.** The instance catalog in `infra/hardware.py` knows the six g7e sizes and rejects anything
  else at synth. Another GPU family means adding its entries there; nothing else assumes g7e.
