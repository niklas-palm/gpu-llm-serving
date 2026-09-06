"""One GPU serving cluster: VPC, ECS with a GPU capacity provider, an internal ALB behind CloudFront,
and a vLLM service.

One stack. It is small enough that splitting it would add navigation cost without
buying anything, and a single `cdk destroy` then removes everything. `__init__` builds it in one pass;
the banner comments mark the sections.

WHY EC2 AND NOT FARGATE
    Fargate has no GPU support. GPU containers on ECS must run on EC2, so the cluster needs an Auto
    Scaling Group and an ECS capacity provider.

HOW REQUESTS ARE AUTHENTICATED
    An ALB cannot validate API keys natively, so the listener's default action is a hard 403 and
    traffic reaches the target group only if the request carries `Authorization: Bearer <key>`, the
    header every OpenAI-compatible client and gateway already sends. What
    that key is and is not is explained where it is built, and in README.md.

HOW THE ENDPOINT GETS HTTPS WITHOUT A DOMAIN
    CloudFront, with its own *.cloudfront.net name and certificate, reaching the load balancer through
    a VPC origin. The load balancer is internal and never sees the internet; CloudFront is the only way
    in. Nothing to own, nothing to validate, nothing to renew.
"""

from __future__ import annotations

import re
import sys

from aws_cdk import CfnOutput, Duration, RemovalPolicy, SecretValue, Stack
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sns as sns
from constructs import Construct

from hardware import (ROOT_VOLUME_GIB, ROOT_VOLUME_IOPS, ROOT_VOLUME_THROUGHPUT_MBPS,
                      ConfigError, _flag, _given, _list, _num, bytes_per_param_for, get_instance,
                      gpus_per_replica, memory_pressure_warning, model_bytes, resolve_topology,
                      validate_tuning)

CONTAINER_PORT = 8080

# How long CloudFront waits for the first byte of a response, and then between bytes. 120 s is the
# most the account quota allows without a support request. A STREAMED answer never gets near it -
# tokens arrive continuously. A non-streamed answer has to finish inside it, whole.
CLOUDFRONT_READ_TIMEOUT_S = 120

# Where the container keeps downloaded weights, and the host directory backing it. Must match the
# path used by container/serve.
CONTAINER_MODEL_PATH = "/opt/model"

# The metrics sidecar: the AWS Distro for OpenTelemetry collector, run unmodified from the public
# image with its configuration passed inline. Pinned, because a floating tag would change what the
# dashboard reads without any change in this repository.
METRICS_SIDECAR_IMAGE = "public.ecr.aws/aws-observability/aws-otel-collector:v0.43.3"
METRICS_SIDECAR_MEMORY_MIB = 256
# The engine metrics the dashboard reads. Four of the ~86 families the engine exposes; the rest are
# either derivable from these, duplicated by the load balancer, or histograms CloudWatch cannot turn
# into percentiles. Custom metrics are billed per name, so the shortlist is also the bill.
ENGINE_METRICS = (
    "vllm:num_requests_waiting",   # queued but not running - saturation, before latency shows it
    "vllm:num_requests_running",   # sequences in the batch, against maxNumSeqs
    "vllm:kv_cache_usage_perc",    # 0..1; near 1 means preemption is next
    "vllm:num_preemptions_total",  # counter; anything above zero is the engine going backwards
)
HOST_MODEL_CACHE = "/opt/modelcache"


class ServingStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: dict, **kw) -> None:
        super().__init__(scope, cid, **kw)

        inst = get_instance(cfg["instanceType"])
        # Resolved ONCE, before anything reads it. It is used in three places, and as a bare
        # cfg.get("useSpot") a quoted "false" from YAML was truthy at every one of them - silently
        # turning spot ON for someone who had written it off.
        use_spot = _flag(cfg.get("useSpot"), "useSpot")

        # The model's parameter count is not knowable at synth time without a network call, so it is
        # an ESTIMATE, defaulting to a size that suits a single 96 GiB GPU. `estimatedParamsBillions`
        # in config.yaml overrides it, and getting it wrong matters: it is what the tensor-parallel
        # degree and the memory-pressure check are derived from, so a 70B model left at the default
        # derives TP=1, synthesises cleanly, deploys, and then runs out of VRAM while loading.
        # minimum, because a finite-but-nonsensical size passed silently: a negative parameter count
        # produced negative weight bytes and derived a happy TP=1.
        est_params_b = _num(_given(cfg.get("estimatedParamsBillions"), 30),
                            "estimatedParamsBillions", float, minimum=0.001)
        # Considers the MODEL ID as well as `quantization`: a publisher's official fp8 build is
        # already quantised on disk and correctly leaves `quantization` empty, so keying off that
        # alone doubles the estimated weight size and misreads a correct configuration.
        bytes_per_param = bytes_per_param_for(cfg["modelId"], cfg.get("quantization"))
        est_weight_bytes = model_bytes(est_params_b, bytes_per_param)

        # Resolve tensorParallel and replicas together - they are one decision about how the
        # instance's GPUs are divided into engines, and the default is one engine per GPU so that a
        # multi-GPU instance uses all of them without the user computing anything.
        tuning = resolve_topology(inst, validate_tuning(inst, cfg.get("tuning")),
                                  est_weight_bytes)

        # Print the assumption, on stderr, because a silent wrong estimate is the failure above.
        print(f"assuming ~{est_params_b:g}B parameters at {bytes_per_param:g} byte(s) each = "
              f"{est_weight_bytes / 1024 ** 3:.1f} GiB of weights "
              f"-> tensorParallel={tuning['tensorParallel']}, replicas={tuning['replicas']}.\n"
              f"  Set estimatedParamsBillions in config.yaml if that is not your model's size.",
              file=sys.stderr)

        warning = memory_pressure_warning(est_weight_bytes // tuning["tensorParallel"], tuning,
                                          quantised=bytes_per_param < 2.0)
        if warning:
            # stderr, because `cdk synth > template.yaml` is normal and stdout carries the template.
            print(f"\nWarning: {warning}\n", file=sys.stderr)

        # ------------------------------------------------------------------ networking
        # Always its own VPC. Not every region has a default VPC - us-east-2 does not - so relying on
        # one makes the stack undeployable in exactly the regions most likely to have spare GPU
        # capacity.
        #
        # By default it spans up to FOUR AZs, and that is a capacity decision rather than a resilience
        # one: GPU capacity is allocated per availability zone and the scarce shapes are unavailable
        # in most of them at any moment, so more zones means more places the ASG can look.
        #
        # But an AZ that does not OFFER the instance type is worse than useless - the ASG will pick it
        # and the launch fails with `Unsupported`, which is permanent rather than transient, so it does
        # not resolve by retrying. Not hypothetical: us-east-2 has three AZs and offers g7e in two.
        # Set `availabilityZones` to the offering zones (config.yaml has the command to find them) and
        # the VPC, and therefore the ASG, is confined to them.
        configured_azs = [z.strip() for z in _list(cfg.get("availabilityZones"), "availabilityZones")
                          if z.strip()]
        # DISTINCT zones, not entries. `["us-west-2a", "us-west-2a"]` is an easy thing to end up with
        # while editing a list, and counting entries passed it - then CloudFormation rejected two
        # subnets in one AZ minutes into the deploy, which is exactly what this check exists to
        # forestall.
        if configured_azs and len(set(configured_azs)) < 2:
            raise ConfigError(
                f"availabilityZones needs at least two DIFFERENT zones (got {configured_azs}).\n"
                "  A load balancer requires subnets in two, even when the GPU instances only ever\n"
                "  land in one of them."
            )
        vpc_kwargs = dict(
            ip_addresses=ec2.IpAddresses.cidr("10.30.0.0/16"),
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC,
                                        cidr_mask=20),
                ec2.SubnetConfiguration(name="private", cidr_mask=20,
                                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            ],
        )
        if configured_azs:
            vpc_kwargs["availability_zones"] = configured_azs
        else:
            vpc_kwargs["max_azs"] = 4
        vpc = ec2.Vpc(self, "Vpc", **vpc_kwargs)

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        # ------------------------------------------------------- GPU capacity provider
        instance_role = iam.Role(
            self, "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"),
                # Lets you open a shell on the host without SSH or a bastion, which is how you
                # inspect the NVIDIA driver or the ECS agent log when something is wrong.
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
            ],
        )

        # An EXPLICIT launch template rather than letting the ASG construct create one.
        #
        # Two reasons. First, CDK's AutoScalingGroup creates an AWS::AutoScaling::LaunchConfiguration
        # by default, and launch configurations do not support MixedInstancesPolicy - which is the
        # only way to request spot. Second, an explicit template means the ASG's exact configuration
        # is visible here instead of depending on the construct's internal child naming, which has
        # changed between CDK versions.
        gpu_user_data = ec2.UserData.for_linux()
        if use_spot:
            # Drain the task when a spot reclaim notice arrives, instead of letting the instance
            # vanish mid-request. Without this the ECS agent does not deregister the target, so the
            # load balancer keeps sending traffic to a dead task until the health check fails it -
            # interval x threshold, up to ~150 s of 502s. Draining deregisters immediately and lets
            # in-flight requests finish. Set here rather than via asg.add_user_data() because with an
            # externally supplied launch template the ASG does not own the user data.
            gpu_user_data.add_commands(
                "echo ECS_ENABLE_SPOT_INSTANCE_DRAINING=true >> /etc/ecs/ecs.config")

        launch_template = ec2.LaunchTemplate(
            self, "GpuLaunchTemplate",
            instance_type=ec2.InstanceType(inst.name),
            # Amazon Linux 2023, NOT Amazon Linux 2 - a hard requirement. The AL2 GPU AMI's NVIDIA
            # driver is too old for this GPU generation, and it fails silently: the instance boots,
            # passes every health check, and registers with an EMPTY GPU list, so tasks sit in
            # PROVISIONING forever with no error anywhere. Resolved from an SSM parameter, so no
            # custom AMI and no per-region id. See docs/troubleshooting.md for the diagnostic.
            machine_image=ecs.EcsOptimizedImage.amazon_linux2023(ecs.AmiHardwareType.GPU),
            role=instance_role,
            security_group=ec2.SecurityGroup(self, "InstanceSg", vpc=vpc,
                                             description="GPU serving instances"),
            user_data=gpu_user_data,
            require_imdsv2=True,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/xvda",
                volume=ec2.BlockDeviceVolume.ebs(
                    # Sizing and throughput are explained on the constants in hardware.py.
                    ROOT_VOLUME_GIB,
                    volume_type=ec2.EbsDeviceVolumeType.GP3,
                    throughput=ROOT_VOLUME_THROUGHPUT_MBPS,
                    iops=ROOT_VOLUME_IOPS,
                    delete_on_termination=True),
            )],
        )

        # Fleet size, resolved once. `maxInstanceCount` doubles as the autoscaling on/off switch:
        # anything above `instanceCount` creates a scaling policy, equal or unset pins the fleet.
        # `_given`, not `or`: a key written as `instanceCount:` with nothing after it parses to None,
        # which satisfies .get() and then fails on int(None), so the default has to cover a blank
        # value - but 0 is falsy and must NOT be swallowed by it.
        # ZERO is allowed and meaningful: it is the durable way to stop paying while keeping the
        # stack, endpoint, DNS and certificate. Scaling to zero with the CLI is drift that the next
        # `cdk deploy` reverts - MinSize, DesiredCount and the scalable floor are all re-established
        # from the template - so config is the only place a zero fleet survives a deploy.
        instance_count = _num(_given(cfg.get("instanceCount"), 1), "instanceCount", minimum=0)
        configured_max = _num(_given(cfg.get("maxInstanceCount"), instance_count),
                              "maxInstanceCount", minimum=0)
        if instance_count == 0 and configured_max > 0:
            # A parked fleet has to be a FIXED-size fleet of zero, and this combination is neither
            # parked nor autoscaling. It leaves MaxSize above zero and DesiredCapacity absent, so the
            # deploy never lowers ASG capacity and the shutdown falls back on the ~15-minute managed
            # scale-in below. Worse, it cannot come back: with no tasks the target group publishes no
            # ALBRequestCountPerTarget datapoints at all, and target tracking does not scale out on a
            # missing metric - nor on a metric below its target, which is the other half of the trap.
            raise ConfigError(
                f"instanceCount is 0 but maxInstanceCount is {configured_max}.\n"
                "  Parking the deployment means a fixed-size fleet of zero, so set BOTH to 0:\n"
                "      instanceCount: 0\n"
                "      maxInstanceCount: 0\n"
                "  Left as it is, autoscaling owns a fleet it can never grow: with no tasks running\n"
                "  there is no request-rate metric to scale on, and the empty fleet would not be\n"
                "  durable either - MaxSize stays above zero and the deploy never lowers capacity."
            )
        if configured_max < instance_count:
            # Otherwise CDK raises a bare jsii RuntimeError about ASG bounds, which reads as an
            # internal fault rather than as the config mistake it is.
            raise ConfigError(
                f"maxInstanceCount ({configured_max}) is below instanceCount ({instance_count}).\n"
                "  maxInstanceCount is the autoscaling CEILING, so it has to be at least the floor.\n"
                "  Set them equal for a fixed-size fleet, or raise it to allow scale-out."
            )
        max_instances = configured_max
        autoscaling_enabled = max_instances > instance_count

        asg = autoscaling.AutoScalingGroup(
            self, "GpuAsg",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            launch_template=launch_template,
            # MinSize IS the floor, and that solves two problems at once.
            #
            # An ASG created without DesiredCapacity takes MinSize as its initial desired capacity. So
            # with min_capacity=instanceCount a FIRST deploy comes up at the right size with nothing
            # else needing to intervene - where min_capacity=0 plus an omitted DesiredCapacity would
            # come up with zero instances and wait for something to scale it, which for a brand-new
            # scalable target is not documented to happen (AWS documents min/max enforcement for
            # UPDATING an existing scalable target, not for registering one).
            #
            # And because MinSize is a floor rather than a target, a later `cdk deploy` does NOT pull a
            # scaled-out fleet back down: CloudFormation leaves DesiredCapacity alone when it is absent
            # from the template. That is the important half - resetting desired capacity terminates
            # instances that still hold tasks and tens of GiB of loaded weights.
            #
            # `instanceCount` is documented as the minimum, so a floor is also what it should mean.
            # Scale-to-zero is a deliberate manual operation that lowers MinSize explicitly.
            min_capacity=instance_count,
            max_capacity=max_instances,
            # Declared only when nothing else manages the fleet, so a hand-scaled ASG returns to the
            # configured size on the next deploy. (CDK warns about that reset; it is the intent.)
            **({} if autoscaling_enabled else {"desired_capacity": instance_count}),
            # An ASG publishes NOTHING about its own size unless group metrics are enabled - the
            # `GroupInServiceInstances` / `GroupDesiredCapacity` metrics simply do not exist until
            # then, so a dashboard widget built on them renders an empty graph and looks like a broken
            # fleet rather than a missing setting. Enabling them is free, and unconditional because
            # they are the only historical record of how big the fleet was at a given moment; that is
            # the first thing anyone asks when reading a latency spike after the fact.
            group_metrics=[autoscaling.GroupMetrics(
                autoscaling.GroupMetric.IN_SERVICE_INSTANCES,
                autoscaling.GroupMetric.DESIRED_CAPACITY,
            )],
        )

        if use_spot:
            # Spot is a separate capacity pool from on-demand and often has availability when
            # on-demand does not. It also has a SEPARATE, much smaller quota - see README.md.
            #
            # `capacity-optimized-prioritized` rather than `capacity-optimized`: the latter ignores
            # the override order entirely and launches from whichever pool has the most spare
            # capacity, which silently gives you a different instance type than you asked for.
            cfn_asg = asg.node.default_child
            cfn_asg.add_property_override("MixedInstancesPolicy", {
                "LaunchTemplate": {
                    "LaunchTemplateSpecification": {
                        "LaunchTemplateId": launch_template.launch_template_id,
                        "Version": launch_template.latest_version_number,
                    },
                    "Overrides": [{"InstanceType": inst.name}],
                },
                "InstancesDistribution": {
                    "OnDemandAllocationStrategy": "prioritized",
                    "OnDemandBaseCapacity": 0,
                    "OnDemandPercentageAboveBaseCapacity": 0,
                    "SpotAllocationStrategy": "capacity-optimized-prioritized",
                },
            })
            # The plain LaunchTemplate property and MixedInstancesPolicy are mutually exclusive;
            # CloudFormation rejects a template carrying both.
            cfn_asg.add_property_deletion_override("LaunchTemplate")
            # Start replacing an instance on the rebalance recommendation, which arrives BEFORE the
            # two-minute reclaim notice, so the replacement has a head start on loading weights.
            cfn_asg.add_property_override("CapacityRebalance", True)
            # Draining is enabled through the launch template's user data (see `gpu_user_data`),
            # because with an externally supplied launch template the ASG does not own user data.

        capacity_provider = ecs.AsgCapacityProvider(
            self, "GpuCapacity", auto_scaling_group=asg,
            # Left OFF. Enabled, it stops the ASG terminating an instance that still has
            # tasks - which also means a deliberate scale to zero waits on it. The measured ~3 minute
            # teardown depends on this being off.
            enable_managed_termination_protection=False,
        )
        # Spot draining is enabled through the launch template's user data rather than the construct's
        # `spot_instance_draining`: that property works by writing to the ASG's user data, which it
        # cannot do when the launch template is supplied externally (verified - it produces no
        # ManagedDraining in the template here). See `gpu_user_data` above for the real mechanism.
        cluster.add_asg_capacity_provider(capacity_provider)

        # ------------------------------------------------------------------ API key
        #
        # ONE value, for the listener rule and the stored copy alike - two values here would mean
        # retrieving the key the documented way locked you out.
        #
        # It must come from config rather than being generated here, because this runs at SYNTH time:
        # a fresh value would be a NEW key on every `cdk deploy`. app.py generates one once and
        # persists it to config.local.yaml so redeploys are stable.
        #
        # BE CLEAR ABOUT WHAT THIS IS. An ALB listener rule is evaluated by the load balancer, which
        # cannot resolve a Secrets Manager reference, so the condition needs a LITERAL. The key
        # therefore appears in the template, the stack outputs and the stored secret alike - it is not
        # confidential from anyone who can read the stack. A lightweight gate that keeps unauthenticated
        # traffic off the model, and NOT an authorization layer. README.md has the upgrade path.
        key_value = str(cfg.get("apiKey") or "").strip()
        if not key_value:
            raise ConfigError(
                "apiKey is not set.\n"
                "  Deploy through `cdk deploy` from the infra/ directory and one is generated and\n"
                "  saved to config.local.yaml automatically. If you are calling the stack directly,\n"
                "  pass apiKey in the config - it cannot be generated here, because synth runs on\n"
                "  every deploy and a fresh value would rotate the key each time."
            )

        # A convenience copy, so callers have somewhere conventional to read it from - NOT a
        # confidentiality boundary, for the reason above. `unsafe_plain_text` is the honest API here:
        # the value is in the template. A generator would produce a SECOND value.
        api_key = secretsmanager.Secret(
            self, "ApiKey",
            description=("Inference endpoint API key. Convenience copy - the same value is in the "
                         "stack template and outputs, so treat it as non-confidential."),
            secret_string_value=SecretValue.unsafe_plain_text(key_value),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ------------------------------------------------------------------ load balancer
        #
        # INTERNAL, always. Nothing here faces the internet: CloudFront (below) is the only way in, and
        # it reaches this load balancer through a VPC origin - a network interface CloudFront creates
        # inside the private subnets. The hop from CloudFront to here therefore never leaves the AWS
        # network, so a plain HTTP listener is acceptable when a *public* HTTP listener would
        # not be: the API key is a request header, and a header is exactly as private as the transport
        # under it.
        alb = elbv2.ApplicationLoadBalancer(
            self, "Alb", vpc=vpc, internet_facing=False,
            # The ALB's default idle timeout is 60 seconds, and for a NON-streaming generation the
            # engine sends nothing until the whole response is finished - so the idle timer covers the
            # entire generation, not the gaps within it. 300 s here so this hop is never the limit;
            # CloudFront's read timeout (CLOUDFRONT_READ_TIMEOUT_S) is the one that binds.
            idle_timeout=Duration.seconds(300),
        )

        listener = alb.add_listener(
            "Endpoint", port=80, protocol=elbv2.ApplicationProtocol.HTTP,
            # `open=False` so CDK does not add an unconditional 0.0.0.0/0 ingress rule.
            open=False,
            # Anything without a valid key gets a flat 403 and never reaches the model.
            default_action=elbv2.ListenerAction.fixed_response(
                403, content_type="application/json",
                message_body='{"error":"missing or invalid Authorization: Bearer <key> header"}'),
        )
        # Who may reach the load balancer: CloudFront, by its managed prefix list of origin-facing
        # addresses. NOT the VPC CIDR - that was tried first and CloudFront timed out on every connect.
        # A VPC origin's traffic enters through an interface CloudFront places in the private subnets,
        # but the packets carry CloudFront's own source addresses, not the interface's private IP, so
        # a rule on the VPC range never matches. The prefix list is looked up by name so this works in
        # any region (its id differs per region). The result is cached in cdk.context.json, which is
        # gitignored, so each clone performs the lookup once on its first synth.
        # No apostrophes in rule descriptions: EC2 rejects them at deploy time, and synth does not check.
        cloudfront_origins = ec2.PrefixList.from_lookup(
            self, "CloudFrontOriginFacing",
            prefix_list_name="com.amazonaws.global.cloudfront.origin-facing")
        alb.connections.allow_from(ec2.Peer.prefix_list(cloudfront_origins.prefix_list_id),
                                   ec2.Port.tcp(80), "CloudFront origin-facing addresses")

        target_group = elbv2.ApplicationTargetGroup(
            self, "Targets",
            vpc=vpc, port=CONTAINER_PORT, protocol=elbv2.ApplicationProtocol.HTTP,
            # `ip` is required for awsvpc networking, where each task has its own ENI and address.
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                path="/health",
                # Loading a large model takes minutes, during which the container is up but not yet
                # answering. A generous threshold stops the deployment being marked failed while it
                # is legitimately still starting.
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                healthy_threshold_count=2,
                unhealthy_threshold_count=5,
            ),
            # Long enough for an in-flight generation to finish. The default 30 s (and the container's
            # 30 s stop timeout below) killed any request still generating when a task was replaced -
            # on every scale-in and every deployment - and a non-streaming completion routinely runs
            # longer than that. Draining holds the target open until the request completes.
            deregistration_delay=Duration.seconds(180),
        )

        listener.add_action(
            "AuthedForward", priority=10,
            conditions=[elbv2.ListenerCondition.http_header("Authorization",
                                                            [f"Bearer {key_value}"])],
            action=elbv2.ListenerAction.forward([target_group]),
        )

        # ------------------------------------------------------------------ CloudFront
        #
        # The public HTTPS endpoint, with nothing to own: CloudFront's *.cloudfront.net name and
        # certificate. It is a pass-through, not a cache - every setting below exists to make it get
        # out of the way of an API that streams:
        #
        #   caching disabled            responses are never cached, and POST is never cacheable anyway
        #   all viewer headers forwarded  Authorization must reach the load balancer's rule (Host is
        #                               dropped, because the origin has its own)
        #   all methods                 POST
        #   compression off             a compressed stream is a buffered stream
        #   HTTPS only                  a plain-http call gets a 403 rather than a silent redirect
        #                               that would turn a POST into a GET
        #
        # Streaming works because CloudFront forwards bytes as the origin sends them, and its read
        # timeout resets on every byte. The one limit it imposes is on answers that do NOT stream: the
        # whole response must arrive within CLOUDFRONT_READ_TIMEOUT_S or the caller gets a 504 while
        # the engine is still happily generating. docs/tuning.md covers what that means for sizing.
        distribution = cloudfront.Distribution(
            self, "Cdn",
            comment=f"{self.stack_name}: HTTPS front door for the inference endpoint",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.VpcOrigin.with_application_load_balancer(
                    alb,
                    protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
                    http_port=80,
                    read_timeout=Duration.seconds(CLOUDFRONT_READ_TIMEOUT_S),
                    # Reuse connections to the load balancer under load. Must stay below the ALB's
                    # 300 s idle timeout, or CloudFront reuses a connection the ALB has closed.
                    keepalive_timeout=Duration.seconds(60),
                ),
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                compress=False,
            ),
            http_version=cloudfront.HttpVersion.HTTP2_AND_3,
            # CloudFront otherwise caches error responses for 10 s. A 503 while a task restarts would
            # be replayed to every caller for those 10 s, including ones the fleet could have served.
            error_responses=[cloudfront.ErrorResponse(http_status=code, ttl=Duration.seconds(0))
                             for code in (500, 502, 503, 504)],
        )
        endpoint = f"https://{distribution.distribution_domain_name}"

        # ------------------------------------------------------------------ the service
        log_group = logs.LogGroup(self, "Logs", retention=logs.RetentionDays.ONE_WEEK,
                                 removal_policy=RemovalPolicy.DESTROY)

        task_def = ecs.Ec2TaskDefinition(self, "TaskDef", network_mode=ecs.NetworkMode.AWS_VPC)

        # A shared HOST directory for the weights cache. This is required, not an optimisation.
        #
        # Without it, each container start writes its own copy of the weights into the container's
        # writable layer, and a stopped container keeps that layer. A task that crash-loops therefore
        # accumulates a full copy of the model per attempt: a dozen restarts of a ~57 GiB model
        # consumed close to 500 GB of disk, at which point nothing can start and the failure presents
        # as a model load that never completes rather than as a disk error.
        #
        # Mounting a host path instead means one copy on the instance, reused by every task. That
        # also removes download time from every subsequent start, which is minutes per restart.
        task_def.add_volume(name="modelcache",
                            host=ecs.Host(source_path=HOST_MODEL_CACHE))

        # Optional Hugging Face token for gated models, read from a Secrets Manager secret the user
        # created (README, "Gated models"). Passed as an ECS secret, so it reaches the engine as HF_TOKEN
        # and never appears in the template, the outputs or the logs. Without it, public models work and
        # gated ones fail with a 401 while pulling.
        hf_secret_name = str(_given(cfg.get("hfTokenSecretName"), "")).strip()
        container_secrets = {}
        if hf_secret_name:
            hf_secret = secretsmanager.Secret.from_secret_name_v2(self, "HfToken", hf_secret_name)
            container_secrets["HF_TOKEN"] = ecs.Secret.from_secrets_manager(hf_secret)

        env = {
            "MODEL_ID": cfg["modelId"],
            "PORT": str(CONTAINER_PORT),
            "TENSOR_PARALLEL": str(tuning["tensorParallel"]),
            # 0 means "omit the flag and let the engine choose" for both of these.
            "MAX_MODEL_LEN": str(tuning["maxModelLen"]),
            "MAX_NUM_BATCHED_TOKENS": str(tuning["maxNumBatchedTokens"]),
            "MAX_NUM_SEQS": str(tuning["maxNumSeqs"]),
            "GPU_MEMORY_UTILIZATION": str(tuning["gpuMemoryUtilization"]),
            "ENABLE_PREFIX_CACHING": "true" if tuning["enablePrefixCaching"] else "false",
            "KV_CACHE_DTYPE": tuning["kvCacheDtype"],
            "ENABLE_EXPERT_PARALLEL": "true" if tuning["enableExpertParallel"] else "false",
        }
        if cfg.get("quantization"):
            env["QUANTIZATION"] = cfg["quantization"]
        # Escape hatch for engine flags this project does not model - container/serve appends these
        # verbatim to `vllm serve`. Plumbed because the docs point at it as *the* way to try an
        # unmodelled flag (data parallelism, for one) and to diagnose a reluctant engine, and without
        # this there was nowhere to actually put it. Unvalidated by design: anything here bypasses the
        # checks in validate_tuning, which is the point, and the risk.
        if cfg.get("extraArgs"):
            env["EXTRA_ARGS"] = str(cfg["extraArgs"])

        container = task_def.add_container(
            "vllm",
            image=self._container_image(cfg["image"]),
            # Derived from HOST RAM, not GPU count - see hardware.py for why that distinction
            # matters. ECS reserves the whole amount, so this also caps replicas per instance.
            memory_limit_mib=inst.container_memory_mib // tuning["replicas"],
            gpu_count=gpus_per_replica(inst, tuning),
            environment=env,
            secrets=container_secrets or None,
            logging=ecs.LogDrivers.aws_logs(stream_prefix="vllm", log_group=log_group),
            port_mappings=[ecs.PortMapping(container_port=CONTAINER_PORT)],
            # /dev/shm, set UNCONDITIONALLY - do not make this depend on tensorParallel. The per-GPU
            # workers pass tensors through POSIX shared memory, and Docker's 64 MiB default kills any
            # degree above 1 seconds after start. TP=1 never touches that path, so a conditional value
            # looks fine until someone raises the degree. See docs/troubleshooting.md.
            linux_parameters=ecs.LinuxParameters(self, "Linux", shared_memory_size=8192),
            # Paired with deregistration_delay above: ECS sends SIGTERM and then waits this long
            # before SIGKILL. The default is 30 s, which cuts off a generation still in progress.
            stop_timeout=Duration.seconds(120),
        )
        # One copy of the weights per instance, shared by every task and surviving restarts.
        container.add_mount_points(ecs.MountPoint(container_path=CONTAINER_MODEL_PATH,
                                                 source_volume="modelcache",
                                                 read_only=False))

        # ------------------------------------------------------------------ engine metrics
        # The load balancer can say THAT the fleet is slow; only the engine can say WHY. It publishes
        # Prometheus metrics on /metrics, and this sidecar scrapes them over localhost (awsvpc puts
        # both containers in one network namespace) and writes them to CloudWatch as embedded-metric-
        # format log records, which CloudWatch turns into metrics under `<stack>/Engine`.
        #
        # No dimensions. CloudWatch then aggregates every engine's samples per minute, so
        # `Maximum` is the worst engine and `Average` the typical one - which is what the dashboard
        # needs - and the bill is four metrics regardless of fleet size. A per-task dimension would
        # let you name the sick engine, at a cost that grows with the fleet; the task's own logs in
        # the log group already serve that purpose.
        engine_metrics_namespace = f"{self.stack_name}/Engine"
        collector_config = f"""
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: engine
          scrape_interval: 30s
          static_configs:
            - targets: ["localhost:{CONTAINER_PORT}"]
processors:
  filter/shortlist:
    metrics:
      include:
        match_type: strict
        metric_names: {list(ENGINE_METRICS)}
exporters:
  awsemf:
    region: {self.region}
    namespace: {engine_metrics_namespace}
    log_group_name: {log_group.log_group_name}
    log_stream_name: engine-metrics
    dimension_rollup_option: NoDimensionRollup
    metric_declarations:
      - dimensions: [[]]
        metric_name_selectors: [".*"]
service:
  pipelines:
    metrics:
      receivers: [prometheus]
      processors: [filter/shortlist]
      exporters: [awsemf]
"""
        task_def.add_container(
            "metrics",
            image=ecs.ContainerImage.from_registry(METRICS_SIDECAR_IMAGE),
            # Taken from the 40% of host memory the engine does not reserve, so the engine's own
            # limit - and the replicas-per-instance arithmetic built on it - is unchanged.
            memory_limit_mib=METRICS_SIDECAR_MEMORY_MIB,
            # Not essential: if the collector dies the engine keeps serving and the dashboard's engine
            # row goes blank. The alternative - a metrics bug taking down inference - is the wrong trade.
            essential=False,
            environment={"AOT_CONFIG_CONTENT": collector_config},
            logging=ecs.LogDrivers.aws_logs(stream_prefix="metrics", log_group=log_group),
        )
        # What the collector needs: to write metric records into this stack's log group, and for
        # CloudWatch to accept them as metrics in this stack's namespace. Nothing wider.
        task_def.add_to_task_role_policy(iam.PolicyStatement(
            actions=["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
            resources=[log_group.log_group_arn, log_group.log_group_arn + ":*"]))
        task_def.add_to_task_role_policy(iam.PolicyStatement(
            actions=["cloudwatch:PutMetricData"], resources=["*"],
            conditions={"StringEquals": {"cloudwatch:namespace": engine_metrics_namespace}}))

        service = ecs.Ec2Service(
            self, "Service",
            cluster=cluster,
            task_definition=task_def,
            # With awsvpc each task gets its own ENI and IP address, so replicas can all bind the
            # same container port on one instance and the ALB round-robins across them.
            #
            # ALWAYS declared, including with autoscaling on, and that is a deliberate trade.
            #
            # Omitting it means CloudFormation omits DesiredCount, and ECS then defaults a new service
            # to ONE task - so a first deploy would serve from a single task on a full fleet of
            # instances, waiting for something to raise it. Registering a scalable target with a higher
            # MinCapacity is not documented to do that on creation (only on update of an existing one),
            # so the initial size has to be stated here.
            #
            # The cost: a later `cdk deploy` resets a scaled-out service back to
            # this number, and the scaling policy then takes ~11 minutes to climb again. The deploy
            # terminates nothing directly - the ASG above keeps its instances because DesiredCapacity is
            # absent from the template - but managed termination protection is DISABLED, so the capacity
            # provider notices the surplus and scales in about 15 minutes later, and the warm
            # /opt/modelcache on those instances goes with them. Still the milder half of the problem it
            # replaces, where the deploy terminated instances immediately and mid-request.
            desired_count=instance_count * tuning["replicas"],
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(capacity_provider=capacity_provider.capacity_provider_name,
                                             weight=1)],
            min_healthy_percent=0,
            # Circuit breaker OFF. Enabled, it reverts to the previous task definition
            # after repeated start failures - and on a single-service GPU deployment that revision is
            # frequently also broken, so the two contend for the same GPU and the service never
            # converges. What you give up: a bad deploy stays broken until you notice. Right when an
            # operator is watching and GPUs are scarce; turn it on for an unattended fleet with spare
            # capacity. See docs/troubleshooting.md.
            circuit_breaker=None,
            health_check_grace_period=Duration.minutes(30),
        )
        service.attach_to_application_target_group(target_group)

        # ------------------------------------------------------------------ autoscaling
        #
        # Created only when maxInstanceCount exceeds instanceCount. The shipped config does exceed it
        # (6 -> 8), so autoscaling is ON by default; set them equal for a fixed-size fleet.
        #
        # Either way it is burst absorption, NOT right-sizing: a new task loads tens of GiB of weights,
        # so it is minutes from scale-out decision to serving traffic. Nothing that slow can protect a
        # seconds-scale latency budget, and a fleet sized on the assumption that it can will breach the
        # budget for the whole cold start. The minimum has to cover steady state with headroom.
        #
        # RequestCountPerTarget rather than GPU utilisation (which sits near 100% while latency is
        # still fine, so it carries no SLA information) or response-time p95 (direct but lagging - it
        # only rises once requests are already slow). Request count leads both and needs no custom
        # metric. Engine queue depth would be better still, but must be published first.
        #
        # The THRESHOLD is workload-specific and the default will be wrong for many callers: request
        # rate scales inversely with prompt length, so a fleet on 4x longer prompts needs roughly a
        # quarter of it or never scales at all. config.yaml shows the derivation.
        # None when there is no scaling policy, so the dashboard knows not to draw a threshold line
        # for a threshold that does not exist. Resolved inside the branch, not before it, so a fleet
        # with autoscaling off is not newly rejected for a value nothing reads.
        requests_per_target = None
        if autoscaling_enabled:
            requests_per_target = _num(_given(cfg.get("scalingRequestsPerTarget"), 925),
                                       # minimum=1: a zero or negative target is rejected at synth
                                       # rather than by the Application Auto Scaling API after the
                                       # VPC, ALB and ASG already exist.
                                       "scalingRequestsPerTarget", minimum=1)
            scaling = service.auto_scale_task_count(
                # Bounds are TASK counts, so both are multiplied by replicas-per-instance. Using
                # instance counts directly would cap a multi-engine fleet at a fraction of its tasks.
                min_capacity=instance_count * tuning["replicas"],
                max_capacity=max_instances * tuning["replicas"],
            )
            scaling.scale_on_request_count(
                "RequestsPerTarget",
                requests_per_target=requests_per_target,
                target_group=target_group,
                # Asymmetric: scale out readily, scale in reluctantly, because discarding a warm
                # engine costs minutes of weight loading to undo. Measured, these cooldowns are NOT the
                # dominant term - scale-out took ~11 min to usable capacity, mostly metric lag and
                # weight loading, so lowering the 3 minutes would change almost nothing. See "What
                # autoscaling actually does" in docs/tuning.md.
                scale_out_cooldown=Duration.minutes(3),
                scale_in_cooldown=Duration.minutes(15),
            )
        # The ALB must be allowed to reach the tasks. `connections.allow_to` writes both halves of
        # the rule; adding ingress alone leaves the ALB's egress blocked and health checks silently
        # never arrive.
        alb.connections.allow_to(service, ec2.Port.tcp(CONTAINER_PORT),
                                 "ALB to vLLM tasks")

        # ------------------------------------------------------------------ observability
        #
        # One dashboard, on every deployment, with nothing to switch on. Two kinds of metric feed it:
        #
        # * What the load balancer and the Auto Scaling group already publish: latency, request rate,
        #   errors, healthy tasks, instances. Free, and there is no collection path that can break.
        # * What the engine itself reports, via the metrics sidecar defined with the task above: queue
        #   depth, batch occupancy, KV cache usage, preemptions. These are the numbers that say WHY
        #   latency is rising rather than just that it is; four custom metrics, about $1.20 a month.
        #
        # Everything user-facing here is written in plain language on purpose. The reader is someone
        # woken by an alarm who has never seen this stack, so a widget titled "TargetResponseTime p95"
        # is worth less than one saying what a bad reading means and what to do about it.

        # The p95 you are willing to serve, if you have one. There is no defensible default: a chat UI
        # and a nightly batch job disagree by two orders of magnitude, and shipping someone else's number
        # here would put an arbitrary red line on the graph and page whoever crossed it. So 0 means off -
        # the p50/p95/p99 graph is drawn either way, and the budget line and its alarm appear only once
        # you say what the budget is.
        latency_alarm = _num(_given(cfg.get("latencyAlarmSeconds"), 0),
                             "latencyAlarmSeconds", float, minimum=0)

        # `.metrics.` rather than the `metric_*` methods on the constructs: those are deprecated, and
        # CDK prints a warning per call, which buries the memory-pressure warning above - the one this
        # stack actually needs you to read.
        tg_metrics, alb_metrics = target_group.metrics, alb.metrics
        minute = Duration.minutes(1)
        tasks = instance_count * tuning["replicas"]

        def response_time(percentile: str, **kw) -> cloudwatch.Metric:
            return tg_metrics.target_response_time(statistic=percentile, period=minute, **kw)

        def asg_metric(name: str, label: str) -> cloudwatch.Metric:
            # Raw, because CDK exposes no helper for ASG group metrics. Enabled on the ASG above.
            return cloudwatch.Metric(
                namespace="AWS/AutoScaling", metric_name=name, label=label,
                dimensions_map={"AutoScalingGroupName": asg.auto_scaling_group_name},
                statistic="Maximum", period=minute)

        # Named after the stack so it is findable without hunting through a hashed logical id, and so
        # the console URL can be printed as a plain string rather than a Ref nobody can click.
        dashboard_name = f"{self.stack_name}-serving"
        dashboard = cloudwatch.Dashboard(self, "Dashboard", dashboard_name=dashboard_name)
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title=("Is it slow? - response time"
                       + (f" vs your {latency_alarm:g}s budget" if latency_alarm else "")), width=12,
                left=[response_time("p50", label="typical request (p50)"),
                      response_time("p95", label="slow request (p95)"),
                      response_time("p99", label="slowest requests (p99)")],
                left_y_axis=cloudwatch.YAxisProps(label="seconds", show_units=False),
                left_annotations=([cloudwatch.HorizontalAnnotation(
                    value=latency_alarm, label=f"your budget: {latency_alarm:g}s",
                    color="#d13212")] if latency_alarm else None),
            ),
            cloudwatch.GraphWidget(
                title="How hard is each engine working? - requests a minute per task", width=12,
                left=[tg_metrics.request_count_per_target(
                    period=minute, label="requests/min handled by one task")],
                left_y_axis=cloudwatch.YAxisProps(label="requests/min", show_units=False),
                left_annotations=([cloudwatch.HorizontalAnnotation(
                    value=requests_per_target,
                    label=f"adds capacity above {requests_per_target}", color="#ff7f0e")]
                    if requests_per_target else None),
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Is the load balancer failing? - usually no healthy task", width=6,
                left=[alb_metrics.http_code_elb(elbv2.HttpCodeElb.ELB_5XX_COUNT, period=minute,
                                                label="requests the ALB could not serve")],
            ),
            cloudwatch.GraphWidget(
                title="Are the engines erroring? - the model returned a 5XX itself", width=6,
                left=[tg_metrics.http_code_target(elbv2.HttpCodeTarget.TARGET_5XX_COUNT,
                                                  period=minute, label="errors from the model")],
            ),
            cloudwatch.GraphWidget(
                title="Did an engine die mid-request? - dropped connections", width=6,
                left=[alb_metrics.target_connection_error_count(
                    period=minute, label="connections that failed")],
            ),
            cloudwatch.GraphWidget(
                title=(f"Is CloudFront timing out? - non-streamed answers over "
                       f"{CLOUDFRONT_READ_TIMEOUT_S}s"), width=6,
                # CloudFront publishes its metrics in us-east-1 whatever region the stack is in.
                left=[cloudwatch.Metric(
                    namespace="AWS/CloudFront", metric_name="5xxErrorRate", region="us-east-1",
                    dimensions_map={"DistributionId": distribution.distribution_id,
                                    "Region": "Global"},
                    statistic="Average", period=minute, label="% of requests failing at the edge")],
                left_y_axis=cloudwatch.YAxisProps(label="percent", show_units=False, min=0),
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title=(f"Are the engines up? - {tasks} healthy is normal" if tasks
                       else "Are the engines up? - fleet parked, none expected"), width=8,
                left=[tg_metrics.healthy_host_count(label="healthy and serving", period=minute),
                      tg_metrics.unhealthy_host_count(label="failing health checks", period=minute)],
                left_y_axis=cloudwatch.YAxisProps(label="tasks", show_units=False),
            ),
            cloudwatch.GraphWidget(
                title="Did AWS give us the instances? - a gap means no capacity", width=8,
                left=[asg_metric("GroupInServiceInstances", "instances running"),
                      asg_metric("GroupDesiredCapacity", "instances wanted")],
                left_y_axis=cloudwatch.YAxisProps(label="instances", show_units=False),
            ),
            cloudwatch.GraphWidget(
                title="How much traffic is arriving? - whole fleet", width=8,
                left=[alb_metrics.request_count(period=minute, label="requests/min")],
            ),
        )

        def engine_metric(name: str, statistic: str, label: str) -> cloudwatch.Metric:
            return cloudwatch.Metric(namespace=engine_metrics_namespace, metric_name=name,
                                     statistic=statistic, label=label, period=minute)

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Is work queueing inside the engines? - waiting means saturated", width=8,
                left=[engine_metric("vllm:num_requests_waiting", "Maximum", "waiting, busiest engine"),
                      engine_metric("vllm:num_requests_waiting", "Average", "waiting, typical engine"),
                      engine_metric("vllm:num_requests_running", "Average",
                                    f"running, typical engine (max {tuning['maxNumSeqs']})")],
                left_y_axis=cloudwatch.YAxisProps(label="requests", show_units=False, min=0),
            ),
            cloudwatch.GraphWidget(
                title="Is the KV cache filling up? - near 100% means preemption is next", width=8,
                left=[engine_metric("vllm:kv_cache_usage_perc", "Maximum", "fullest engine"),
                      engine_metric("vllm:kv_cache_usage_perc", "Average", "typical engine")],
                left_y_axis=cloudwatch.YAxisProps(label="fraction of cache", show_units=False,
                                                  min=0, max=1),
            ),
            cloudwatch.GraphWidget(
                title="Are engines redoing work? - preemptions, any is bad", width=8,
                left=[engine_metric("vllm:num_preemptions_total", "Sum", "preemptions per minute")],
                left_y_axis=cloudwatch.YAxisProps(label="preemptions", show_units=False, min=0),
            ),
        )

        # Two alarms, plus an optional latency one. More would mostly restate
        # these, and an alarm nobody trusts is worse than no alarm - an earlier round of work on this project lost real time to a check that
        # cried wolf on every normal model swap.
        #
        # No SNS topic is created here, because a topic with no subscription notifies nobody while
        # looking like it does. Supply `alarmTopicArn` and the alarms notify it; leave it out and they
        # still change state in the console and in `describe-alarms`.
        topic_arn = str(_given(cfg.get("alarmTopicArn"), "")).strip()
        if topic_arn and not re.match(r"^arn:[a-z0-9-]+:sns:[a-z0-9-]+:\d{12}:.+$", topic_arn):
            raise ConfigError(
                f"alarmTopicArn must be an SNS topic ARN (got {topic_arn!r}).\n"
                "  Expected arn:aws:sns:REGION:ACCOUNT:topic-name. Leave it out for alarms that change "
                "state without notifying anyone.")
        actions = ([cw_actions.SnsAction(sns.Topic.from_topic_arn(self, "AlarmTopic", topic_arn))]
                   if topic_arn else [])

        # NOTE the metrics below carry no `label`. A labelled metric cannot be expressed in
        # CloudFormation's simple alarm form, so CDK silently renders the alarm as a metric-math query
        # instead - which works, but leaves MetricName absent from the template and makes the alarm
        # harder to read in the console and in `describe-alarms`. Label for widgets, not for alarms.
        #
        # The descriptions are written to be read on a phone by someone who did not deploy this.
        alarms = [
            tg_metrics.unhealthy_host_count(period=minute).create_alarm(
                self, "UnhealthyTargetsAlarm",
                alarm_name=f"{self.stack_name}-engines-unhealthy",
                alarm_description=(
                    f"At least one engine has been failing its health check for 15 minutes "
                    f"({tasks} should be healthy). Normal during a deploy while weights load; "
                    f"otherwise the container is crashing - check the task's logs in "
                    f"{log_group.log_group_name}."),
                threshold=0, evaluation_periods=15,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                # Fifteen minutes, because a starting task is legitimately unhealthy while it pulls the
                # image and loads tens of GiB of weights. Measured on a fresh instance with a cold cache:
                # about 7 minutes unhealthy in the target group, 14 from launch to healthy. A 5-minute
                # window fired on every first deploy, which is how an alarm gets ignored.
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ),
            alb_metrics.http_code_elb(elbv2.HttpCodeElb.ELB_5XX_COUNT,
                                      period=minute).create_alarm(
                self, "AlbErrorAlarm",
                alarm_name=f"{self.stack_name}-load-balancer-erroring",
                alarm_description=(
                    "Clients are getting 5XX from the load balancer rather than from the model. "
                    "503 means there was no healthy task to send it to; 504 means a request ran past "
                    "the load balancer's 300-second timeout. Check whether the engines are up."),
                # Not zero. A deploy briefly has no healthy target and a scale-in drains a task, so a
                # zero threshold fires on both. Sustained is what matters.
                threshold=10, evaluation_periods=2,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ),
        ]
        if latency_alarm:
            alarms.append(response_time("p95").create_alarm(
                self, "LatencyAlarm",
                alarm_name=f"{self.stack_name}-too-slow",
                alarm_description=(
                    f"Requests are taking longer than {latency_alarm:g}s at p95. The fleet is "
                    f"overloaded or an engine is unhealthy. Check requests/min per task on the "
                    f"{self.stack_name}-serving dashboard: if it is high, the fleet needs more "
                    f"instances (raise maxInstanceCount, or instanceCount if it is already capped)."),
                threshold=latency_alarm,
                # Three minutes, not one: a single minute over budget is what a task starting or
                # draining looks like, and a fleet that cannot grow faster than ~11 minutes gains
                # nothing from being told sooner.
                evaluation_periods=3,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                # An idle fleet publishes no response times at all. Without this the alarm sits in
                # INSUFFICIENT_DATA whenever traffic stops, which trains everyone to ignore it.
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            ))
        for alarm in alarms:
            for action in actions:
                alarm.add_alarm_action(action)


        # ------------------------------------------------------------------ outputs
        CfnOutput(self, "Endpoint", value=endpoint,
                  description="HTTPS, via CloudFront. Send the API key as Authorization: Bearer <key>")
        CfnOutput(self, "ResponsesApi", value=f"{endpoint}/v1/responses")
        CfnOutput(self, "ChatCompletionsApi", value=f"{endpoint}/v1/chat/completions")
        CfnOutput(self, "DistributionId", value=distribution.distribution_id,
                  description="The CloudFront distribution in front of the load balancer")
        CfnOutput(self, "ApiKeyValue", value=key_value,
                  description="Send as Authorization: Bearer <key>. Also visible in the template and in "
                              "Secrets Manager - not confidential from anyone who can read the stack")
        CfnOutput(self, "ModelName", value=cfg["modelId"],
                  description="The model id to put in requests; also what /v1/models returns")
        CfnOutput(self, "ApiKeySecret", value=api_key.secret_name,
                  description="Convenience copy of the same value, not a confidentiality boundary")
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "AsgName", value=asg.auto_scaling_group_name)
        CfnOutput(self, "LogGroup", value=log_group.log_group_name)
        # "DashboardName", not "Dashboard": the Dashboard construct above already owns that id in this
        # scope, and two children with the same id is a synth error.
        CfnOutput(self, "DashboardName", value=dashboard_name)
        # The URL, not just the name. A dashboard nobody can find is not observability, and the console
        # path for a named dashboard is not something anyone guesses. `self.region` resolves to the
        # literal region here and to the AWS::Region pseudo-parameter in an env-agnostic stack, so this
        # is correct either way. Printed by scripts/endpoint_info.py too.
        CfnOutput(self, "DashboardUrl",
                  value=(f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
                         f"?region={self.region}#dashboards:name={dashboard_name}"),
                  description="Open this to see whether the fleet is healthy")
        CfnOutput(self, "ResolvedTensorParallel", value=str(tuning["tensorParallel"]))
        CfnOutput(self, "GpusPerReplica", value=str(gpus_per_replica(inst, tuning)))
        CfnOutput(self, "ContainerMemoryMib", value=str(inst.container_memory_mib
                                                       // tuning["replicas"]))
        CfnOutput(self, "PurchaseModel", value="spot" if use_spot else "on-demand")

    def _container_image(self, image_uri: str) -> ecs.ContainerImage:
        """Resolve the image so ECS is actually allowed to pull it.

        `from_registry` passes the URI through as an opaque string, so CDK cannot tell it names an ECR
        repository and grants the task execution role nothing beyond `ecr:GetAuthorizationToken`. That
        authenticates but cannot fetch a manifest or layers, so the task fails at start with:

            CannotPullContainerError: ... denied

        CDK warns about it ("Proper policies need to be attached before pulling from ECR repository,
        or use 'fromEcrRepository'"), and that warning is easy to read as boilerplate.

        The repository is built from the ARN implied by the URI's OWN account and region rather than by
        comparing them to this stack's: the CDK CLI overwrites `CDK_DEFAULT_ACCOUNT` from the ambient
        credentials, so `self.account` is often not the account in the URI, and a comparison silently
        fell through to `from_registry` with no pull permissions at all.

        Anything that is not an ECR URI - Docker Hub, a public gallery - falls through unchanged.
        """
        # `[^:@]+` for the repository name, and the tag separator matched explicitly, so a DIGEST URI
        # (repo@sha256:...) is handled rather than mangled: `[^:]+` swallowed the `@sha256` into the
        # repository name, producing an invalid ARN and leaving the task unable to pull - the exact
        # failure this method exists to prevent. `\.cn` because ECR in China is amazonaws.com.cn.
        # Stripped ONCE, and the stripped value used down both branches. Cleaning the string for the
        # match and then passing the raw one to from_registry put the padding of
        # `image: "  vllm/vllm-openai:v0.11  "` straight into the task definition, where it failed at
        # task start.
        uri = str(image_uri).strip()
        match = re.match(
            r"^(\d{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?/([^:@]+)"
            r"(?::(.+)|@(sha256:[0-9a-f]+))?$",
            uri)
        if not match:
            return ecs.ContainerImage.from_registry(uri)
        account, region, name, tag, digest = match.groups()
        # A trailing slash is newly reachable now that the tag is optional, and it would produce an
        # ARN ending in `repository/name/` - accepted by IAM, matching nothing.
        name = name.rstrip("/")
        repository = ecr.Repository.from_repository_attributes(
            self, "ImageRepo",
            repository_arn=f"arn:{self.partition}:ecr:{region}:{account}:repository/{name}",
            repository_name=name)
        if digest:
            return ecs.ContainerImage.from_ecr_repository(repository, digest)
        # An untagged reference is legal and means :latest. Requiring a tag meant it fell through to
        # from_registry with no pull grant - the failure this method exists to prevent.
        return ecs.ContainerImage.from_ecr_repository(repository, tag or "latest")
