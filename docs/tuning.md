# Choosing an instance type and tuning the engine

The defaults in `config.yaml` come from the measurements in this document. They are here so you can
tell when your workload justifies a different value, and so you know what was not measured.

---

## The one mental model worth having

A request has two phases with opposite performance characteristics.

**Prefill** reads the prompt. All input tokens go through the model in one pass. Compute-bound; sets
time-to-first-token.

**Decode** generates output one token at a time, each depending on the previous ones.
Memory-bandwidth-bound: every forward pass streams the model's activated weights out of VRAM to
produce one token per sequence.

Same operation, different batch sizes, either side of a crossover:

| Tokens in one forward pass | Bound by | GPU arithmetic units busy |
|---|---|---|
| 1 (decode, one request) | **memory** | almost idle |
| ~8–64 (decode, batched) | memory | rising |
| ~400+ (prefill, or heavy batching) | **compute** | saturated |

Two consequences:

- **Decode speed per request is capped by bandwidth ÷ bytes-read-per-token.** Only reading fewer bytes
  makes a single request faster.
- **Batching is what makes a GPU efficient.** One weight read serving 64 tokens instead of 1 is 64× the
  useful work; KV cache size caps how many requests are in flight.

---

## Choosing an instance type

Every g7e size carries the same GPU: 96 GiB of VRAM at about 1,600 GB/s (the Server Edition runs its GDDR7
at 25 Gbps; the 1,792 GB/s often quoted is the workstation card). Larger sizes add GPUs, vCPU
and host RAM.

Per-GPU columns:

| Instance | GPUs | VRAM | Host RAM | vCPU/GPU | Host RAM/GPU | Use it when |
|---|---|---|---|---|---|---|
| `g7e.2xlarge` | 1 | 96 GiB | 64 GiB | **8** | 64 GiB | The model fits one GPU and you want the lowest cost. **Start here.** |
| `g7e.4xlarge` | 1 | 96 GiB | 128 GiB | 16 | 128 GiB | Same GPU, twice the host: the hedge if 8 vCPU is thin |
| `g7e.8xlarge` | 1 | 96 GiB | 256 GiB | **32** | 256 GiB | Same GPU, the most host per GPU in the family |
| `g7e.12xlarge` | 2 | 192 GiB | 512 GiB | 24 | 256 GiB | The model needs more than 96 GiB |
| `g7e.24xlarge` | 4 | 384 GiB | 1 TiB | 24 | 256 GiB | The model needs more than 192 GiB |
| `g7e.48xlarge` | 8 | 768 GiB | 2 TiB | 24 | 256 GiB | The model needs more than 384 GiB |

vCPU per GPU is not monotonic: the `g7e.8xlarge` gives 32 per GPU, the multi-GPU sizes 24, the
`g7e.2xlarge` 8. This workload is overhead-bound rather than bandwidth-bound (see *Topology* below), so
host CPU per GPU can be part of the ceiling.

### For a model that fits one GPU, buy the smallest instance

Two effects compound.

**Per-GPU price is not flat.** The `g7e.2xlarge` is roughly **19% cheaper per GPU** than every larger
size, which are all priced alike.

**Granularity waste.** The per-GPU rate from *How to size N* below (16,075 tok/s, measured on a
`g7e.2xlarge`) against a demand of 70,028 tok/s is 4.36 GPUs:

| Shape | GPUs | $/GPU/hr | Instances needed | Fleet $/hr | Idle GPUs paid for |
|---|---|---|---|---|---|
| **`g7e.2xlarge`** | 1 | **3.36** | 5 | **16.82** | 0.64 |
| `g7e.4xlarge` | 1 | 4.00 | 5 | 19.99 | 0.64 |
| `g7e.12xlarge` | 2 | 4.14 | 3 | 24.86 | 1.64 |
| `g7e.24xlarge` | 4 | 4.14 | 2 | 33.14 | 3.64 |
| `g7e.48xlarge` | 8 | 4.14 | 1 | 33.14 | 3.64 |

**Five `g7e.2xlarge` cost 32% less than three `g7e.12xlarge`, and 49% less than one `g7e.48xlarge`,
for the same served throughput.**

The `g7e.48xlarge` is the worst value for a model that fits on one GPU. Buy it when a model needs 8
GPUs' worth of VRAM, not for 8 GPUs of throughput.

Unique prompts, same engine configuration, on a `g7e.2xlarge` with a third of the host per GPU:

| | vCPU for its GPU | Uncached prefill @128 | Decode | p95 |
|---|---|---|---|---|
| One GPU of a `g7e.12xlarge` | 48 | 16,176 | 35.5 | 7.37 s |
| **`g7e.2xlarge`** | **8** | **16,195** | 36.5 | 7.35 s |

Identical, +0.1%.

**But not for prefix-sharing workloads.** On cached prompts the `2xlarge` is **15–25% slower** (23,095
against 29,888 at concurrency 128; 40,227 against 46,645 at 256; single runs, hence the range). If
your prompts share a long prefix, prefer a size with more host per GPU.

Prices are illustrative on-demand rates for one region; re-check them for yours.

### How much host CPU an engine needs, and when it matters

One engine, pinned to a subset of one instance's CPUs with `taskset`; only CPU count varied:

| CPUs given to the engine | Cached @128 | Cached @256 | Uncached @128 | Uncached @256 |
|---|---|---|---|---|
| 48 | 30,024 | 46,747 | 16,176 | 21,665 |
| 8 | 29,888 | 46,645 | 15,886 | 22,549 |
| 4 | 29,691 | **32,455** | 15,969 | 21,691 |

**Roughly 8 vCPU per engine is enough, and more is not better.** 8 to 48 changes nothing measurable in
any column.

**Prefix cache hits move the bottleneck from the GPU to the CPU.** A cache hit removes the prefill work
but none of the per-request CPU work (tokenising, detokenising, HTTP, scheduling), so on cached traffic
CPU binds first: 4 CPUs costs **30% of cached throughput at concurrency 256** (46,747 → 32,455) while
barely touching uncached throughput.

| Your traffic | Does vCPU per GPU matter? |
|---|---|
| Unique prompts | **No.** 8 vCPU per engine matches 48. Buy the cheapest GPU. |
| Heavy prefix reuse at high concurrency | **Yes.** Prefer a size with more host per GPU. |

### Will my model fit?

Weights are roughly `parameters × bytes-per-parameter`:

| Precision | Bytes/param | A 30B model | A 70B model |
|---|---|---|---|
| bf16/fp16 | 2 | ~56 GiB | ~130 GiB |
| fp8 | 1 | ~28 GiB | ~65 GiB |
| 4-bit | 0.5 | ~14 GiB | ~33 GiB |

Some VRAM must stay free for the KV cache, activations and CUDA graphs. This project reserves a **fixed 16 GiB per GPU**,
subtracted from what the engine claims, so the weight budget is
`96 GiB × gpuMemoryUtilization − 16 GiB` = **75.2 GiB per GPU** at the default 0.95. Fixed rather than
proportional because activation peaks and CUDA graphs are roughly constant in absolute terms. The
"does not fit" error quotes the same numbers.

One 96 GiB GPU holds a 30B model in bf16, or a 70B in fp8.

---

## Tensor parallelism: a way to make a model fit, not a way to go faster

`tensorParallel` splits one model across N GPUs. Set it to `0` and it is derived: the smallest degree
whose combined VRAM holds the model. Smallest, because splitting costs an all-reduce on every layer of
every forward pass; the GPUs advance in lockstep. With one request that pays. At saturation the synchronisation cancels the parallelism: on two GPUs
with unique prompts, the second GPU added nothing to prefill under TP=2 (22,685 → 21,665), while the
same GPU running an independent engine added 43% (→ 32,467).

Over a slow interconnect it is worse: at TP=4 on PCIe-connected GPUs the model reached a lower fraction
of its memory-bandwidth ceiling than at TP=1 and delivered less in absolute terms.

**Use tensor parallelism when the weights do not fit on one GPU, not to make a fitting model faster.**
If the model does not fit, quantizing to fp8 to get back to TP=1 is usually a better trade than adding
GPUs. Spare GPUs get an independent engine each; see *Topology: replicas or tensor parallelism?*.

The degree must be a power of two (1, 2, 4, 8) and must divide the model's attention head count.

**If you raise the degree outside this project's config, raise `/dev/shm` too.** Workers exchange
tensors through POSIX shared memory; with Docker's 64 MiB default any degree above 1 exits within
seconds with `Insufficient space in /dev/shm`. The task definition here sets 8 GiB. TP=1 never touches
that path, so a working single-GPU deployment proves nothing about a multi-GPU one.

---

## Measured results, and what they imply

All figures below: same model class (~30B mixture-of-experts), same GPU generation, same harness,
1,000-token prompts. Decode is tokens/sec per request under load; prefill is peak tokens/sec.

**How to read these numbers.** Repeat runs of an identical configuration came out **0.7% and 2.1%**
apart. Treat differences of **8% or more as real** and a few percent as noise, including where a table
shows a "winner" by 2–3%. Single runs unless stated.

**Precision, on one GPU.** Relative fleet cost is the cost to serve a fixed request rate; lower is
better.

| Configuration | Decode | Prefill | Relative fleet cost |
|---|---|---|---|
| **FP8, TP=1, 1 GPU** | **118.5** | **9,670** | **1.00** |
| bf16, TP=1, 1 GPU | 109.9 | 6,313 | 1.53 |

Treat 1.53 as a floor on the cost of bf16: it is a single-GPU figure at moderate concurrency with
prefix cache hits. At each precision's own latency-passing knee the gap is closer to **2×** (7,903
input tok/s per GPU against 15,941). See *Every weight option, measured*.

**How to spend two GPUs.** The answer depends on the concurrency and on whether prompts share a prefix.
At **saturation** (concurrency 256, FP8 weights, fp8 KV cache), same model, same instance:

| Topology | GPUs | Cached prefill | Cached decode | Uncached prefill | Uncached decode |
|---|---|---|---|---|---|
| TP=1, one GPU, other idle | 1 | 46,747 | 51.7 | 22,685 | 26.3 ✗ |
| **TP=1 × 2 replicas** | 2 | 47,913 | 52.6 | **32,467** | **36.5 ✓** |
| TP=2 | 2 | **49,201** | **56.5** | 21,665 | 29.1 ✗ |

✓/✗ is a 31.7 tok/s per-request decode gate. **With unique prompts, two replicas are the only two-GPU
topology that passes it**, and they beat TP=2 on prefill by 50%. With cache hits the three are within
5% and TP=2 edges ahead.

The same comparison measured earlier at **concurrency 8–32 with cache hits**, the wrong place to
measure:

| Precision | Topology | Decode | Prefill |
|---|---|---|---|
| bf16 | TP=2 | 119.9 | 9,850 |
| bf16 | 2 replicas, TP=1 each | 94.4 | 9,107 |
| bf16 | TP=1, second GPU idle | 109.0 | 8,584 |
| FP8 | TP=2 | 140.4 | 10,943 |
| FP8 | 2 replicas, TP=1 each | 90.0 | 7,218 |

There TP=2 looks 52% ahead of replicas on prefill. At saturation the same comparison is **+2.7% with
cache hits and −33% without**.

### Quantisation is two independent decisions

Precision is two choices, set by different keys:

| Decision | Key | Default here | What it affects |
|---|---|---|---|
| **Weight** precision | `modelId` / `quantization` | whatever the model ships | the model's own parameters |
| **KV cache** precision | `tuning.kvCacheDtype` | `fp8` | cached attention state, not the weights |

Either can be taken without the other: quantised weights change the model you run, a quantised KV
cache only how attention state is stored.

**Take both if you can.** FP8 weights measured **~35% lower fleet cost** than bf16, with **+53%
prefill** and **+8% decode**, a larger effect than any engine flag. An fp8 KV cache adds **+12%
decode**, growing with concurrency. The defaults assume both.

The weight gain exceeds "half the bytes" because prefill is compute-bound and this GPU generation has
native FP8 tensor cores.

They interact in one direction:

| Weights | Use for the KV cache | Why |
|---|---|---|
| quantised (FP8/4-bit) | **`fp8`** | +12% decode, and small weights leave ample headroom |
| unquantised (bf16) | **`auto`** | fp8 raises memory pressure here rather than lowering it; see `gpuMemoryUtilization` |

#### Every weight option, measured

Every row **meets** the latency budget, at the highest concurrency where it does. Plan against the
unique-prompt column unless you know your prompts share a prefix:

| Weights | Unique prompts | Instances | Shared prefix | Instances |
|---|---|---|---|---|
| AWQ 4-bit ⚠ | **33,496** @256 | **2.1** | **82,112** @512 | **0.9** |
| official FP8 | 32,467 @256 | 2.2 | 74,687 @512 | 0.9 |
| load-time FP8 | 28,363 @256 | 2.5 | 75,368 @512 | 0.9 |
| bf16 | 15,805 @128 | 4.4 | 46,808 @256 | 1.5 |

Prefill tokens/sec on two GPUs as 2 × TP=1 with an fp8 KV cache (`auto` for the bf16 row); instances
to serve a fixed 70,000 input tok/s.

Only the unique-prompt column shows:

- **Unquantised weights cost most without prefix reuse.** 15,805 against 46,808 with cache hits: 4.4
  instances rather than 1.5.
- **The operating concurrency drops to 256 across the board**: per-request decode falls through the
  latency floor before throughput peaks.

⚠ **AWQ 4-bit is the fastest measured option and is not a recommendation.** See below.

#### Load-time FP8 measures the same as an official FP8 build

`quantization: fp8` applied to bf16 weights measured **75,368** tok/s against **74,687** for the
publisher's own FP8 build: parity within 1%. Both routes store the same fp8 numbers in VRAM and run the
same kernels. The differences are not performance:

| | Official FP8 build | `quantization: fp8` |
|---|---|---|
| Download | ~29 GiB | ~57 GiB, converted after loading |
| Startup | loads directly | one conversion pass first |
| Scale factors | the publisher's calibration | per-tensor absolute-maximum |

**Prefer the official build when one exists**: smaller download on every task start, and scale factors
chosen with access to the model and data.

**Planning consequence:** bf16 is *not* the fallback for a model with no official FP8 build. Choose it
only on quality grounds.

#### NVFP4: Blackwell's native 4-bit, measured on eight instances

`nvidia/Qwen3-30B-A3B-NVFP4` with an fp8 KV cache, against load-time FP8 on the same eight
`g7e.2xlarge`, unique 1,000-token prompts:

| Concurrency | FP8 input tok/s | NVFP4 | Gain | p95 FP8 | p95 NVFP4 |
|---|---|---|---|---|---|
| 256 | 59,213 | 73,923 | **+25%** | 4.03 s | 3.34 s |
| 512 | 95,539 | 118,125 | **+24%** | 5.23 s | 4.38 s |
| 768 | 119,223 | 152,672 | **+28%** | 6.33 s | 5.06 s |

p99 spiked in two of the five levels (8.2 s at 256, 9.6 s at 768) where FP8 did not; a 60-second level
is too short to say whether that is noise. Weights are half the size of FP8 again, so the KV cache gets
another ~15 GiB. Output quality was not measured, the same caveat as AWQ below, which is why FP8 stays
the default. To use it, set `modelId` to the NVFP4 checkpoint and clear `quantization`.

**Online NVFP4 does not run on this GPU.** vLLM 0.28 also has `quantization: nvfp4_per_token`, which
quantises bf16 weights at load time the way `fp8` does. On g7e the engine refuses to start:
`nvfp4_per_token online quantization requires a Blackwell (SM100) GPU`. The RTX PRO 6000 is Blackwell
SM120, not the SM100 of B200. So on this hardware NVFP4 means a calibrated checkpoint, and the load-time
convenience of fp8 has no NVFP4 equivalent.

**NVFP4 and EAGLE-3 stack.** The two together, on six instances at the same 96 requests per engine:
133,463 input tok/s, 22,244 per instance against 14,903 for the shipped FP8, **+49% per GPU**, p95
4.54 s against 6.33 s. Acceptance length fell from 2.2 to 1.95, since the speculator was trained against
the bf16 model, and it still added 17% on top of NVFP4 alone.

#### AWQ 4-bit: measured fastest, quality unvalidated

A 4-bit AWQ build measured the best throughput on both prompt conditions: **82,112** cached and
**33,496** unique, against 74,687 and 32,467 for FP8. About 15 GiB of weights instead of 29 leaves more
of the card for KV cache, and cache space caps concurrency.

**This is not a recommendation.** Everything measured here is throughput; **output quality was never
evaluated.** That matters most at 4 bits: it is a far larger perturbation than FP8, a sparse
mixture-of-experts has less redundancy per expert to absorb the error than a dense model of the same
size, and community 4-bit uploads vary in calibration quality.

Validate it against your own evaluations, as you would any model change. Do not adopt it on a
throughput number alone.

#### If you cannot quantise the weights

Choose bf16 only if quantised weights are unacceptable on quality grounds; "no official FP8 build" is
**not** a reason, since load-time `quantization: fp8` matches one. Then:

```yaml
modelId: <publisher's bf16 model>
quantization: ""          # leave empty - do not quantize at load time
tuning:
  kvCacheDtype: auto      # NOT fp8 with unquantised weights - see below
```

**With unquantised weights, do not quantise the cache.** The combination either runs out of VRAM under
load or, at the lower utilisation that survives, is slower than not doing it (37,877 against 46,808).
The stack warns about it at synth time.

The cost **depends heavily on whether your prompts share a prefix**:

| Weights | Prompts | Operating concurrency | Prefill tok/s | Relative fleet |
|---|---|---|---|---|
| FP8 | shared prefix | 512 | 74,687 | **1.00** |
| bf16 | shared prefix | 256 | 46,808 | 1.60 |
| FP8 | unique | 256 | 32,467 | 2.30 |
| **bf16** | **unique** | **128** | **15,805** | **4.70** |

Per GPU, at each precision's own latency-passing knee, bf16 sustained **7,903 input tok/s against
15,941 for FP8**: almost exactly **2×**.

- **With prefix reuse, bf16 costs about 1.6× the fleet** at half the operating concurrency.
- **bf16 loses on two counts at once, so the gap is 2× rather than the ~1.5× weight size predicts.** It
  is 1.5× behind FP8 at the same concurrency, *and* its usable load is lower: 64 concurrent per GPU
  against 128, because per-request latency reaches the budget sooner.

Comparing FP8 against bf16 on your own evaluations is cheaper than doubling a fleet.

Two further consequences:

- **The weights need roughly twice the VRAM** (*Will my model fit?*). If bf16 pushes a model past one
  card, use `tensorParallel: 2`. **Do not raise it while the model still fits**: a ~30B model is
  ~57 GiB in bf16, inside one 96 GiB card, and TP=2 measured **6,991 tok/s per GPU against 6,989 for
  TP=1**. Two containers, one per GPU, measured **13% better** than TP=2 on the same two GPUs (15,805
  against 13,981). The one-engine-per-GPU rule holds at both precisions.
- **Expert parallelism may help in bf16 where it hurts FP8.** Measured **+15%** in bf16 and **−13%** in
  FP8, so test `enableExpertParallel: true` on a bf16 deployment.

#### If you cannot quantise the KV cache either

Set `kvCacheDtype: auto`. **An fp8 KV cache is a lossy store for attention state**: the computation is
unchanged, but the cached intermediate values lose precision. If you ruled out weight quantisation on
quality grounds, decide this one explicitly too.

`auto` costs the +12% decode above and nothing else. Use it as the quality baseline; if fp8 compares
acceptably, take it. With unquantised weights, `auto` is required anyway.

### Topology: replicas or tensor parallelism?

**If the model fits on one GPU, run one engine per GPU.** Use tensor parallelism when the model does
not fit, or when your prompts share a long common prefix.

The mechanism:

**Tensor parallelism splits every request across GPUs**, with an all-reduce per layer, so the GPUs
advance in lockstep. At high concurrency **the second GPU contributed nothing to uncached prefill**
(22,685 on one GPU, 21,665 across two).

**Independent engines split the requests instead of the request**, with no coordination at any layer:
22,685 → 32,467 uncached prefill for the same second GPU.

TP buys lower latency on one request; replicas buy more requests at once. A saturated server needs the
second.

**The distinction is one model split versus independent replicas, not one process versus several.**
vLLM can run N independent replicas inside one process with `--data-parallel-size N`, and that
configuration clears the split-model plateau.

#### Data parallelism: an option that exists, and loses on realistic traffic

`--data-parallel-size N` runs N replicas inside one process with vLLM routing between them, which could
route by cache affinity where a load balancer round-robins blindly. **Tested directly, it did not win.**

Three prompt shapes at concurrency 256, on two GPUs:

| Prompt shape | Two containers (ALB) | DP=2 (vLLM routing) | Winner |
|---|---|---|---|
| Every prompt unique | **32,467** | 27,793 | containers, **+17%** |
| **8 shared prefixes + unique tails** | **36,106** | 32,399 | containers, **+11%** |
| Every prompt identical | 47,913 | **54,019** | DP, +13% |

The middle row (a common system prompt with per-request content) is the realistic one and was
constructed to favour DP's routing. DP lost it by 11%.

**DP only wins when every request is literally the same prompt**, a property of benchmarks, not
workloads. Use one container per GPU.

DP has **better decode** (45.3 against 41.3 tok/s) and **worse prefill and time-to-first-token**
(3.05 s against 1.84 s): every request queues behind one API server. On unique prompts at concurrency
256 it also fails the latency budget, p95 8.57 s against 8 s.

**Why DP wins the identical-prompt case is unexplained.** The middle row rules out cache affinity.

Two further reasons for separate containers: a DP process is one load balancer target, so a sick
replica inside it is invisible to health checks; and one process is one failure domain.

To try it anyway, set `extraArgs: "--data-parallel-size 2"` and read the ENTRYPOINT note in
[troubleshooting.md](troubleshooting.md) first.

**Where each wins:**

| | One engine per GPU | Tensor parallelism |
|---|---|---|
| Model does not fit on one GPU | impossible | **the only option** |
| Unique prompts, saturated | **+50% prefill, and the only topology that met the latency gate** | fails the gate |
| Shared prefixes, saturated | within 3% | **slightly ahead** (49,201 vs 47,913) |
| Low concurrency | worse | **better**: nothing else can use the idle GPU |
| Fault isolation | **one engine dying costs 1/N of capacity** | one failure domain |
| Rolling model updates | **yes** | no |
| Reasoning about it | **one model, one GPU, no interaction** | collectives, shared KV, lockstep |

**"Replicas are not a throughput strategy" holds only for cached workloads at moderate concurrency**,
where it was measured; without cache hits the picture inverts.

For throughput, in order:

1. **Prefer FP8** so the model fits on fewer GPUs.
2. **One engine per GPU** if it fits: `tensorParallel: 1`, `replicas:` = the instance's GPU count.
3. **Tensor parallelism only when the weights need it**, or when your prompts reliably share a prefix.
4. **Scale by adding instances.**

`replicas × tensorParallel` must not exceed the instance's GPU count.

---

## Larger instances: what is measured and what is not

**Measured:** one GPU at TP=1, and two GPUs three ways (TP=2, TP=1 with one GPU idle, 2 × TP=1
replicas) at concurrency 256, cached and uncached.

**Not measured: TP=4 and TP=8.** Four- and eight-GPU instances of this generation were not obtainable
during testing.

**First, check you want an 8-GPU instance at all**; if the model fits on one GPU, single-GPU instances
are cheaper for the same throughput (*For a model that fits one GPU, buy the smallest instance*).

### Recommended starting point for an 8-GPU instance: 8 replicas × TP=1

```yaml
instanceType: g7e.48xlarge
tuning:
  tensorParallel: 1
  replicas: 8
```

One engine per GPU:

1. **It is the measured shape, wider.** 2 × TP=1 won at two GPUs; 8 × TP=1 is the same topology.
2. **The model does not need more than one GPU.** A ~30B model in FP8 is about 29 GiB and fits a 96 GiB
   card with room for KV cache.
3. **Tensor parallelism does not scale under load.** The second GPU added nothing to uncached prefill
   at TP=2 (22,685 → 21,665). Eight GPUs will not do better: the collective gets wider and the
   synchronisation more expensive.
4. **It fails gracefully.** One engine dying costs an eighth of capacity. With 1 × TP=8 it costs all of
   it.
5. **It is simpler to reason about.** One model, one GPU, no collectives.

### The alternative: 4 replicas × TP=2, for high-cache-hit workloads

```yaml
tuning:
  tensorParallel: 2
  replicas: 4
```

Only if your prompts reliably share a long prefix: TP=2 measured slightly ahead of 2 × TP=1 (49,201 vs
47,913 prefill, about 3%). Without cache hits, at two GPUs TP=2 *failed* the latency gate that
2 × TP=1 passed.

### What to measure first

In order:

1. **Host contention at 8 engines.** Contention for vCPU, host memory bandwidth and PCIe was measured
   with **two** engines, not eight; an 8-GPU instance has proportionally more host per GPU, but four
   times the engines is well outside what was tested.
2. **8 × TP=1 against 4 × TP=2 on your own prompts.** The crossover is prefix-sharing.
3. **8 × TP=1 against 1 × TP=8.** Expected to lose badly, but nobody has measured TP=8 on this
   hardware.

If host contention bites at 8 engines, the fix is fewer, wider engines: 4 × TP=2.

### Two practical points for a multi-engine instance

- **Host memory is divided by the number of engines, and getting it wrong is silent.** ECS reserves the
  full memory limit per task, so eight tasks each asking for the whole instance's share means seven
  never place: one engine at an eighth of the throughput you pay for. This project divides
  `container_memory_mib` by `replicas` automatically; if you set memory yourself, divide it.
- **`/dev/shm` is already sufficient** for any topology up to TP=8.

---

## Engine tuning

### `gpuMemoryUtilization: 0.95`: the biggest single effect for a model that fills the card

The +27% below is for a model occupying most of the card, where the last 5% is a large share of what
remains for the KV cache. For a model that leaves tens of GiB free, the same 5% is a few percent of the
cache and this setting is not a lever.

Not the engine's 0.90 default. The KV cache is whatever remains after the weights; on a model that
fills most of the card, the 5% of VRAM that 0.90 leaves unclaimed can be larger than the entire KV
cache.

Measured, 0.90 → 0.95 on a model occupying most of a card: **+27% throughput, −22% cost per token**.
The single largest win of any parameter tested.

**Do not exceed 0.97, and 0.95 is the better default anyway.** Above it the engine passes startup
memory profiling and fails under real load. 0.95 → 0.97
measured **+1.6%**; at 0.97 the mixture-of-experts model above was stable, while a dense 27B on the
same hardware crashed under load.

**0.95 itself is not universally safe.** Two independent cases where it is too high:

1. **Dense models**, as the 27B above.
2. **Unquantised weights with an fp8 KV cache.** Starts cleanly, passes its health check, then fails
   with CUDA out-of-memory under traffic, surfacing as `502`. An fp8 cache holds twice as many tokens
   per byte, so batches grow, so the per-step fused-expert workspace grows, and bf16 weights have
   already taken most of the card.

   **The fix is a plain KV cache, not lower utilisation.** Lowering it stops the crash and is still a
   net loss: at 0.90 the same deployment measured **37,877** tokens/sec against **46,808** for a plain
   cache at 0.95.

   The stack warns at synth time when it detects this; [troubleshooting.md](troubleshooting.md) has
   the diagnostic.

   | weights | KV cache | why |
   |---|---|---|
   | FP8 | **fp8** | +12% decode, and 29 GiB of weights leaves ample headroom |
   | bf16 | **default** | fp8 either crashes or, once made safe, is slower than not doing it |

**Do not bother with `--kv-cache-memory`**, even though the engine suggests it at startup:

```
Actual usage is 52.7 GiB for consumed memory (weights + non-torch), 5.65 GiB for peak
activation, and 0.47 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with
`--kv-cache-memory=38082119680` (35.47 GiB) to fully utilize gpu memory.
```

It is a **no-op**: setting it explicitly measured 74,491 against 74,687 prefill for the fraction alone.

### `enablePrefixCaching: true`: always on, but read the caveat

Reuses computed attention state across requests sharing a prompt prefix. For a common system prompt or
tool schema (most agent and RAG traffic) it is worth up to an order of magnitude on time-to-first-token
under load. It removes prefill work only; decode still re-reads the whole cache for every token, so it
does nothing at concurrency 1.

**Leave it on whatever your workload.** Turning it off on prompts that share nothing measured
**−1.2%**.

**The caveat is about concurrency, not the flag.** Every throughput figure here was measured with
prompts that share a prefix; *Choosing an operating concurrency* gives both numbers. Sizing a fleet from
the cached figure when your traffic has no shared prefix **under-provisions by about 2×**.

### `maxModelLen: 0` (the model's own maximum)

Maximum tokens per request, input plus output.

**"Size the context window to your workload" is widely repeated and measurably false here.** Cutting it
8× (32,768 → 4,096 on a ~1,200-token workload) changed throughput by −1.6%, noise. vLLM allocates KV
cache in fixed-size blocks on demand; the ceiling is not a reservation, so lowering it frees nothing.

Set it only to **reject** requests longer than some limit. Not to go faster.

### `maxNumSeqs: 256`

Ceiling on concurrent sequences. A ceiling, not a reservation: it reserves no memory; too low leaves
the GPU idle.

128 → 256 measured **+1.4%**; 256 → 512 was noise. Both on 1,000-token prompts.

**Keep it well above the concurrency you load-test at**, or you are measuring this flag, not the
hardware: requests past it queue instead of batching.

**With long prompts the cap has to come from the KV cache, not from this flag.** The cache holds a fixed
number of tokens: on this GPU with FP8 weights and an fp8 cache, about 63 GiB ÷ 48 KiB per token ≈
1.3 million tokens. 256 sequences of 1,200 tokens fit ten times over; 256 sequences of 12,000 tokens do
not, and the engine preempts rather than refuse. Derive the cap for your longest common request:
`cache tokens ÷ (input + output tokens)`, and set `maxNumSeqs` at or below it. The dashboard's KV cache
and preemption widgets show when this is the limit.

### `maxNumBatchedTokens: 0` (engine default)

Token budget per scheduler step.

**No measured effect from 8k to 64k; leave it alone.** 64k landed within noise (decode 136.0 → 136.2,
prefill 19,069 → 19,274), as did cutting it 4× to 8k (prefill 49,603 vs 49,774). The prefill batch is
never the constraint at this scale, so the default omits the flag.

### `kvCacheDtype: fp8`

Stores the KV cache at 8 bits instead of 16. Measured **+12% decode** at full load (163.7 vs 146.7,
like-for-like), up from +4.9% at low concurrency.

At high concurrency the KV cache is roughly half of all decode memory traffic, and at concurrency 1
almost none, so halving it is worth close to 10% with a full batch and nothing alone. Smaller entries
also fit more requests in the same VRAM.

**On by default here.** It is a lossy store for cached attention state; to rule that out, set
`kvCacheDtype: auto`.

### `enableExpertParallel: false`: measure it, do not assume

For mixture-of-experts models, pins whole experts to specific GPUs instead of sharding every expert
across all of them. Communication changes shape: one all-reduce per layer becomes an all-to-all
dispatch and combine.

**The measurements disagree, and that is the finding:**

| Hardware and configuration | Effect on decode |
|---|---|
| bf16, TP=2, fast interconnect | **+15%** (138.4 vs 119.9) |
| FP8, TP=2, fast interconnect, at full load | **−13%** (142.5 vs 163.7) |
| bf16, TP=4, PCIe-connected GPUs | **−18%** |

Two variables move the answer. **Interconnect:** each token routes to a handful of experts scattered
across GPUs, and all-to-all punishes a slow link far harder than all-reduce. **Precision:** in FP8 the
GPU finishes its share sooner, so communication is a larger fraction of the step.

No correct default exists, so this project ships it off. Expect a gain in bf16 on a fast interconnect,
a loss in FP8, and a significant loss over PCIe. Measure it: one flag, one restart.

---

## Settings that sound useful and measurably are not

Each was tested and made things worse.

### `--enforce-eager`: measured **−85% decode**, and it is the advice you will be given

Disables CUDA graphs: every forward pass is dispatched op by op from Python.

Of 23 configurations tested, this was **the only one that failed a latency budget**: decode fell from
117.5 to **17.8 tokens/sec per request**. A decode step does little GPU work per token, so launch
overhead dominates without CUDA graphs.

It is the standard first suggestion when an engine will not start, and it does fix startup problems
(lower memory use, no graph capture). Use it to *diagnose*, through `EXTRA_ARGS`, and remove it before
measuring or serving.

`validate_tuning` rejects `enforceEager: true` outright with this number attached.

### N-gram speculative decoding: measured **−58%**

Drafts several tokens ahead from the prompt, then verifies them in one pass. Output is identical, so it
is pure speedup when it works. It failed for three compounding reasons:

1. **It spends compute to buy latency**: drafting 5 tokens means ~6× the work per forward pass,
   refunded only when the draft is accepted, and at high concurrency there is no spare compute.
2. **Acceptance was near zero**: matching against the prompt works where output quotes input
   (summarisation, code editing, extraction), not for open-ended generation.
3. **It silently disables asynchronous scheduling.**

Revisit only for a large model, at low concurrency, on a workload whose output quotes its input.

### Speculative decoding with EAGLE-3: the largest gain measured, and it is a flag

An EAGLE-3 draft model is a small separate checkpoint trained against the target model. vLLM 0.28 loads
it with `--speculative-config`; nothing has to ship inside the target checkpoint. Publishers release them
alongside the model. For the shipped model:

```yaml
extraArgs: --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3","num_speculative_tokens":3}'
```

Measured on eight `g7e.2xlarge`, FP8 weights, fp8 KV cache, unique 1,000-token prompts, 190-token
answers, 60 s per level after warm-up:

| Concurrency | Without, input tok/s | With EAGLE-3 | Gain | p95 without | p95 with | Decode tok/s per request |
|---|---|---|---|---|---|---|
| 8 | | 7,381 | | | 1.22 s | 189 |
| 64 | | 32,842 | | | 2.02 s | 105 |
| 256 | 59,213 | 83,707 | **+41%** | 4.03 s | 3.17 s | 48.9 → 67.9 |
| 512 | 95,539 | 123,924 | **+30%** | 5.23 s | 4.32 s | 39.2 → 50.6 |
| 768 | 119,223 | 147,914 | **+24%** | 6.33 s | 5.62 s | 33.0 → 40.7 |

Mean acceptance length 2.2 of 3 drafted tokens. Output is the same distribution as without speculation;
the engine warns that `min_p` and `logit_bias` are unsupported with it. Not shipped as the default because
the draft model is specific to the target: change `modelId` and this line must change with it, and a
mismatch fails at startup.

An earlier version of this document said EAGLE and multi-token prediction were not configuration options
and had to ship inside the checkpoint. That was true of older engine versions and is wrong for 0.28.

---

## Choosing an operating concurrency

Measured on the recommended topology (two GPUs as 2 × TP=1, FP8 weights, fp8 KV cache, ~1,200-token
prompts) against an 8-second p95 budget and a 31.7 tok/s per-request decode floor:

| Concurrency | Prompts | Prefill tok/s | Decode per request | p95 | Verdict |
|---|---|---|---|---|---|
| 256 | shared prefixes | 46,235 | 51.9 | 4.8 s | pass, wide margins |
| **512** | **shared prefixes** | **74,687** | **42.3** | **5.9 s** | **best measured** |
| 512 | unique | 44,705 | 25.1 | 10.6 s | **fails both** |
| 768 | either | 35,669 ↓ | 10.9 | 19.3 s | **fails badly** |

**Two answers:**

| Your traffic | Operating concurrency | Usable prefill | Instances for 70,000 input tok/s |
|---|---|---|---|
| Requests share a prefix (common system prompt, tool schema, shared document) | **512** | 74,687 | **0.9** |
| Every prompt is unique | **256** | 32,467 | **2.2** |

Quote 74,687 *with its condition attached*.

**Per GPU, the unique-prompt limit is ~128 concurrent requests** (256 across two GPUs). An 8-instance
fleet of single-GPU instances hit its knee at the same 128 per instance (*What a deployed fleet
actually held up to*).

**If you do not know which row you are on, plan for the unique-prompt row.** It is a factor of ~2.4 in
fleet size. Prefix reuse is easy to overestimate: a shared system prompt only helps as identical bytes
from the start of the request.

**Past 512 the cliff is sharp.** At 768, prefill halves (74,687 → 35,669), per-request decode falls to
10.9, and p95 more than triples to 19.3 s.

**Topology at the load that matters, both precisions:** FP8 at concurrency 512 with cache hits, 2 × TP=1
delivered 74,687 against TP=2's 65,152 (**+15%**). bf16 at 256, the highest concurrency either passes
the budget: 46,808 against 42,947 (**+9%**). At 256 in FP8 TP=2 is marginally ahead, so compare at the
load you will run. Comparing bf16 at 512 would reverse this and mean nothing: both bf16 configurations
fail the budget there (decode 27.7 and 29.6 against 31.7 required).

### What a deployed fleet actually held up to

Measured on the shipped default (`g7e.2xlarge` on spot, one engine per GPU): **1,000-token unique
prompts, 190 output tokens**, over `/v1/responses`.

| Instances | Concurrency | Input tok/s | Requests/sec | p50 | p95 | Errors |
|---|---|---|---|---|---|---|
| 1 | 128 | 16,075 | 17.2 | not recorded | 7.42 s | not recorded |
| **6** | 768 | **95,645** | 102.1 | 7.45 s | 8.38 s | 0.01% |
| **8** | 1024 | **132,772** | 141.8 | 6.92 s | 7.60 s | 0.00% |

**Fleet capacity is linear in instance count, and a single-instance benchmark predicts it within ~3%.**
Six instances delivered 99.2% of six times the one-instance figure; eight delivered 103% of eight
times. Each engine has its own GPU, KV cache and process; nothing shared appears as the fleet widens.

#### The whole envelope, on eight instances

Same workload, same fleet, concurrency swept:

| Concurrency | Per instance | Requests/sec | Input tok/s | p50 | p95 | p99 | Within 8 s |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 59.7 | 55,917 | 4.12 s | 4.51 s | 4.67 s | yes |
| 512 | 64 | 96.7 | 90,579 | 5.15 s | 5.74 s | 6.28 s | yes |
| **1024** | **128** | **141.8** | **132,772** | 6.92 s | **7.60 s** | 13.10 s | **yes** |
| 1536 | 192 | 167.6 | 156,998 | 8.49 s | 11.04 s | 12.47 s | **no** |

**The knee is at ~128 concurrent per instance, where a single instance also saturated.** The limit
belongs to one engine on one GPU and does not move behind a load balancer.

**Past the knee you buy throughput with latency.** 1024 to 1536 concurrent (+50% offered load) bought
**+18% throughput and +45% p95**, which left the budget.

**The tail degrades first.** At 1024 concurrent p95 read 7.60 s while **p99 was 13.10 s**. Size on p99.

An earlier datum: the same 6-instance fleet served **~115 requests/second at ~0.1% errors** with
**1-token** outputs, a request-rate figure only; it shows the load balancer, the header-auth listener
rule and 6-way balancing hold up.

These are **unique-prompt** figures. With prefix cache hits the same hardware goes roughly 2× further;
see *Choosing an operating concurrency*.

### If you are benchmarking this yourself

- **Do not stop at 32.** Extending one sweep from 32 to 64 nearly doubled measured peak prefill
  (10,943 → 20,293). A number from concurrency 32 roughly **doubles the instance count** you conclude
  you need.
- **Sweep until throughput stops rising**, then step back one point. Here 512 was only identifiable by
  seeing 768 fall; the interesting region was 256–768.
- **Raise `maxNumSeqs` above your highest test point** first, or you are measuring the flag.
- **Report aggregate and per-request numbers together.** Size the fleet on the first, check the latency
  budget on the second. At 512 uncached only the second fails, and a throughput-only sweep would have
  called it the winner.
- **Take both numbers from the same run.** Peak throughput from one concurrency and a passing latency
  check from a lower one overstated per-instance capacity by up to 21% in one harness. **A throughput
  number is only usable if the same run met the latency budget.**
- **Ask for token counts explicitly when streaming** (`"stream_options": {"include_usage": true}` on
  the OpenAI-compatible API). Otherwise `prompt_tokens` reads as zero and aggregate prefill silently
  computes as zero or falls back to counting chunks.
- **Wait for healthy, then warm up, then measure.** Benchmarking from the instant a target reports
  healthy understated throughput by ~21% here: the first requests pay for CUDA graph capture and kernel
  autotuning.
- **If throughput FALLS as you add concurrency, suspect your load generator.** A real server plateaus;
  it does not halve. One Python process driving 768 connections measured **33,688 tok/s against a true
  95,645** (53,492 at 384, then 33,688 at 768); 12 processes gave the correct answer. Even at 384 the
  single-process result was 16% low.
- **Watch p99, not just p95.** At 1024 concurrent, p95 was 7.60 s while p99 was 13.10 s.
- **Use your own prompt shapes and prefix-sharing.** It moves the answer by 2×.

`scripts/test_endpoint.py` is a smoke test: it shows the endpoint holds up, not where its ceiling is.
`scripts/benchmark.py` does the sweep above: multi-process, warm-up first, aggregate and per-request
numbers from the same run, unique prompts unless you pass `--shared-prefix`.

---

## Sizing a fleet, and when to autoscale

Everything above configures one instance. This is how many to buy.

### Default shape: one task per GPU, on the smallest instance that fits

Express it as *N instances of size X, one task per GPU*:

```yaml
instanceType: g7e.2xlarge     # 1 GPU
instanceCount: 5              # 5 instances -> 5 tasks
tuning:
  tensorParallel: 1
  replicas: 1                 # tasks per instance = GPUs per instance
```

`instanceCount × replicas` is the task count; `replicas × tensorParallel` must not exceed the
instance's GPUs. On a 2-GPU instance set `replicas: 2`.

Four reasons:

1. **~19% cheaper per GPU** than any larger size (*Choosing an instance type*).
2. **Scales in 1-GPU steps.** Most demand figures do not land on a multiple of 8.
3. **Failure isolation.** Losing one instance costs 1/N of capacity.
4. **Spot availability, often the deciding factor.** A 1-GPU `g7e.2xlarge` was obtained on spot in
   **22 seconds**. `g7e.24xlarge` and `g7e.48xlarge` returned `InsufficientInstanceCapacity` on **300+
   consecutive attempts across all four availability zones**, with 768 vCPU of quota free: capacity,
   not quota. **Large GPU instances are hard to obtain; small ones are not.** Recovery from spot
   reclaims follows the same asymmetry.

### How to size N

1. Deploy **one** instance.
2. Find the highest concurrency at which it meets your latency budget (sweep as in *Choosing an
   operating concurrency*, watching p95 and per-request decode).
3. Convert to the unit your demand is in (requests/second, or input tokens/second).
4. Divide demand by it and round **up**.

One `g7e.2xlarge` sustained **16,075 input tok/s** inside an 8-second p95 budget, about **16
requests/second** at this prompt size. Against demand of 70,028 input tok/s:

```
70,028 / 16,075 = 4.36  ->  5 instances minimum, 6 with headroom
```

Round up, then add headroom (next section).

The one-instance figure predicted the 6- and 8-instance fleets within ~3%
(*What a deployed fleet actually held up to*).

### What to scale on

**Autoscaling is ON in the shipped config**: `instanceCount: 16` with `maxInstanceCount: 24`, which
creates target tracking on the load balancer's request count per target. Set the two equal (or leave
`maxInstanceCount` unset) for a fixed-size fleet, and no scaling policy is created.

Signals:

| Signal | Verdict |
|---|---|
| GPU utilisation | **Bad.** Sits near 100% while latency is still fine. It has no relationship to the SLA, so it either scales constantly or never. |
| ALB `TargetResponseTime` p95 | Direct, but **lagging**: it only rises once requests are already slow, so you scale *after* breaching the budget. |
| **ALB `RequestCountPerTarget`** | **Recommended.** Leads both, native to ECS target tracking, no custom metrics to publish. |
| Engine `num_requests_waiting` | Best signal in principle (a queue forming *is* saturation), but it must be published as a custom metric first. |

### `scalingRequestsPerTarget`: derive it, do not inherit it

**The shipped 445 is specific to one workload**; expect to change it.

It is the **requests per minute, per task**, at which capacity is added. Derivation:

```
one g7e.2xlarge sustained ~17 requests/sec inside an 8 s p95 budget   (1,000-token prompts)
17 × 60                = 1,020 requests/minute per task at saturation
1,020 × 0.9            ≈ 925                                          <- for 1,000-token prompts
```

The **shipped default is 445, not 925**: the example workload it is sized for mixes two request shapes with a
request-weighted average input of ~1,780 tokens rather than 1,000. `15,500 / 1,780 = 8.7` req/s per
task, x60 x0.85 = 445. Longer prompts mean fewer requests carrying the same tokens, so the threshold
comes down (next section). Split into one fleet per shape and the thresholds become ~925 and ~330; a
blended threshold serves neither shape well.

The 0.9 is the margin, and 90% is *late* given the ~11 minutes scale-out takes; to grow sooner, lower
the multiplier, not the measured rate.

#### Estimate it from traffic you already have, before deploying anything

**If you know your average prompt length**: per-GPU input tokens/sec is roughly constant across prompt
sizes (16,075 against 15,277 for a 4x difference in prompt length, table below). So:

```
requests/sec per task  =  ~16,000  ÷  your average INPUT tokens per request
threshold              =  that  × 60  × 0.85
```

**If you only know aggregate volume**, derive the average request size:

```
average tokens per request  =  (tokens per minute ÷ 60)  ÷  requests per second
average INPUT tokens        =  that  −  your average output length
```

Do this even if you think you know your prompt size.

Treat ~16,000 as an order-of-magnitude figure for this model class on this GPU. It moves with weight
precision (unquantised weights are roughly half) and with the model. Being 20% out costs a little early
scaling; being 4x out means never scaling at all.

#### Then confirm it by measuring

1. Deploy one instance and find the highest concurrency it sustains inside your latency budget (see
   *Choosing an operating concurrency*).
2. Take the requests/second it achieved there and multiply by 60.
3. Multiply by 0.8–0.9.

If the measurement disagrees with the estimate by more than about 30%, trust the measurement and check
whether your real prompts are longer than you assumed, the usual cause.

#### Why this number does not transfer between workloads

**Request rate scales inversely with prompt length, while tokens/sec per GPU stays roughly constant.**
Measured on identical hardware:

| Prompt size | Input tok/s per GPU | Requests/sec per GPU | Correct threshold |
|---|---|---|---|
| 1,000 tokens | 16,075 | 17.1 | **~925/min** |
| 4,000 tokens | 15,277 | 4.2 | **~225/min** |

Same GPU work in both rows, packaged into a quarter as many requests, so the request threshold comes
down by the same factor.

**Quadruple your prompt size and keep 925:** the task saturates at 4.2 requests/sec, 252/minute, so
925 is **3.7× higher than the task can ever reach**. The alarm never fires, the fleet never grows, and
there is no error.

Rule of thumb without re-measuring: scale the threshold by `1,000 ÷ your average prompt tokens`.

#### The blind spot

**Request count is indifferent to request cost.** A hundred 200-token requests and a hundred
4,000-token requests look identical to this metric, and they are twenty times apart in GPU work. If
your traffic mixes prompt sizes unpredictably, either split it into separate fleets per shape, or
publish `num_requests_waiting` as a custom metric and scale on that.

#### The whole scaling surface

Four values, all in `config.yaml`:

| Setting | Controls |
|---|---|
| `instanceCount` | the minimum, and the fixed size when autoscaling is off |
| `maxInstanceCount` | the ceiling, **and whether autoscaling exists at all**: equal to `instanceCount` (or 0) creates no scaling policy |
| `scalingRequestsPerTarget` | the threshold above |
| `useSpot` | unrelated to scaling, but it decides how obtainable each added instance is |

The cooldowns (3 minutes out, 15 in) are not exposed. They are not the dominant term in either
direction (see the timings below).

As a safety net against a slow-but-not-yet-queueing regression, add a step-scaling alarm on p95
`TargetResponseTime` above about **75% of your budget** (6 s for an 8 s budget).

### What autoscaling actually does, measured

Timings from a real deployment: 6 -> 8 single-GPU instances, ~29 GiB of weights read from S3. They
scale with weight size and fleet shape.

| Transition | Elapsed | Waiting on |
|---|---|---|
| load starts -> tasks 6 -> 8 | **~6 min** | load balancer metric publication, then a 3-datapoint alarm |
| -> new tasks healthy | **+~5 min** | pulling and loading weights |
| load stops -> tasks 8 -> 7 | **~17 min** | the 15-datapoint low alarm |
| -> instance terminated | **+~15 min** | the capacity provider's own scale-in evaluation |
| full 8 -> 6 convergence | **~45–60 min** | one step per cooldown |

Consequences:

- **Scale-out is ~11 minutes to usable capacity, not 3.** Metric lag, three alarm datapoints, then
  weight loading; the cooldown is almost irrelevant. The minimum has to cover steady state on its own.
- **Target tracking scales in ONE STEP AT A TIME.** 8 -> 6 is two steps of roughly 15 minutes. Expect
  a slow shrink after a spike.
- **The instance outlives the task by about 15 minutes.** The ECS capacity provider runs its own
  scale-in evaluation after the service removes a task: tasks drop to 7 while the Auto Scaling Group
  stays at 8 for another cooldown, and **you keep paying for the GPU instance**.
- **Do not treat autoscaling as a way to save money on a GPU fleet.** The shrink is slow. Size the
  minimum for steady state and treat scale-out as insurance.

**What it is for.** A 15-minute sustained run at 768 concurrent scaled 6 → 8 mid-run and averaged
**111.7 requests/second at p95 7.11 s, with 2 failures in 100,520 requests**. The identical load
against a *fixed* 6 instances sat at p95 **8.38 s**, outside an 8-second budget. Recovering the latency
budget, not cost, is the case for enabling it.

Two things about scale-in that are not obvious from the settings: the Auto Scaling group picks the
instance to terminate by its own policy, not by which one is busiest, so a scale-in after a burst can
drain a loaded engine while an idle one survives, and its replacement cold-starts elsewhere. And on a
fixed fleet a redeploy takes engines down for the reload time, because with fully reserved GPUs no new
task can be placed until an old one stops; with headroom ECS rolls instead. README.md, "Changing the
model, tuning or image later" has the measurements and the choice.

---

## Watching a running fleet

Every deployment gets one CloudWatch dashboard and two alarms (a third, on latency, once you set what
slow means). Nothing to switch on: the URL is a stack output (`DashboardUrl`) and
`python3 scripts/endpoint_info.py` prints it next to the endpoint and the key.

The top rows use metrics the load balancer and the Auto Scaling group already publish. The bottom row
comes from inside the engines through a sidecar (*Engine metrics* below).

Widgets are titled as questions:

| Widget | Read it for |
|---|---|
| *Is it slow?* (p50/p95/p99, with `latencyAlarmSeconds` drawn on it if set) | The only number with an SLA. The load balancer times a request until the engine starts answering: the whole answer for a non-streamed call, the first token for a streamed one. **A rising p99 against a flat p50 means queueing, not a slow model.** |
| *How hard is each engine working?* (requests/min per task, with `scalingRequestsPerTarget` drawn on it) | The leading indicator, and the metric the scaling policy compares against. |
| *Is the load balancer failing?* (ALB 5XX) | The *load balancer* failing: 503 = no healthy target, 504 = a request outran its 300 s idle timeout. |
| *Is CloudFront timing out?* (CloudFront 5xx rate) | A non-streamed answer that took longer than CloudFront's 120 s read timeout. Invisible to the load balancer, so it has its own widget. |
| *Are the engines erroring?* (target 5XX) | The *engine* answering with an error. A different problem from the row above, so a separate widget. |
| *Did an engine die mid-request?* (target connection errors) | A container that went away with a request in flight. |
| *Are the engines up?* (healthy / unhealthy targets) | Healthy targets falling while instances stay in service means engines are dying, not capacity going away. |
| *Did AWS give us the instances?* (in service vs desired) | Whether the ASG is still trying to grow. Capacity it cannot get looks like a persistent gap here. |
| *How much traffic is arriving?* (requests/min, fleet total) | Offered load, for correlating everything else against. |

Every alarm treats missing data as *not breaching*: an idle load balancer publishes nothing,
and an alarm in `INSUFFICIENT_DATA` every quiet hour is an alarm nobody reads.

| Alarm | Fires on | Why that delay |
|---|---|---|
| `<stack>-too-slow` (only if `latencyAlarmSeconds` is set) | p95 above it for 3 minutes | One minute over budget is what a task starting or draining looks like. A fleet that cannot grow faster than ~11 minutes gains nothing from being told sooner. |
| `<stack>-engines-unhealthy` | any unhealthy target for 15 minutes | A fresh instance is unhealthy for about 7 minutes while it pulls the image and loads weights (14 from launch to healthy). A 5-minute window fired on every first deploy. |
| `<stack>-load-balancer-erroring` | more than 10 ALB 5XX/min for 2 minutes | Not zero: a long prompt hitting the idle timeout produces a 504, and a deploy briefly has no healthy target. Sustained is what matters. |

Each description says what the alarm means and what to check.

Set `alarmTopicArn` to notify an SNS topic. None is created for you: a topic with no subscription
looks like a working notification and is not one.

### Engine metrics: what the load balancer cannot see

The numbers that say *why* the fleet is slow are on vLLM's Prometheus endpoint at `/metrics`. A
sidecar in every task scrapes eight of them over localhost and writes them to CloudWatch under the
namespace `<stack>/Engine`, the bottom two rows of the dashboard. Nothing to enable.

| Metric | Widget | What a bad reading means |
|---|---|---|
| `vllm:num_requests_waiting` | *Is work queueing inside the engines?* | **The one to watch.** Requests admitted but not yet running. A queue forming *is* saturation, and it forms before latency moves - the leading indicator `RequestCountPerTarget` only approximates. |
| `vllm:num_requests_running` | same widget | Sequences in the current batch. Pinned at `maxNumSeqs` means the batch ceiling is the limit; well below it while requests wait means KV cache is. |
| `vllm:kv_cache_usage_perc` | *Is the KV cache filling up?* | 0 to 1. Near 1 is the real ceiling on concurrency, and what `gpuMemoryUtilization: 0.95` buys more of. |
| `vllm:num_preemptions_total` | *Are engines redoing work?* | **Anything above zero is trouble.** The engine evicted a running sequence to free cache and will recompute it from scratch. Rising preemptions are the mechanism behind a collapsing p99. |

| `vllm:time_to_first_token_seconds` | *How long does a request take inside the engine?* | Average time to first token: queue wait plus prefill. Rising while *waiting* is zero means prefill itself is the cost (long prompts). |
| `vllm:e2e_request_latency_seconds` | same widget | Average whole-request time as the engine saw it. Compare with the load balancer's p50: a gap is the network path, not the engine. |
| `vllm:request_prompt_tokens` | *What shape are the requests being served?* | Average input tokens per request. The number `scalingRequestsPerTarget` is derived from; if it drifts, so should the threshold. |
| `vllm:request_generation_tokens` | same widget | Average output tokens per request. Pinned at a round number means callers hit their `max_output_tokens`. |

The last four are histograms in the engine, but the collector passes CloudWatch their sum and count,
not their buckets, so the dashboard shows **averages** over the requests completed that minute and no
percentiles. Latency percentiles come from the load balancer widget. (Tested: the exported record is
`{Sum, Count}` even with the collector's detailed-metrics option; the buckets do not survive the path.)

The metrics carry no per-task dimension; CloudWatch aggregates every engine's samples each minute.
**Maximum** is the busiest engine, **Average** the typical one. A large gap between them is an uneven
load balancer or one sick task, not a capacity problem.

On a real fleet, eight `g7e.2xlarge` engines, FP8, 1,000-token prompts, 190-token answers:

| Offered concurrency | req/s | input tok/s | p95 | running (busiest) | waiting (busiest) | KV cache (fullest) | preemptions |
|---|---|---|---|---|---|---|---|
| 768 | 121 | 113,000 | 6.6 s | 92 | 0 | 9% | 0 |
| 1,920 | 191 | 179,000 | 12.0 s | **244** | 13 | 20% | 0 |

Second row: throughput rose 58% and p95 doubled. *Running* is pressed against the `maxNumSeqs: 256`
batch ceiling, the KV cache is a fifth full, nothing was preempted. The fleet was out of **batch
slots**, not memory, so the lever is `maxNumSeqs`, not more instances or a smaller `maxModelLen`.

The first four or five scrapes in the sidecar's log fail with `Failed to scrape Prometheus endpoint`:
the collector starts in seconds, the engine takes minutes to load weights. It retries every 30 seconds.
Warnings that continue past startup mean the engine is not listening on its port.

How it is built:

* The unmodified public AWS Distro for OpenTelemetry collector image (`METRICS_SIDECAR_IMAGE` in
  `infra/serving_stack.py`), 256 MiB, non-essential so a metrics problem cannot stop inference. Its
  configuration is a ~30-line string in the same file, passed inline through the `AOT_CONFIG_CONTENT`
  environment variable.
* It scrapes `localhost:8080/metrics` every 30 seconds; `awsvpc` networking puts both containers in
  one network namespace.
* Metrics are written as embedded-metric-format records into the stack's own log group (stream
  `engine-metrics`), so retention and teardown are the stack's.
* To add a metric, append its name to `ENGINE_METRICS` and give it a widget. The endpoint exposes ~86
  families; the shortlist is short because **custom metrics are billed per name** (about $0.30 each
  per month) and the rest are derivable from these, duplicated by the load balancer, or histograms,
  which CloudWatch receives as Min/Max/Sum/Count and cannot turn into a p95. Time-to-first-token
  therefore stays on the load balancer's response time graph.

Not collected: **GPU utilisation** needs a host-level agent and NVIDIA's DCGM exporter, and on a
decode-heavy workload it reads near 100% while the memory bus is the limit. **Per-task engine
metrics** would name the sick engine but cost per task per metric; the task's own log stream already
does that.

For real percentiles of queue time, or to scale on `vllm:num_requests_waiting` directly, the same
sidecar can remote-write to Amazon Managed Service for Prometheus by swapping the `awsemf` exporter for
`prometheusremotewrite`. That adds a workspace and a Grafana.

### What happens when a container is overloaded

**vLLM does not shed load, and it has no request timeout.**

An arriving request is tokenised and put on a **waiting queue**. Each scheduler step admits waiting
requests as three limits allow: `maxNumSeqs`, `maxNumBatchedTokens`, and free KV cache blocks. No
queue-depth limit, no admission control, no deadline: a request waits as long as the client holds the
connection.

When the KV cache fills, the scheduler **preempts** a running sequence and later recomputes it from the
beginning. Under sustained overload that recomputation competes with new work, so throughput *falls*
as load rises: doubling offered concurrency from 768 to 1536 lowered aggregate throughput and blew p99
out.

The engine never returns "busy". A queued request ends only by:

- **CloudFront's 120 s read timeout, then the ALB's 300 s idle timeout.** Either returns a **504 while
  the engine is perfectly healthy**; neither applies to a streamed response, whose bytes reset both
  timers;
- **the client's own timeout**, the only backstop you control directly, so keep it shorter than your
  latency budget;
- **finishing**, eventually.

An overloaded fleet degrades silently until something external gives up. Size the *minimum* for steady
state; autoscaling takes ~11 minutes. Real load shedding has to go in front of the engine: a
concurrency limit at the client, or `maxNumSeqs` plus a short client timeout so over-limit work fails
fast.

---

## Interpreting your own measurements

Decode ceiling for your configuration:

```
ceiling (tokens/sec) = GPU bandwidth ÷ (activated weight bytes ÷ tensor-parallel degree)
```

`infra/hardware.py` has this as `decode_ceiling_tokens_per_sec`. For a mixture-of-experts model, use
*activated* parameters, not total: typically 10–20%, so such models decode far faster than their size
suggests.

**The ratio of measured to ceiling is diagnostic:**

| Measured ÷ ceiling | Meaning | What helps |
|---|---|---|
| **> 60%** | bandwidth-bound | a faster GPU, or fewer bytes per token (quantization) |
| **20–40%** | overhead-bound | lower tensor parallelism, fewer GPUs, larger batches |
| **< 20%** | something is wrong | check tensor parallelism and interconnect first |

Do this before buying hardware: a workload at 25% of its ceiling will not go faster on a card with
twice the bandwidth.
