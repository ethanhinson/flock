# WGSL kernels: Qwen3 end to end, no ONNX

Nine WebGPU compute kernels and a `Model` that composes them into the whole of
Qwen3-0.6B — token ids in, next token id out — consuming Q8_0 weights straight
out of a GGUF file. No ONNX export, no Python build step, no pre-built artifacts.

**This is now flock's only inference path.** The coordinator runs `Model` with a
`cut` (layers 0..cut-1, plus the embedding, output_norm and the tied head); a bird
runs `Layer.fromBuffers` over weights streamed straight from HuggingFace. ONNX
remains only as the reference below.

```bash
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_q8.ts      # reference matvec
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_coop.ts    # optimized Q8_0 + Q4_0
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_ops.ts     # rmsnorm, rope, swiglu, add, attention
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_head.ts    # embedding, argmax, prefill attention
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_layer.ts   # whole layer vs CPU reference
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_prefill.ts # prefill vs N decode steps
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_onnx.ts    # one layer vs web/shard0.onnx
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_model.ts   # WHOLE MODEL vs the ONNX pipeline
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_frombuffers.ts # the bird's constructor == the tested one
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_split.ts  # sharded across a cut == unsplit
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_shard.ts      # TENSOR-PARALLEL matvec, both directions
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_shard_head.ts # tensor-parallel LM head
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_shard_mem.ts  # per-shard buffers under a limit
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/bench.ts        # matvec benchmark
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/bench_layer.ts  # layer benchmark
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/bench_model.ts  # embedding, head, prefill, token
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/bench_shard.ts  # sharding overhead, incl. the reduction
```

189 assertions in the pre-tensor-parallel suite plus 127 in the three shard
suites, all passing, stable across repeated runs.

## Status

**The engine generates the same text as ONNX, token for token.**

```
prompt  "Capital of France?"
WGSL    [785,6722,315,9625,374,3070,59604,334,13]   "The capital of France is **Paris**."
ONNX    [785,6722,315,9625,374,3070,59604,334,13]   identical, both stop at EOS

prompt  "What is 2 + 2?"
WGSL    [17,488,220,17,284,220,19,13]               "2 + 2 = 4."
ONNX    [17,488,220,17,284,220,19,13]               identical
```

ONNX here is the full five-graph pipeline flock ships — `embed.onnx`,
`layers.onnx` (layers 0-23), `shard0.onnx`, `shard1.onnx`, `head.onnx` — under
onnxruntime-node, with the same tokenizer and chat template.

| kernel | file | validated against |
|---|---|---|
| Q8_0 matvec, reference | `q8_matmul.wgsl` | strict-f32 CPU, **bit-exact** |
| Q8_0 matvec, cooperative (+batched) | `q8_coop.wgsl` | strict-f32 CPU, **bit-exact** |
| Q4_0 matvec, cooperative | `q4_coop.wgsl` | strict-f32 CPU, **bit-exact** |
| RMSNorm (incl. per-head) | `rmsnorm.wgsl` | strict-f32 CPU, **bit-exact** |
| RoPE (both pairings) | `rope.wgsl` | strict-f32 CPU, ~1.5 ULP |
| SwiGLU + add | `elementwise.wgsl` | add bit-exact, swiglu ~2.5 ULP |
| Causal GQA attention, decode | `attention.wgsl` | strict-f32 CPU, ~1.5e-5 |
| Causal GQA attention, prefill | `attention_prefill.wgsl` | streaming-softmax CPU ref, ~4e-7 |
| Token embedding gather | `embed.wgsl` | on-disk-layout decoder, **bit-exact**, real 165 MB tensor |
| Argmax reduction | `argmax.wgsl` | host scan, **exact**, incl. the tie rule |
| Whole layer, decode | `layer.ts` | CPU reference, 1-2 ULP; ONNX shard cosine 0.9997 |
| Whole layer, prefill | `layer.ts` | CPU ref 2e-7; **bit-identical to N decode steps** |
| Whole model | `model.ts` | **same token ids as the ONNX pipeline** |
| Whole model, sharded at a cut | `model.ts` | **bit-identical to the unsplit pass** |
| Matvec, tensor-parallel row-wise | `q8_shard.wgsl` | unsharded kernel, **bit-identical**, N = 1-4 |
| Matvec, tensor-parallel column-wise | `q8_shard.wgsl` | FMA-modelled sharded-order CPU ref, **bit-exact** |
| Partial-sum reduction | `shard_reduce.wgsl` | strict-f32 serial CPU ref, **bit-exact** |
| LM head, tensor-parallel | `shard.ts` | unsharded engine, **all 151936 logits bit-identical** |
| Layer from pre-built GPU buffers | `layer.ts` | **bit-identical to `Layer.create`** |

The strongest single piece of evidence is not a tolerance. `encodePrefill(N)` is
**bit-identical** (`0.0e+0`) to N sequential decode steps — hidden states and KV
cache — and the two paths share no attention code: one-pass versus streaming
softmax, different kernels, different uniforms, different dispatch shapes.

### Where long generations diverge, and why

A 60-token generation does eventually disagree, and the divergence is measured
rather than assumed. For `"Explain in three sentences why the sky is blue."` the
first difference is at generated token **30**:

```
WGSL picks 4803   logit 24.57773
ONNX picks 12203  logit 24.48939 under WGSL
gap/scale 3.59e-3
```

Three things establish this is quantization, not a bug, and all three are
assertions in `test_model.ts`:

- The gap is **smaller than the quantization difference**. The hidden state
  feeding the projection differs from ONNX's by 1.22e-2 relative (Q8_0 weights
  versus f32), which is 3.4x larger than the gap. Neither engine is "right".
- ONNX's token is WGSL's **rank #2 of 151936**. A wiring bug scatters the
  reference's choice into the tail; it does not leave it second.
- It happens at the run's **tightest margin**. Over 66 steps the margin/scale
  distribution is min 3.59e-3, median 8.94e-2, max 4.27e-1, and only 4 of 66
  steps are under 1e-2. Agreement lasts 30 tokens because nearly every step is
  decided by a margin an order of magnitude clear of the noise.

### Not done

- **No sampling.** Greedy decode only.
- **`maxKeys` is 2048** for decode, because `attention.wgsl`'s softmax is not
  streaming and its workgroup array is an occupancy cliff (see below).
  `attention_prefill.wgsl` has no such cap — it is validated at pos0=2100 — so
  the ceiling is the decode kernel's alone, and lifting it means giving decode
  the streaming softmax too.
- **No tiled GEMM for prefill.** The batched matvec saves N-1 dispatches per
  projection but re-reads the weights per token, so it does not reuse weight
  loads across the batch. Prefill still reaches 501 tok/s; a real tile would
  beat that and the amount is unmeasured.
- **The 165 MB tied matrix needs raised device limits.** `getDevice()` requests
  the adapter's real `maxStorageBufferBindingSize`; the 128 MiB default is too
  small and the failure is silent (see below).
- **`unpack4xI8` is still unbenchmarkable here** — Deno's wgpu does not
  implement it, so the fallback path is what all of these numbers describe.
- **Tensor parallelism is validated but not wired into `Model`.** `shard.ts`
  holds N shards of one tensor and `ShardedHead` runs a sharded LM head
  correctly, but `Model` still binds `token_embd` as one buffer. Wiring it in is
  a `Model` change, and the coordinator/mesh side that would place shards on
  different devices is owned elsewhere. See "Tensor parallelism" below.

## Tensor parallelism: splitting ONE tensor

Layer-range splitting (`test_split.ts`) has a hard ceiling: **no number of
devices makes a single tensor smaller.** Qwen3-14B Q4_K_M's `output.weight` is
**638 MB as one tensor** with `tie_word_embeddings: False`, so it cannot be
shared with the embedding, and WebGPU's default `maxStorageBufferBindingSize` is
**128 MiB**. This adapter grants 4295 MB, which is why the 0.6B model works here
at all; a phone does not.

`q8_shard.wgsl` is `q8_coop.wgsl`'s inner loop plus two uniforms (`out_off`,
`col_off`). Both split directions are built, because "one is better" is not a
finding unless the other exists and was measured.

| | row-wise (output-split) | column-wise (input-split) |
|---|---|---|
| shard owns | rows `[r0, r1)` | cols `[c0, c1)` of every row |
| needs | the WHOLE `x` | only `x[c0:c1]` |
| produces | `y[r0:r1]`, concatenates | a PARTIAL `y`, must be summed |
| vs unsharded | **bit-identical** | reassociated, 1-3 ULP |
| the slice is | a contiguous byte range | strided, needs a gather |
| argmax | max of per-shard pairs, free | needs the full reduction first |

**Row-wise was chosen for the LM head.** Row `r`'s 64 lanes walk the same
`cols/32` blocks in the same order whether the shard holds 4 rows or 151936 —
`row` is an address, not an operand — so the result is bit-identical, asserted
as `=== 0` at N = 1-4 including non-divisible splits. Column-wise **cannot** be:
N shards partition each row's blocks before the lanes see them, so f32 addition
reassociates. That is asserted honestly instead — bit-identical to an
FMA-modelled strict-f32 reference **of the sharded order**, which is strictly
stronger than a tolerance against the unsharded answer, because a 1e-6 tolerance
would not notice one wrong block out of 32.

The measured residual of column-wise against unsharded is **7.4e-8 to 1.7e-7**
of the output scale, across every shape and shard count. For scale: Q8_0
quantization itself costs **1.34e-2** (measured, `test_onnx.ts`), so
reassociation is five orders of magnitude smaller than the error the model
already carries.

### Memory, which is the entire point

| tensor | unsharded | 64MiB | 128MiB | 256MiB | 1GiB |
|---|---|---|---|---|---|
| Qwen3-0.6B `token_embd` (Q8_0, tied) | 155.6 MB | 3 | 2 | 1 | 1 |
| Qwen3-14B `output.weight` (Q6_K) | 638.1 MB | 10 | 5 | 3 | 1 |
| a 248320-row head (Q8_0, 5120 wide) | 1271.4 MB | 19 | 10 | 5 | 2 |

Shard counts needed to fit one binding. **The 638 MB tensor split 8 ways is
79.8 MB per binding**, under 128 MiB. Every reported minimum is verified to fit
*and* one fewer to not fit. The real 155.6 MB tied tensor is uploaded as 5 and 8
shards under a simulated 32 MiB limit and the answer stays bit-identical.

The limit applies **per binding**, not per shard: `qs` and `scales` are separate
bindings, so a shard with 130 MB of `qs` and 9 MB of `scales` fails a 128 MiB
limit on `qs` alone. The check is arithmetic and happens *before* upload, because
an oversized binding is a validation error that makes a kernel write zeros rather
than throwing (correctness trap 3).

Capacity relief is the same in both directions — `rows * cols / N` either way —
so **memory does not choose the direction.** Bit-exactness, contiguity and wire
traffic do.

### What crosses the wire

Per call at N=4, computed from the shapes:

| | row-wise | column-wise | ratio |
|---|---|---|---|
| LM head 151936x1024 | 610 KB | 2378 KB | 3.9x |
| LM head 248320x5120 | 1050 KB | 3900 KB | 3.7x |
| ffn_down 1024x3072 | 52 KB | 28 KB | **0.5x** |

Row-wise broadcasts `cols` floats of `x` and collects `rows/N` back per shard;
column-wise sends `cols/N` and collects a **full-length** partial back from every
shard. An LM head's output is the whole vocabulary, so column-wise loses badly
there — but `ffn_down` is wider than it is tall and column-wise wins. **The
direction is a per-tensor choice, not a global one.**

### What it costs, on ONE device

`bench_shard.ts`. All shards run on the same GPU in the same queue, so **nothing
here is parallel** — this measures the overhead of splitting, not the speedup of
N devices, which this machine cannot demonstrate. Sharding on one device is
expected to be slower; it buys capacity, exactly like pipeline parallelism.
Map fence 24.25 ms subtracted; the per-dispatch floor is **18.46 us**.

The real tied LM head, 151936 x 1024, us per call:

| dir | N | shards | reduce | total | vs N=1 | GB/s | largest binding |
|---|---|---|---|---|---|---|---|
| row | 1 | 732.3 | — | 732.3 | 1.00x | 226 | 155.6 MB |
| row | 2 | 754.6 | — | 754.6 | **1.03x** | 219 | 77.8 MB |
| row | 4 | 797.1 | — | 797.1 | **1.09x** | 207 | 38.9 MB |
| row | 8 | 873.8 | — | 873.8 | **1.19x** | 189 | 19.4 MB |
| col | 1 | 733.3 | 38.7 | 772.0 | 1.00x | 214 | 155.6 MB |
| col | 2 | 1590.4 | 23.2 | 1613.6 | **2.09x** | 102 | 77.8 MB |
| col | 4 | 3158.2 | 30.0 | 3188.2 | **4.13x** | 52 | 38.9 MB |
| col | 8 | 6322.7 | 31.2 | 6353.9 | **8.23x** | 26 | 19.4 MB |

**Row-wise sharding of the head is nearly free**, and the overhead is accounted
for rather than asserted: +64.7 us at N=4 against `3 x 18.46 = 55.4 us` of extra
dispatch floor. An 8-way split of the largest tensor in the model — the one that
does not fit a default-limits device at all — costs **19%**.

**Column-wise costs a factor of N, and that is not dispatch overhead.** At N=8 it
is 5582 us slower than unsharded, of which dispatches are 129 us and the
reduction 31 us; **~5400 us is unexplained by either.** The cause is lane
starvation and it is structural: `q8_shard.wgsl` gives every row `LANES = 64`
threads walking that row's `n_cols/32` blocks, and a column shard shrinks the
block count without shrinking the lane count. At `cols = 1024` a shard has 32
blocks unsharded (half the lanes already idle) and **4 blocks at N=8, so 60 of 64
lanes contribute nothing** while still paying a full workgroup and a full 64-wide
tree reduction. Row-wise keeps every row's full column count, so every lane stays
fed — the same property that makes it bit-identical, showing up as a performance
result too.

A column-wise kernel that scaled `LANES` down with `n_cols` and `ROWS_PER_WG` up
to keep the workgroup full would recover most of this. It is **not written**: the
head is row-wise, where the problem does not arise.

At layer-sized shapes both directions are ~`N x` the unsharded cost, because a
1024x1024 matvec is 24 us against an 18.5 us dispatch floor — it *is* the floor,
so it cannot absorb `N-1` more of them. The reduce dispatch is 18.6-19.7 us at
every shape and shard count, i.e. **the reduction's arithmetic is free and only
its dispatch costs anything.** Sharding a layer projection is therefore never
worth it on capacity grounds alone; the head is the only tensor that needs it.

## Benchmarks

Apple Silicon, Deno 2.1.2 (wgpu/Metal). Measured with a map-readback fence and
2000 dispatches per batch — see "the measurement traps" below, because the first
two attempts at these numbers were wrong by 20x and 4400 GFLOP/s respectively.

### Matvec, per call at Qwen3-0.6B shapes

| shape | reference | cooperative | speedup |
|---|---|---|---|
| 1024x1024 (attn_k/v) | 138.7 us | 24.7 us | **5.6x** |
| 2048x1024 (attn_q/out) | 137.2 us | 30.5 us | **4.5x** |
| 3072x1024 (ffn_gate/up) | 139.0 us | 36.9 us | **3.8x** |
| 1024x3072 (ffn_down) | 369.8 us | 30.0 us | **12.3x** |

Q4_0 lands within noise of Q8_0 at every shape (23.9-36.4 us): the kernels are
not bandwidth-bound at these sizes, so halving the weight bytes buys nothing.

### What each optimization actually bought

| change | measured effect |
|---|---|
| separate qs/scales buffers + vec4 `dot()` + workgroup reduction | 3.8-12.3x on matvec |
| sizing attention's workgroup array to the cache, not 16 KB | **15x** on that dispatch (328 us -> 21 us) |
| caching bind groups instead of rebuilding per step | ~4 ms/layer of 5.4 |
| not reading back per layer (`encode()` vs `forward()`) | ~27 ms/layer |
| sharing one command buffer across layers | ~0.07 ms/layer |
| `unpack2x16float` vs a hand-written `half_to_f32` | **nothing, ~1%** (negative result) |
| `unpack4xI8` vs shift/mask | **could not measure** — not implemented by Deno's wgpu |

### Layer decode step

| how the host batches | per layer |
|---|---|
| `forward()` — own command buffer, own readback | 28.9 ms |
| `encode()` x28 — own command buffer, one readback | 1.46 ms (19.7x) |
| `encode()` x28 — one command buffer, one readback | **1.39 ms (20.8x)** |
| floor: this layer's 16 dispatches, timed individually | ~0.72 ms |
| the 7 matvecs alone | 0.21 ms |

A 28-layer pass projects to **25.7 tok/s** batched, versus 1.24 tok/s with a
readback and a submit per layer. The arithmetic is ~15% of a batched step; the
rest is per-dispatch overhead, which is why attention is one fused dispatch.

### Embedding, LM head, argmax

`bench_model.ts`. Batched figures are per dispatch with the fence's ~26 ms removed.

| op | shape | batched | note |
|---|---|---|---|
| embedding gather | 1 token | 18.5 us | |
| embedding gather | 16 tokens | 18.7 us | |
| embedding gather | 64 tokens | 18.3 us | |
| embedding gather | 256 tokens | 18.4 us | |
| LM head matvec | 151936x1024 | **785 us** | 396 GFLOP/s, 211 GB/s, 37984 workgroups |
| same kernel, layer shape | 3072x1024 | 38 us | 88 GB/s |
| argmax pass1 + pass2 | 151936 | 31 + 20 us | |

The embedding is **flat from 1 to 256 tokens** — 256x the work for the same time,
which is what a dispatch that is pure launch overhead looks like. A decode step's
gather is 5 KB and costs the same as an empty kernel, so there is nothing to win.

The LM head is the largest single op in the model: ~21x the biggest matvec inside
a layer, and 4% of a 28-layer step's total arithmetic. **It is not a bad dispatch
shape** — that was checked rather than assumed, by running the same kernel at a
layer-sized shape with the same per-byte accounting. The head sustains 211 GB/s
against the layer shape's 88, i.e. **2.4x better** per byte: 3.3 MB is too little
work to saturate the GPU and 165 MB is not. So the head is already doing the best
this kernel does, and a faster one needs fewer *bytes* (Q4 on this one tensor) or
fewer *rows*, not a better dispatch. Whether 211 GB/s is near this machine's
ceiling was not measured, so no absolute headroom claim is made.

**Argmax on the GPU is a negative result.** It was expected to beat reading
151936 floats back; measured, it is **1.04x**:

| | one call |
|---|---|
| GPU reduction + 4-byte readback | 26.3 ms |
| copy 0.61 MB back + scan in JS | 27.4 ms |

Not because the reduction is slow — it is 50 us of kernel — but because *both*
paths pay one map readback, and that readback is ~26 ms regardless of size on
this backend, swamping both the copy and the kernel. The reduction is still the
right choice (never slower, keeps 0.6 MB off the bus, and wins properly on a
backend whose readback scales with size), but 1.04x is the honest number.

### Prefill and a whole token

| | per token | tok/s |
|---|---|---|
| prefill, 1 token | 42.1 ms | 24 |
| prefill, 8 tokens | 7.71 ms | 130 |
| prefill, 16 tokens | 4.54 ms | 220 |
| prefill, 64 tokens | 2.51 ms | 399 |
| prefill, 128 tokens | 2.10 ms | 477 |
| prefill, 256 tokens | **2.00 ms** | **501** |
| decode at 16 keys | 43.7 ms | 22.9 |
| decode at 128 keys | 44.5 ms | 22.5 |
| decode at 512 keys | 57.7 ms | 17.3 |

**Prefill is 19.8x faster than decoding the same tokens**: 128 tokens take 279 ms
as one prefill and 5520 ms as 128 decode steps. That ratio is the entire
justification for prefill existing — a decode step's cost is dominated by
per-dispatch and per-submit overhead, and prefill pays it once for N tokens.

A decode token is dominated by the map readback, not by the model:

| K decode steps behind one fence | per token | tok/s |
|---|---|---|
| K=1 (what generation actually pays) | 43.1 ms | 23 |
| K=4 | 21.5 ms | 47 |
| K=16 | 16.8 ms | 59 |
| K=64 | 15.8 ms | 63 |

The gap between K=1 and K=64 *is* the readback, ~26 ms. Greedy decode cannot
amortize it — it needs each id on the host before choosing the next input — so
**23 tok/s is the honest generation figure** and 63 tok/s is the GPU work
underneath it. `stepsAmortized` exists only to measure this and deliberately
computes the wrong tokens.

## The correctness traps

These are the ones that produce a **working model that is wrong**, which is worse
than a broken one. Each was found by measurement, and each is now an assertion.

**1. RoPE's pairing convention depends on the WEIGHTS, not the file format.**
There are two conventions: GGUF/llama.cpp NORM pairs adjacent elements
`(0,1),(2,3),...`; HuggingFace NEOX pairs halves `(0,64),(1,65),...`. Both are
self-consistent rotations, so the wrong one does not crash and does not look
wrong — it produces fluent, wrong text.

The intuitive inference — "weights came out of a GGUF, so use NORM" — is false
here. llama.cpp's converter permutes q/k for Llama-style models so that NORM
reproduces HF; this Qwen3 GGUF was **not** permuted, so it still needs HF
pairing. Measured over 24 layers on a 16-token prompt:

```
              token 0          token 15        whole tensor
  NORM      cosine 0.999997   cosine 0.557    cosine 0.9967
  NEOX      cosine 0.999997   cosine 0.9997   cosine 0.999989
```

The **failure signature is the useful part**: token 0 was essentially exact while
every later token degraded monotonically with position. Position 0 has a zero
rotation angle under either convention, so it is the one token both agree on. A
per-token cosine that is perfect at 0 and decays with position indicts the
rotation, not the arithmetic — which is why the magnitudes matched all along
(|max| 8152 against ONNX's 8150).

Two existing tests could not see this bug, and the reason generalizes:

- `test_onnx.ts` diffs at **position 0 only**, where the conventions are
  identical. It passed at cosine 0.999814 before the fix and 0.999735 after.
- `test_ops.ts` asserted NORM deliberately, against a reference that stated the
  same choice — so kernel and reference agreed with each other about the wrong
  thing. **A consistency test between a kernel and its reference cannot catch a
  shared assumption.** Both conventions are now built and tested, with a planted
  input that asserts element 0's partner *is* the expected one *and* that the
  other convention's partner is untouched.

**2. `queue.writeBuffer` ignores a TypedArray view's offset** on Deno 2.1.2's
wgpu, uploading the whole underlying ArrayBuffer from 0. Writing
`Float32Array([1..8]).subarray(4,8)` into a 4-float buffer reports "Copy of
0..32 would end up overrunning the bounds of the Destination buffer of size 16".

The bounds error only appears when the destination is too small. With buffers
sized for `maxPrefill` it **silently lands the wrong bytes**: feeding decode steps
as `hidden.subarray(i * H, ...)` made every step consume token 0, so the KV cache
filled with one key repeated — and the failing test blamed the prefill, which was
correct. `layer.ts` `writeView()` passes `(buffer, byteOffset, byteLength)`
explicitly, which *is* honoured.

**3. The default `maxStorageBufferBindingSize` is too small for a tied LM head,
and overflowing it is silent.** The split `token_embd` is 155.6 MB; the default
limit is 128 MiB. An oversized bind group is a *validation error*, not an
exception at the call site, so the symptom was a kernel writing zeros. This
adapter allows 4 GiB, so `getDevice()` asks for the adapter's real limits. Note
the Deno quirk the fallback has to handle: a **failed** `requestDevice` still
invalidates the adapter, so the retry must request a fresh one.

**4. `output_norm` is applied by the last SHARD, not by `head.onnx`.**
`flock_export_coordinator.py` builds `Head(model.model.norm, model.lm_head)`
whose forward is `self.head(hidden).argmax(-1)` — it takes the norm in its
constructor and never calls it. `flock_export.py` applies it when `is_last`.
Reading `head.onnx` as "norm then project" and applying `output_norm` twice on
the WGSL side would be plausible and wrong.

## The measurement traps

Eight of these cost real time. They are documented because each one produced
confident, plausible, wrong numbers.

**1. `onSubmittedWorkDone()` does not wait.** On Deno 2.1.2's wgpu it returns
before the GPU has finished. Calibrated against a kernel with a deliberately long
dependent loop:

```
10,000,000 iterations   fence says 0.13 ms   map readback says 476 ms
```

Timing anything with that fence measures how fast the host fills a command
buffer. It produced a benchmark table claiming **4400 GFLOP/s** for a 1024-column
Q8_0 matvec — several times this machine's memory bandwidth, and "faster" for 3x
the work. Only `mapAsync` on a buffer the pass wrote is a real barrier.

**2. Dispatch cost has a hard floor.** An empty one-workgroup kernel costs 17.6 us
— the same as a 3072x1024 matvec. So a single matvec at these shapes is *entirely*
queue overhead, and per-shape "1.08x" speedups measured that way are noise. Real
numbers need many dispatches behind one fence, with the fence's own ~24 ms
subtracted.

**3. Comparing f32 against f64 makes a correct kernel look broken.** This one was
already in the README and was still not fixed in the test: the committed harness
compared against an f64 accumulator and *failed* the correct kernel at 1024x1024
with 1.5e-4. f32 summation of 1024 terms loses that much by itself.

But a naive strict-f32 reference is **worse** than the f64 one (1.4e-4 vs
2.8e-5), because the GPU contracts every multiply-add into one FMA with a single
rounding. Model that and agreement becomes *exact*:

```
                            vs f64    vs naive f32    vs FMA-modelled f32
1024x1024 Q8_0 matvec       6.2e-5       1.4e-4              0.0
```

Same for `1/sqrt(v)`: Metal computes it with one rounding, and a reference that
rounds the `sqrt` and then the divide is 1.9e-7 off at every RMSNorm size. Match
the hardware's *rounding count*, not just its arithmetic, and the tolerance
argument disappears — `lib.ts` and `ops_ref.ts` are built on an exact `fma32`.

**4. Per-element relative error is the wrong metric for anything with
cancellation.** RoPE reported 7.8e-5 while its max *absolute* error over the same
tensor was 1.79e-7, uniformly — one output had landed near zero through
cancellation. Matvec cross-checks flickered between 3.9e-5 and 2.2e-4 on
different random draws from an identical kernel. Sums of signed terms get judged
by `absErrScaled` (max absolute error over the data's scale) instead.

**5. The readback is not a constant you can difference away.** Splitting a decode
token into "layers" and "head" by timing `step()` against `hiddenState()` and
subtracting reported **"the head is 0% of a token"**. Both calls *end* in the
~26 ms map readback, so the subtraction cancels the layers and the readback
together and leaves noise. This is trap 1 wearing a different hat: the readback is
not overhead around the measurement, it *is* the measurement. The split is now
made by amortizing K steps behind one fence — 43.1 ms/token at K=1 falling to
15.8 at K=64, and the gap is the readback.

**6. A signal cancelled down to the noise cannot be tested, at any threshold.**
Two attempts to assert that prefill applies RoPE per token both failed on a
*correct* kernel. Qwen3's `rope_base` is 1e6, so at positions 0-5 the rotation
angles are tiny:

```
positions 0..5, identical input tokens:     |positional signal|   ~1.0e-5
GPU vs CPU reference on those same outputs:  absolute error       ~1.0e-5
```

Both are one ULP of an output near 48. So "more than 90% of entries differ"
failed (86% is the correct answer) and comparing the GPU's positional delta
against the reference's gave a ratio of 0.8-1.2 — one ULP against one ULP. No
metric built at those positions can discriminate. The assertion now runs at
**pos0=900**, where the rotation is real: max |diff| 41.7 against |out| 43.2.
This is trap 4's deeper form — before choosing a tolerance, check that the
quantity has any signal-to-noise margin at all.

**7. Thousands of compute PASSES in one command buffer wedge the backend.**
Dispatches batch; passes do not. Benchmarking a sharded matvec by calling
`encode()` 2000 times behind one fence builds 2000 compute passes into one
command buffer, and the submit never completes — `mapAsync` never resolves, at
**0% CPU and 0.19 s of total CPU time**. That is indistinguishable from "the
benchmark is just slow" until you look at the process, which is why it cost a
full run. `bench_shard.ts` puts the repetitions inside ONE pass
(`dispatchShards`), which is also the right unit: a real forward pass encodes one
pass over many operations.

**8. An error whose magnitude does not move with N is not a sharding error.**
A sharded-head test failed by 5.6 at N = 2, 4 and 8 — the *same* number. The
cause was upstream of the split entirely: `logits(PROMPT)` then
`normedState(PROMPT)` on the same `Model` runs the prompt twice, at positions
16..31, with the first pass's keys still cached, producing a perfectly valid
hidden state for a different input. Worth stating as a rule because the shape of
the number localized the bug faster than reading the code did: sharding errors
scale with the shard count, and anything constant in N is in the input.

## Device-specific gotchas

- **`unpack4xI8` / `unpack4xU8` / `dot4I8Packed` are not implemented by Deno
  2.1.2's wgpu.** `unpack2x16float` and `extractBits` are. Both decode paths are
  written and feature-detected (`probeUnpack`), and both are validated to produce
  bit-identical output wherever they compile — but the hardware-unpack path
  **cannot be benchmarked on this machine**, so any claim about what it buys has
  to come from a browser.
- **`getCompilationInfo` does not exist** on Deno's shader modules. Feature
  probing has to rely on the error scope alone.
- **Workgroup memory is an occupancy cliff, not a gradient.** Declaring the full
  16 KB the spec guarantees made attention 15x slower than declaring 8 KB, at
  identical work. `maxKeys` defaults to 2048 for that reason.
- **In-place ops are a validation error, not a hazard.** WebGPU forbids binding
  one buffer as both `read` and `read_write` in a dispatch, so `norm(q -> q)`
  fails outright. RoPE is in place legally: it declares `x` as `read_write` and
  reads no other storage buffer.
- **Metal's `exp()` differs from V8's `Math.exp()` by up to 15 ULP** (1.0e-6
  relative) over `[-20, 0]`, the range a max-subtracted softmax feeds it. That is
  the floor for anything with a softmax or a sigmoid, and it is why attention's
  1.5e-5 is not a bug — a reference rewritten to mirror the kernel's exact
  reduction order gives the same number.
- **Metal's `rsqrt` is an approximation plus a refinement**, not a correctly
  rounded `1/sqrt`, so RMSNorm is 1 ULP off at some widths. Verified
  deterministic (0 drift over 6 runs), which is what distinguishes a rounding
  difference from a missing barrier; the determinism check is part of the suite.
- **f16 encoders must round, not truncate.** `man >> 13` costs ~3 bits of an
  11-bit mantissa: 5.8e-2 worst-case relative error versus 4.9e-4. This bug was
  in `web/js/wire.mjs`, so every activation flock sent over the network was 100x
  less accurate than f16 allows.

## Qwen3 specifics worth stating

Read from the GGUF metadata, not assumed: hidden 1024, ffn 3072, 16 query heads,
8 kv heads, head_dim 128, rope base 1e6, rms eps 1e-6.

- **q_norm and k_norm are per head**, over head_dim=128, not once over the whole
  projection. Qwen3 is unusual in having them at all. `rmsnorm.wgsl` takes an
  `n_vecs` parameter so one dispatch normalizes every head.
- **RoPE pairs HALVES here** (HuggingFace NEOX), not adjacent elements, even
  though the weights come from a GGUF — this GGUF was converted without
  llama.cpp's q/k permutation. See correctness trap 1: this was measured against
  ONNX, and an earlier version of this README asserted the opposite.
- **GGUF tensor shapes are `[in, out]`.** `attn_q.weight [1024, 2048]` is 2048
  rows of 1024 columns.
- **Tensor offsets are relative to `dataStart`**, not absolute. Reading them as
  absolute fetches wrong bytes that decode as valid-looking Q8_0.
- **Norm gains are not near 1.** `blk.24.attn_norm.weight` has mean 10.8 and max
  69, and q_norm has an interleaved pattern that looks exactly like a misaligned
  read. Both verified real by re-fetching each tensor by its own byte range. A
  sanity check built on the "gains near 1" intuition failed a correct loader.

## Layout

```
lib.ts            quantize, repack, strict-f32 references, WebGPU helpers, fma32
ops_ref.ts        CPU references for the non-matmul ops
head_ref.ts       CPU references for embedding, argmax, prefill attention
layer.ts          Layer: encode() / encodePrefill(), KV cache, bind-group cache
layer_ref.ts      CPU reference layer (decode and prefill), from the per-op refs
model.ts          Model: ids -> embedding -> 28 layers -> norm -> head -> argmax
real_weights.ts   range-fetch GGUF tensors/layers, or the whole model, in .cache/
shard.ts          ShardedMatvec / ShardedHead: one tensor across N shards
shard_ref.ts      shard planning, the slicers, and both directions' CPU refs
dequant.ts        Q8_0 -> f32, and the quantization-term estimator
onnx_truth.mjs    runs web/shard0.onnx under Node (native addon, cannot use Deno)
onnx_full.mjs     runs the WHOLE ONNX pipeline + tokenizer, for the text diff
deno.json         import map so the tests can reach @huggingface/gguf
```

### Why `model.ts` keeps everything on the GPU

A map readback is ~26 ms on this backend regardless of size, so the shape of the
API is forced: 28 layers chained through host round-trips would cost 28 x 26 ms
per token for numbers nobody needs yet. `step()` reads back **4 bytes** — the
token id — and everything else (the 151936 logits, every intermediate hidden
state, the KV cache) stays in device memory. One command buffer covers the
embedding and all 28 layers per prefill chunk.

The tied LM head needs no kernel of its own. `token_embd.weight` is 151936 rows
of 1024 columns in the GGUF, which is exactly the layout the matvec kernel already
wants, so the head is `q8_coop.wgsl` at a different shape and the embedding is a
gather over the same buffer. One 155.6 MB repacked upload serves both; a second
copy, or a dequantized one (622 MB), would buy nothing.
