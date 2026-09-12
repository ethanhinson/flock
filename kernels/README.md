# WGSL kernels: a GGUF transformer layer, no ONNX

Six WebGPU compute kernels and a `layerForward()` that composes them into one
Qwen3 transformer layer, consuming Q8_0 weights straight out of a GGUF file. No
ONNX export, no Python build step, no pre-built artifacts.

```bash
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_q8.ts     # reference matvec
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_coop.ts   # optimized Q8_0 + Q4_0
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_ops.ts    # rmsnorm, rope, swiglu, add, attention
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_layer.ts  # whole layer vs CPU reference
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_onnx.ts   # whole layer vs web/shard0.onnx
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/bench.ts       # matvec benchmark
deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/bench_layer.ts # layer benchmark
```

79 assertions, all passing, stable across repeated runs.

## Status

| kernel | file | validated against |
|---|---|---|
| Q8_0 matvec, reference | `q8_matmul.wgsl` | strict-f32 CPU, **bit-exact** |
| Q8_0 matvec, cooperative | `q8_coop.wgsl` | strict-f32 CPU, **bit-exact** |
| Q4_0 matvec, cooperative | `q4_coop.wgsl` | strict-f32 CPU, **bit-exact** |
| RMSNorm (incl. per-head) | `rmsnorm.wgsl` | strict-f32 CPU, **bit-exact** |
| RoPE | `rope.wgsl` | strict-f32 CPU, ~1.5 ULP |
| SwiGLU + add | `elementwise.wgsl` | add bit-exact, swiglu ~2.5 ULP |
| Causal GQA attention | `attention.wgsl` | strict-f32 CPU, ~1.5e-5 |
| Whole layer | `layer.ts` | CPU reference layer, 1-2 ULP; **ONNX shard, cosine 0.9997** |

`layerForward()` is real: two chained WGSL layers reproduce `web/shard0.onnx`
(Qwen3 layers 24-25) at cosine similarity 0.9997, with the remaining gap
accounted for by Q8_0 quantization (measured, see below). **That is the point
at which ONNX can be deleted.**

Not done: no prompt prefill (decode only, one token per step), no embedding or
LM head, no multi-shard chaining, and attention caps at `maxKeys` (2048 by
default) because the softmax is not streaming.

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

## The measurement traps

Four of these cost real time. They are documented because each one produced
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
- **RoPE pairs adjacent elements** (GGUF/llama.cpp NORM), not halves as
  HuggingFace does. Both are self-consistent rotations, so the wrong one degrades
  the model *silently*. Pinned by an exact assertion on constructed input.
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
layer.ts          Layer: encode() / forward(), KV cache, bind-group cache
layer_ref.ts      CPU reference layer, composed from the per-op references
real_weights.ts   range-fetch real GGUF tensors and layers, cached in .cache/
dequant.ts        Q8_0 -> f32, and the quantization-term estimator
onnx_truth.mjs    runs web/shard0.onnx under Node (native addon, cannot use Deno)
deno.json         import map so the tests can reach @huggingface/gguf
```
