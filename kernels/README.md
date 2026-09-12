# Q8_0 WebGPU kernel

`q8_matmul.wgsl` multiplies a Q8_0-quantized weight matrix by an f32 vector,
dequantizing each 32-weight block as it goes. This is the piece that would let
flock consume GGUF weights directly — no ONNX export, no host build step.

```bash
deno run --unstable-webgpu --allow-all kernels/test_q8.ts
```

## Status

The kernel is correct. Validated against a CPU reference at three sizes, and
against **real Q8_0 weights range-fetched from HuggingFace**
(`blk.24.attn_k.weight`, 1024×1024):

```
64x32      max rel err 2.4e-6
128x1024   max rel err 5.9e-6
1024x1024  max rel err 8.3e-5
```

It is one kernel, not an engine. A working bird also needs attention, RMSNorm,
RoPE, SwiGLU, and the graph plumbing to run them in sequence.

## Two things worth knowing

**f32 accumulation dominates the error.** A GPU/CPU gap of ~5e-4 on real weights
looks alarming until you compare against a *strict f32* CPU reference instead of
an f64 one: f32 summation alone accounts for 1.2e-4 on the same data. Measure
like-for-like or you will chase a phantom.

**f16 encoders must round, not truncate.** `man >> 13` looks like a reasonable
way to drop mantissa bits and costs ~3 bits of an 11-bit mantissa:

```
truncating:  5.8e-2 worst-case relative error
rounding:    4.9e-4  (f16's theoretical limit)
```

This bug was in `web/js/wire.mjs` too, so every activation flock sent over the
network was 100x less accurate than f16 allows. Fixed there as well.
