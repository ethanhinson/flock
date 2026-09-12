// The two elementwise ops a Qwen3 layer needs, in one shader with two entry
// points so they share a pipeline layout and a bind group shape.
//
//   swiglu:  o[i] = silu(gate[i]) * up[i],  silu(v) = v * sigmoid(v)
//   add:     o[i] = a[i] + b[i]            (residual connections)
//
// This is where the ONNX Sigmoid/Mul pair and 11 of the Add nodes go. Both are
// pure memory-bound streaming, so there is nothing clever to do: one thread per
// element, coalesced.
//
// silu is written as v / (1 + exp(-v)) rather than v * (1/(1+exp(-v))) because
// that is one fewer rounding and matches the CPU reference. exp() itself is not
// bit-reproducible across implementations, so swiglu is validated to a few ULP
// rather than exactly; see test_ops.ts.

struct Dims { n: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= d.n) { return; }
  let g = a[i];
  o[i] = (g / (1.0 + exp(-g))) * b[i];
}

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= d.n) { return; }
  o[i] = a[i] + b[i];
}
