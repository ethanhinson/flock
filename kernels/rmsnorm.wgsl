// RMSNorm. Replaces the ONNX Pow/ReduceMean/Add/Sqrt/Div/Mul chain:
//
//   out[i] = x[i] / sqrt(mean(x^2) + eps) * g[i]
//
// Qwen3 uses RMSNorm, not LayerNorm: there is no mean subtraction and no bias,
// only a scale by the root-mean-square and a learned per-channel gain. The gain
// tensor is stored f32 in the GGUF (attn_norm.weight and friends are dtype 0).
//
// One workgroup per normalized vector, because a mean is a reduction and a
// reduction across workgroups would need a second dispatch. Every thread sums a
// strided slice of x^2, the partials are tree-reduced in workgroup memory, and
// then the same threads rescale their slice with the result.
//
// `n_vecs` lets one dispatch normalize many vectors: the hidden states of a
// whole prompt, or -- and this is what Qwen3 needs -- every head of q and k
// independently. q_norm/k_norm in Qwen3 apply per head over head_dim=128 rather
// than over the whole 2048-wide projection, which is unusual and is the reason
// this kernel is written around "n_vecs vectors of width n" instead of "one
// vector".
//
// The mean is computed in f32 to match the CPU reference bit-for-bit. A vec4
// version would be faster but changes the summation order, and at eps=1e-6 the
// normalizer is sensitive enough that it is worth keeping the orders identical
// while the engine is still being proven correct.

struct Dims {
  n: u32,        // width of each vector (head_dim, or hidden size)
  n_vecs: u32,   // how many vectors to normalize
  eps: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       g: array<f32>;   // gain, length n
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

const WG: u32 = 128u;
var<workgroup> part: array<f32, WG>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let vec_id = wg.x;
  let t = lid.x;
  let base = vec_id * d.n;

  // Sum of squares. Each thread walks a stride-WG slice so the loads coalesce.
  var acc = 0.0;
  if (vec_id < d.n_vecs) {
    for (var i = t; i < d.n; i = i + WG) {
      let v = x[base + i];
      acc = acc + v * v;
    }
  }
  part[t] = acc;
  workgroupBarrier();
  var stride = WG / 2u;
  while (stride > 0u) {
    if (t < stride) { part[t] = part[t] + part[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  // part[0] now holds the full sum for every thread to read. The barrier above
  // already published it; reading it in all threads needs no further sync
  // because nothing writes `part` again.
  let scale = 1.0 / sqrt(part[0] / f32(d.n) + d.eps);

  if (vec_id < d.n_vecs) {
    for (var i = t; i < d.n; i = i + WG) {
      o[base + i] = x[base + i] * scale * g[i];
    }
  }
}
