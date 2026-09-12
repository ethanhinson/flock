// Causal grouped-query attention for ONE query position, fused into one
// dispatch. Replaces the ONNX MatMul/Div/Trilu/Softmax/MatMul chain.
//
//   scores[t] = dot(q[h], k[t, kvh]) / sqrt(head_dim)
//   probs     = softmax(scores)                       over t = 0..n_keys-1
//   out[h]    = sum_t probs[t] * v[t, kvh]
//
// Three things worth explaining:
//
// GQA. Qwen3-0.6B has 16 query heads and 8 kv heads, so query head h reads kv
// head h/2. The kv layout is [n_keys][n_kv_heads][head_dim] -- keys are the outer
// axis so appending a token to the cache is a contiguous write, which is what
// decode does every step.
//
// No mask tensor. The ONNX graph materializes a causal mask with Trilu because it
// processes a whole prompt at once. For a single query at position n_keys-1, "not
// attending to the future" is exactly the loop bound t < n_keys, so the mask is
// free and there is no mask buffer to allocate or upload.
//
// One workgroup per head, and the scores live in workgroup memory. That is what
// makes the fusion possible: softmax needs a max and a sum over ALL scores before
// any output element can be written, so splitting into separate dispatches would
// round-trip the scores through global memory twice.
//
// $MAX_KEYS is substituted by the host (lib.ts attnSource) rather than fixed at
// the 4096 the guaranteed 16 KB of workgroup storage allows, and that is a
// performance decision, not a tidiness one. Workgroup memory is a hard occupancy
// limit: a workgroup that declares 16 KB is the only one resident per core, so
// the GPU cannot hide any latency behind a second one. Measured on this machine,
// with MAX_KEYS fixed at 4096 this kernel cost 329 us at n_keys=1 -- twelve times
// every other dispatch in the layer, and almost all of it occupancy loss rather
// than work. Sizing the array to the cache the engine actually needs is worth
// recompiling the shader for.
//
// Beyond whatever $MAX_KEYS is set to, a flash-attention-style online softmax
// would be required. Deliberately not attempted yet: an unvalidated streaming
// softmax is a worse problem than a context limit.

struct Dims {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  n_keys: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;   // [n_heads][head_dim]
@group(0) @binding(1) var<storage, read>       k: array<f32>;   // [n_keys][n_kv_heads][head_dim]
@group(0) @binding(2) var<storage, read>       v: array<f32>;   // same layout as k
@group(0) @binding(3) var<storage, read_write> o: array<f32>;   // [n_heads][head_dim]
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 128u;
const MAX_KEYS: u32 = $MAX_KEYSu;

var<workgroup> scores: array<f32, MAX_KEYS>;
var<workgroup> part: array<f32, WG>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let t = lid.x;
  if (h >= d.n_heads) { return; }

  let group_size = d.n_heads / d.n_kv_heads;
  let kvh = h / group_size;
  let hd = d.head_dim;
  let scale = 1.0 / sqrt(f32(hd));
  let qbase = h * hd;

  // --- scores. Each thread handles a stride-WG set of keys. -----------------
  for (var key = t; key < d.n_keys; key = key + WG) {
    var acc = 0.0;
    let kbase = (key * d.n_kv_heads + kvh) * hd;
    for (var i = 0u; i < hd; i = i + 1u) {
      acc = acc + q[qbase + i] * k[kbase + i];
    }
    scores[key] = acc * scale;
  }
  workgroupBarrier();

  // --- max, for the numerically stable softmax ------------------------------
  var m = -3.4028235e38;
  for (var key = t; key < d.n_keys; key = key + WG) { m = max(m, scores[key]); }
  part[t] = m;
  workgroupBarrier();
  var stride = WG / 2u;
  while (stride > 0u) {
    if (t < stride) { part[t] = max(part[t], part[t + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let row_max = part[0];
  workgroupBarrier();   // part is about to be reused for the sum

  // --- exponentiate and sum ------------------------------------------------
  var s = 0.0;
  for (var key = t; key < d.n_keys; key = key + WG) {
    let e = exp(scores[key] - row_max);
    scores[key] = e;
    s = s + e;
  }
  part[t] = s;
  workgroupBarrier();
  stride = WG / 2u;
  while (stride > 0u) {
    if (t < stride) { part[t] = part[t] + part[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let inv = 1.0 / part[0];

  // --- weighted sum of v ---------------------------------------------------
  // Each thread owns a stride-WG set of output channels and walks every key,
  // so nothing needs reducing: one accumulator per output element.
  for (var i = t; i < hd; i = i + WG) {
    var acc = 0.0;
    for (var key = 0u; key < d.n_keys; key = key + 1u) {
      acc = acc + scores[key] * v[(key * d.n_kv_heads + kvh) * hd + i];
    }
    o[h * hd + i] = acc * inv;
  }
}
