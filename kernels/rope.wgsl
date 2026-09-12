// Rotary position embedding. Replaces the ONNX Cos/Sin/Neg/Mul/Add cluster.
//
// Each head's head_dim values are treated as head_dim/2 complex numbers and each
// is rotated by an angle proportional to its position:
//
//   theta_j = pos / base^(2j/head_dim)
//   (a, b) -> (a*cos - b*sin, a*sin + b*cos)
//
// THE PAIRING CONVENTION IS THE THING TO GET RIGHT, and there are two in the wild:
//
//   NORM   (GGUF / llama.cpp)  pairs ADJACENT elements: (0,1), (2,3), (4,5), ...
//   NEOX   (HuggingFace)       pairs HALVES:            (0,64), (1,65), ...
//
// Both are self-consistent rotations, so the wrong one does not crash and does not
// even look wrong -- it quietly degrades the model. It is chosen by the host
// ($ROPE_PAIRING, lib.ts ropeSource) because which one is correct depends on the
// WEIGHTS, not on the file format, and that had to be measured rather than assumed:
//
//   GGUF Q8_0 weights, diffed against the ONNX graphs, 24 layers, 16-token prompt
//     NORM:  token 0 cosine 0.999997, token 15 cosine 0.557   <- position-dependent
//     NEOX:  every token cosine > 0.999
//
// The measurement is in test_rope_convention.ts. The reason NEOX wins here is that
// this GGUF was converted from the HuggingFace checkpoint WITHOUT the q/k weight
// permutation that llama.cpp's converter applies for Llama-style models, so the
// weights are still in HF layout and need HF pairing. A GGUF that had been permuted
// would need NORM, which is why this is a parameter and not a constant.
//
// The failure signature is worth recording because it is what identified the bug
// from output alone: token 0 was essentially EXACT while every later token degraded
// monotonically with position. Position 0 has a zero rotation angle under either
// convention, so it is the one token both agree on -- a per-token cosine that is
// perfect at 0 and decays with position means the rotation, not the arithmetic.
//
// One invocation per rotated PAIR, so the dispatch is n_tokens * n_heads *
// head_dim/2 threads. Writes are in-place-safe only if a and b are read before
// either is written, which they are.

struct Dims {
  n_tokens: u32,
  n_heads: u32,
  head_dim: u32,    // even; 128 for Qwen3
  pos0: u32,        // position of the first token (KV cache offset)
};

@group(0) @binding(0) var<storage, read_write> x: array<f32>;   // [n_tokens][n_heads][head_dim]
@group(0) @binding(1) var<uniform>             d: Dims;
@group(0) @binding(2) var<storage, read>       inv_freq: array<f32>;  // head_dim/2 entries

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half_dim = d.head_dim / 2u;
  let total = d.n_tokens * d.n_heads * half_dim;
  if (gid.x >= total) { return; }

  let j = gid.x % half_dim;                       // which pair within the head
  let head = (gid.x / half_dim) % d.n_heads;
  let token = gid.x / (half_dim * d.n_heads);

  // inv_freq is precomputed on the host rather than with pow() here: pow on a
  // GPU is exp2(log2(x)*y), which does not match the host's pow bit-for-bit and
  // would put an avoidable discrepancy inside a kernel we are trying to validate
  // exactly. It is head_dim/2 values, computed once per model.
  let theta = f32(d.pos0 + token) * inv_freq[j];
  let c = cos(theta);
  let s = sin(theta);

  // Which two elements of this head form the pair. Substituted by the host:
  //   NORM  ia = 2j,  ib = 2j + 1          (adjacent)
  //   NEOX  ia = j,   ib = j + head_dim/2  (halves)
  let head_base = (token * d.n_heads + head) * d.head_dim;
  let ia = head_base + $ROPE_IA;
  let ib = head_base + $ROPE_IB;
  let a = x[ia];
  let b = x[ib];
  x[ia] = a * c - b * s;
  x[ib] = a * s + b * c;
}
