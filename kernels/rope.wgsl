// Rotary position embedding. Replaces the ONNX Cos/Sin/Neg/Mul/Add cluster.
//
// Each head's head_dim values are treated as head_dim/2 complex numbers and each
// is rotated by an angle proportional to its position:
//
//   theta_j = pos / base^(2j/head_dim)
//   (a, b) -> (a*cos - b*sin, a*sin + b*cos)
//
// The pairing convention is the thing to get right, and there are two in the
// wild. GGUF/llama.cpp "NORM" mode -- which is what Qwen3 GGUF uses -- pairs
// ADJACENT elements: (0,1), (2,3), (4,5), ... HuggingFace's implementation
// instead pairs halves: (0, 64), (1, 65), ... Both are self-consistent rotations
// and both produce plausible-looking outputs, so picking the wrong one does not
// crash, it just quietly degrades the model. The convention is compiled in here
// rather than made a parameter so it cannot silently drift; test_rope.ts asserts
// it against a reference that states the same choice.
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

  // NORM/adjacent pairing: element 2j and 2j+1 of this head.
  let base = (token * d.n_heads + head) * d.head_dim + j * 2u;
  let a = x[base];
  let b = x[base + 1u];
  x[base]      = a * c - b * s;
  x[base + 1u] = a * s + b * c;
}
