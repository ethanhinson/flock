// Token embedding lookup: gather N rows out of a Q8_0 matrix and dequantize.
//
//   out[t][c] = dequant(token_embd[ids[t]])[c]
//
// This replaces embed.onnx. It is a GATHER, not a matmul, and that distinction is
// the whole point of the kernel existing. token_embd.weight for Qwen3-0.6B is
// [vocab=151936][hidden=1024] Q8_0 -- 165 MB on disk, 622 MB dequantized. A
// decode step needs exactly ONE of those rows and a prefill needs one per prompt
// token, so dequantizing the matrix (or doing a one-hot matmul against it) would
// do 151936x more work than the answer requires and would not fit in a buffer
// anyway. The weights stay Q8_0 on the GPU for the whole run; only the rows a
// token actually names are ever decoded.
//
// It reads the SPLIT layout (lib.ts splitQ8: contiguous int8 `qs`, f16 `scales`
// packed two per u32) rather than the 34-byte on-disk layout, because the LM head
// matvec needs that same buffer -- the weights are TIED. Uploading one repacked
// copy and pointing both kernels at it is the difference between 165 MB and
// 330 MB of VRAM for the single largest tensor in the model.
//
// One workgroup per token, WG threads walking that row strided. Each output
// element is one scale multiply, so this is pure bandwidth: 1024 int8 reads and
// 1024 f32 writes per token, and at decode it is 4 KB of traffic -- entirely
// dispatch overhead in practice, which is why it is not optimized further.

struct Dims {
  n_tokens: u32,   // how many ids to look up
  hidden: u32,     // row width, a multiple of 32
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       qs:  array<u32>;   // int8 quants, 4 per word
@group(0) @binding(1) var<storage, read>       sc:  array<u32>;   // f16 scales, 2 per word
@group(0) @binding(2) var<storage, read>       ids: array<u32>;   // token ids, length n_tokens
@group(0) @binding(3) var<storage, read_write> o:   array<f32>;   // [n_tokens][hidden]
@group(0) @binding(4) var<uniform>             d:   Dims;

const WG: u32 = 128u;

$HALF_TO_F32

// Decode 4 packed int8 from a u32 into a vec4<f32>. Same two spellings as the
// matvec kernels; substituted by the host (lib.ts coopSource).
fn dec4(w: u32) -> vec4<f32> {
  return $DECODE_I8X4;
}

fn scale_at(i: u32) -> f32 {
  return $DECODE_F16;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tok = wg.x;
  if (tok >= d.n_tokens) { return; }

  let row = ids[tok];
  let nb = d.hidden / 32u;               // blocks in this row
  let row_words = d.hidden / 4u;         // u32 words of qs per row
  let obase = tok * d.hidden;

  // One thread per 4-element group: 256 groups for hidden=1024, so each of the
  // 128 threads handles two. The group index determines the block (8 groups per
  // 32-weight block) and therefore which scale applies.
  let n_groups = d.hidden / 4u;
  for (var g = lid.x; g < n_groups; g = g + WG) {
    let blk = g / 8u;
    let s = scale_at(row * nb + blk);
    let vals = dec4(qs[row * row_words + g]) * s;
    let at = obase + g * 4u;
    o[at]      = vals.x;
    o[at + 1u] = vals.y;
    o[at + 2u] = vals.z;
    o[at + 3u] = vals.w;
  }
}
