// Q8_0 dequantizing matvec, cooperative version. Same math as q8_matmul.wgsl,
// three structural changes:
//
//  1. SPLIT BUFFERS. The 34-byte file layout interleaves an f16 scale with 32
//     int8s, so nothing is u32-aligned and the reference kernel has to extract
//     every single byte with a shift and a mask. Here the host repacks once at
//     upload time (lib.ts splitQ8) into a contiguous `qs` array -- 8 quants per
//     aligned u32 word -- and a separate `scales` array of f16 packed 2 per
//     word. A whole block is then 2 loads instead of 34 byte extractions.
//
//  2. VECTOR DECODE. One u32 of `qs` becomes a vec4<f32> and feeds dot(), so a
//     block is 8 dot-products' worth of work in 8 instructions rather than 32
//     scalar multiply-adds. See DECODE_I8X4 below for the two spellings.
//
//  3. COOPERATIVE REDUCTION. The reference kernel gives one row to one thread,
//     which leaves a matvec's parallelism capped at n_rows and every thread
//     re-reading the same x. Here a workgroup of 256 threads covers ROWS_PER_WG
//     rows: threads are split into LANES groups walking the row's blocks
//     strided, each keeping a private accumulator, and the partials are then
//     tree-reduced in workgroup memory. That gives LANES-way more parallelism
//     per row and lets x stay hot in cache across the workgroup.
//
// The two $-placeholders are substituted by the host (lib.ts coopSource) because
// WGSL has no preprocessor and the 8-bit unpack builtins are not universally
// implemented -- Deno's wgpu has unpack2x16float but not unpack4xI8. Both
// spellings produce bit-identical values; only the instruction count differs.

// BATCHING (n_tokens). Prefill runs N tokens through the same weights, which is a
// GEMM rather than the GEMV decode needs. Rather than a second kernel, the token
// index rides on workgroup_id.y: workgroup (r, t) produces rows r*ROWS_PER_WG..
// for token t, reading x at t*n_cols and writing o at t*n_rows.
//
// The weights are read once per (row, token) pair either way, so this shares no
// more memory traffic than N separate dispatches would -- what it saves is N-1
// dispatches per projection, and at a ~17.6 us dispatch floor and 7 projections
// per layer over 28 layers that is the difference between a prefill dominated by
// arithmetic and one dominated by the queue. A tiled GEMM that actually reuses the
// weight loads across tokens would be faster still; it is not written, and
// bench_model.ts reports what that leaves on the table rather than guessing.
//
// n_tokens == 1 with a 1-wide y dispatch is bit-identical to the pre-batching
// kernel -- every index below reduces to the old one -- so the existing bit-exact
// matvec tests are the regression guard for this change.
struct Dims {
  n_rows: u32,     // output size
  n_cols: u32,     // input size, a multiple of 32
  n_tokens: u32,   // batch size; 1 for decode
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       qs: array<u32>;     // int8 quants, 4 per word
@group(0) @binding(1) var<storage, read>       sc: array<u32>;     // f16 scales, 2 per word
@group(0) @binding(2) var<storage, read>       x:  array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> o:  array<f32>;
@group(0) @binding(4) var<uniform>             d:  Dims;

const WG: u32 = 256u;            // threads per workgroup
const LANES: u32 = 64u;          // threads cooperating on one row (WG / ROWS_PER_WG)
const ROWS_PER_WG: u32 = 4u;     // rows one workgroup produces

// [ROWS_PER_WG rows][LANES partials]. 4 KB, well inside the 16 KB guaranteed.
var<workgroup> part: array<f32, WG>;

$HALF_TO_F32

// Decode 4 packed int8 from a u32 into a vec4<f32>.
fn dec4(w: u32) -> vec4<f32> {
  return $DECODE_I8X4;
}

// Decode the f16 scale at index `i` of the 2-per-word scales array.
fn scale_at(i: u32) -> f32 {
  return $DECODE_F16;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let row_in_wg = t / LANES;     // which of the ROWS_PER_WG rows this thread serves
  let lane = t % LANES;          // which stride offset within that row
  let row = wg.x * ROWS_PER_WG + row_in_wg;
  let tok = wg.y;                // which token of the batch; 0 for decode

  let nb = d.n_cols / 32u;       // blocks per row
  let row_words = d.n_cols / 4u; // u32 words of qs per row (4 quants each)
  // x is indexed in vec4s, so a token's slice starts n_cols/4 vec4s in.
  let xtok = tok * (d.n_cols / 4u);

  var acc = 0.0;
  // Out-of-range rows still run the reduction below -- every thread in the
  // workgroup must reach the same barriers or the result is undefined -- they
  // just contribute zero and skip the store.
  if (row < d.n_rows && tok < d.n_tokens) {
    // Lane `lane` takes blocks lane, lane+LANES, lane+2*LANES, ...
    for (var b = lane; b < nb; b = b + LANES) {
      let s = scale_at(row * nb + b);
      let wbase = row * row_words + b * 8u;
      let xbase = xtok + b * 8u;
      // A 32-weight block is 8 u32 words of quants and 8 vec4s of x. Four
      // (dot + dot) pairs, each folded in with the scale, matching the CPU
      // reference's coop shape exactly.
      for (var g = 0u; g < 4u; g = g + 1u) {
        let d0 = dec4(qs[wbase + g * 2u]);
        let d1 = dec4(qs[wbase + g * 2u + 1u]);
        acc = acc + s * (dot(d0, x[xbase + g * 2u]) + dot(d1, x[xbase + g * 2u + 1u]));
      }
    }
  }

  // Tree-reduce each row's LANES partials. Strides halve within a row's slice
  // of `part`, so rows never touch each other's partials.
  part[t] = acc;
  workgroupBarrier();
  var stride = LANES / 2u;
  while (stride > 0u) {
    if (lane < stride) {
      part[t] = part[t] + part[t + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (lane == 0u && row < d.n_rows && tok < d.n_tokens) {
    o[tok * d.n_rows + row] = part[t];
  }
}
