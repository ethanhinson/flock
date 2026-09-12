// Q4_0 dequantizing matvec, cooperative. Structurally identical to q8_coop.wgsl
// -- split buffers, vec4 decode, workgroup reduction -- so only what Q4_0 does
// differently is commented here.
//
// A Q4_0 block is 18 bytes: an f16 scale plus 16 bytes holding 32 weights as
// nibbles, each biased by 8 (stored 0..15, meaning -8..7). The packing is NOT
// sequential: byte i carries weight i in its low nibble and weight i+16 in its
// high nibble. So one u32 word of four bytes covers columns j..j+3 in its low
// nibbles and columns j+16..j+19 in its high nibbles -- two vec4s of x that are
// 16 columns apart, not adjacent. Assuming adjacency here produces output that
// looks statistically plausible and is entirely wrong, which is why the CPU
// reference in lib.ts mirrors this mapping literally.
//
// A block is therefore 4 word loads (vs 8 for Q8_0) covering the same 32
// weights: half the memory traffic for the same arithmetic, which is the point.

struct Dims {
  n_rows: u32,
  n_cols: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       qs: array<u32>;   // nibble pairs, 8 weights per word
@group(0) @binding(1) var<storage, read>       sc: array<u32>;   // f16 scales, 2 per word
@group(0) @binding(2) var<storage, read>       x:  array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> o:  array<f32>;
@group(0) @binding(4) var<uniform>             d:  Dims;

const WG: u32 = 256u;
const LANES: u32 = 64u;
const ROWS_PER_WG: u32 = 4u;

var<workgroup> part: array<f32, WG>;

$HALF_TO_F32

// `v` arrives already masked to one nibble per byte, so the low and high halves
// share one decoder.
fn dec_nib(v: u32) -> vec4<f32> {
  return $DECODE_NIBX4;
}

fn scale_at(i: u32) -> f32 {
  return $DECODE_F16;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let row_in_wg = t / LANES;
  let lane = t % LANES;
  let row = wg.x * ROWS_PER_WG + row_in_wg;

  let nb = d.n_cols / 32u;
  let row_words = d.n_cols / 8u;   // 4 bits per weight -> 8 weights per u32 word

  var acc = 0.0;
  if (row < d.n_rows) {
    for (var b = lane; b < nb; b = b + LANES) {
      let s = scale_at(row * nb + b);
      let wbase = row * row_words + b * 4u;
      let xbase = b * 8u;         // 8 vec4s of x per 32-column block
      for (var g = 0u; g < 4u; g = g + 1u) {
        let word = qs[wbase + g];
        let lo = dec_nib(word & 0x0F0F0F0Fu);
        let hi = dec_nib((word >> 4u) & 0x0F0F0F0Fu);
        // lo covers cols g*4 .. g*4+3 (vec4 index g), hi covers cols
        // g*4+16 .. g*4+19 (vec4 index g+4).
        acc = acc + s * (dot(lo, x[xbase + g]) + dot(hi, x[xbase + g + 4u]));
      }
    }
  }

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

  if (lane == 0u && row < d.n_rows) {
    o[row] = part[t];
  }
}
