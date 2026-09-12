// Q8_0 matvec for ONE TENSOR-PARALLEL SHARD of a weight matrix.
//
// This is q8_coop.wgsl's inner loop with two extra uniforms and nothing else
// changed: the same LANES/ROWS_PER_WG layout, the same vec4 decode, the same
// `s * (dot + dot)` fold and the same tree reduction. That is deliberate -- the
// arithmetic is what the bit-exact tests in test_coop.ts already pin, so a
// sharded kernel that re-derived it would be a second thing to keep right.
//
// WHAT A SHARD IS. y = W x, W is [rows, cols]. Two ways to cut it:
//
//   ROW-WISE (output-split).  Shard s owns rows [r0, r0+n_rows). It needs the
//     WHOLE x, computes y[r0 .. r0+n_rows), and the shards' outputs CONCATENATE.
//     No reduction. Because row r's lanes still walk every one of the row's
//     nb = cols/32 blocks in exactly the original order, the result for row r is
//     BIT-IDENTICAL to the unsharded kernel's. Set `out_off = r0` and `col_off`
//     is irrelevant (n_cols == cols).
//
//   COLUMN-WISE (input-split).  Shard s owns columns [c0, c0+n_cols) of every
//     row. It needs only x[c0 .. c0+n_cols), computes a PARTIAL y over all rows,
//     and the partials must be SUMMED (shard_reduce.wgsl). Row r's lanes now walk
//     only n_cols/32 blocks, so the lane->block assignment and the tree reduction
//     both differ from the unsharded kernel -- the answer is NOT bit-identical
//     and cannot be. See the note in shard_reduce.wgsl.
//
// Both directions run through THIS kernel. The difference is entirely in what
// the host slices and where it points `out_off`:
//
//   direction   qs/scales slice          n_rows     n_cols    out_off   x
//   row-wise    rows r0..r0+n_rows       shard's    cols      r0        whole
//   col-wise    per-row col window       rows       shard's   s*rows    slice
//
// A row-wise slice is CONTIGUOUS in the split layout (qs is row-major, cols
// wide; scales is row-major, nb wide), so it is a byte range -- which is why
// row-wise also wins on load time and is the only one that can be range-fetched
// straight out of a GGUF. A column-wise slice is strided: every row contributes
// n_cols bytes out of cols, so the host must gather. shard_ref.ts sliceCols does
// that, and its cost is reported in bench_shard.ts rather than hidden.
//
// `col_off` exists so a column-wise shard can read a WHOLE x instead of a
// pre-sliced one -- the tests use it to prove that slicing x on the host and
// offsetting into it on the device give the same answer, i.e. that the split is
// in the indexing and not in the data movement. A real remote shard receives
// only its slice and passes col_off = 0.

struct Dims {
  n_rows: u32,     // rows THIS shard computes
  n_cols: u32,     // columns THIS shard covers, a multiple of 32
  out_off: u32,    // first output index this shard writes (row-wise: r0; col-wise: s*rows)
  col_off: u32,    // first vec4 of x this shard reads, in units of 4 floats
};

@group(0) @binding(0) var<storage, read>       qs: array<u32>;
@group(0) @binding(1) var<storage, read>       sc: array<u32>;
@group(0) @binding(2) var<storage, read>       x:  array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> o:  array<f32>;
@group(0) @binding(4) var<uniform>             d:  Dims;

const WG: u32 = 256u;
const LANES: u32 = 64u;
const ROWS_PER_WG: u32 = 4u;

var<workgroup> part: array<f32, WG>;

$HALF_TO_F32

fn dec4(w: u32) -> vec4<f32> {
  return $DECODE_I8X4;
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

  // nb, row_words and the scale index are all relative to THIS SHARD's width.
  // The host has already sliced the weights, so row `row` of the slice is
  // n_cols wide and its scales are nb long -- the shard does not know or need
  // its own column offset within the original matrix.
  let nb = d.n_cols / 32u;
  let row_words = d.n_cols / 4u;

  var acc = 0.0;
  if (row < d.n_rows) {
    for (var b = lane; b < nb; b = b + LANES) {
      let s = scale_at(row * nb + b);
      let wbase = row * row_words + b * 8u;
      // The ONE place col_off appears. x is indexed in vec4s and a 32-column
      // block is 8 of them, so col_off is in vec4 units and must be a multiple
      // of 8 for a shard boundary to land on a block boundary -- the host
      // enforces that (shardRanges rounds every boundary to 32 columns).
      let xbase = d.col_off + b * 8u;
      for (var g = 0u; g < 4u; g = g + 1u) {
        let d0 = dec4(qs[wbase + g * 2u]);
        let d1 = dec4(qs[wbase + g * 2u + 1u]);
        acc = acc + s * (dot(d0, x[xbase + g * 2u]) + dot(d1, x[xbase + g * 2u + 1u]));
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
    o[d.out_off + row] = part[t];
  }
}
