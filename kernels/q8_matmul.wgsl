// Q8_0 dequantizing matrix-vector multiply.
//
// This is the kernel that makes GGUF weights usable without a conversion step.
// A Q8_0 block is 32 int8 weights sharing one f16 scale, laid out as:
//
//   [ f16 scale | 32 x int8 ]   = 34 bytes per 32 weights
//
// WGSL has no int8 or f16 array type, so the weights arrive as array<u32> and
// we unpack bytes by hand. The multiply happens in f32 after dequantizing each
// block -- that is the "dequantize while multiplying" step that avoids ever
// materialising the full f32 weight matrix in memory.
//
// One invocation computes one output row: out[row] = dot(W[row, :], x[:]).

struct Dims {
  n_rows: u32,        // output size
  n_cols: u32,        // input size (must be a multiple of 32)
  blocks_per_row: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read>       w: array<u32>;   // packed Q8_0 blocks
@group(0) @binding(1) var<storage, read>       x: array<f32>;   // input vector
@group(0) @binding(2) var<storage, read_write> o: array<f32>;   // output vector
@group(0) @binding(3) var<uniform>             d: Dims;

// Decode an IEEE-754 half from its 16 bits. WGSL has no f16 without an
// extension, so this is done in f32 arithmetic.
fn half_to_f32(h: u32) -> f32 {
  let sign = select(1.0, -1.0, (h & 0x8000u) != 0u);
  let exp  = (h >> 10u) & 0x1Fu;
  let man  = h & 0x3FFu;
  if (exp == 0u) {
    return sign * f32(man) * 0.000000059604645;   // 2^-24, subnormal
  }
  if (exp == 31u) {
    return sign * 3.4028235e38;                   // clamp inf to max finite
  }
  return sign * (1.0 + f32(man) / 1024.0) * exp2(f32(exp) - 15.0);
}

// int8 stored in a byte: values 128..255 are negative.
fn to_i8(b: u32) -> f32 {
  return select(f32(b), f32(b) - 256.0, b > 127u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= d.n_rows) { return; }

  var acc = 0.0;
  // Byte offset of this row's first block. 34 bytes per block.
  let row_byte = row * d.blocks_per_row * 34u;

  for (var blk = 0u; blk < d.blocks_per_row; blk = blk + 1u) {
    let base = row_byte + blk * 34u;

    // The f16 scale straddles an arbitrary byte boundary, so read it bytewise.
    let s_lo = (w[base / 4u] >> ((base % 4u) * 8u)) & 0xFFu;
    let b1 = base + 1u;
    let s_hi = (w[b1 / 4u] >> ((b1 % 4u) * 8u)) & 0xFFu;
    let scale = half_to_f32(s_lo | (s_hi << 8u));

    // 32 int8 weights follow the scale.
    var sum = 0.0;
    let col0 = blk * 32u;
    for (var i = 0u; i < 32u; i = i + 1u) {
      let byte_at = base + 2u + i;
      let b = (w[byte_at / 4u] >> ((byte_at % 4u) * 8u)) & 0xFFu;
      sum = sum + to_i8(b) * x[col0 + i];
    }
    acc = acc + scale * sum;
  }
  o[row] = acc;
}
