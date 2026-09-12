// Dequantize Q8_0 weights back to f32, then re-quantize them at f32 precision.
//
// This exists for one purpose: to isolate how much of a WGSL-vs-ONNX difference
// is Q8_0 quantization rather than a bug. The ONNX shard holds original f32
// weights, so comparing against it mixes quantization error with everything else.
//
// The trick is to hold the ARITHMETIC constant and vary only the weights. Running
// the same CPU reference on (a) the real Q8_0 blocks and (b) blocks whose scales
// and quants have been recomputed from the dequantized values at full precision
// gives two outputs whose only difference is quantization -- and the size of that
// difference is what a fair tolerance against ONNX has to be built on.
//
// It is not a perfect stand-in for the original weights: dequantizing Q8_0 cannot
// recover information that was discarded, so this measures the error of ONE more
// round of quantization on already-quantized values, which is a LOWER bound on the
// real term. Stated in test_onnx.ts, and the reason that test allows margin.

import { halfToF32, quantMatrixQ8, Q8_BYTES } from "./lib.ts";
import type { LayerWeights } from "./layer.ts";

/** Dequantize one Q8_0 tensor to f32. */
export function dequantQ8(packed: Uint8Array, rows: number, cols: number): Float32Array {
  const nb = cols / 32, out = new Float32Array(rows * cols);
  const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < nb; b++) {
      const base = r * nb * Q8_BYTES + b * Q8_BYTES;
      const scale = halfToF32(dv.getUint16(base, true));
      for (let i = 0; i < 32; i++) {
        let q = packed[base + 2 + i]; if (q > 127) q -= 256;
        out[r * cols + b * 32 + i] = Math.fround(q * scale);
      }
    }
  }
  return out;
}

/**
 * A LayerWeights with EXTRA quantization noise of the same size Q8_0 already
 * carries, to estimate how much of a WGSL-vs-ONNX gap quantization accounts for.
 *
 * The obvious approach -- dequantize to f32 and re-quantize -- measures nothing:
 * Q8_0 round-tripping is idempotent, because the dequantized values are exactly
 * representable by the scale and quants they came from, so the second pass
 * reproduces the same bytes and the measured difference is 0. (It really does
 * return 0.0; that is not a bug in the measurement, it is the measurement being
 * the wrong idea.)
 *
 * What works instead: perturb each weight by a uniform random amount up to half a
 * quantization step -- exactly the error the original f32 -> Q8_0 rounding
 * introduced -- and re-quantize. Running the reference on these versus the real
 * weights gives an output difference with the same statistics as the real
 * quantization term, which is what a tolerance against ONNX should be built on.
 *
 * `seed` makes it deterministic so the test does not flicker.
 */
export function requantWithNoise(w: LayerWeights, seed = 1): LayerWeights {
  const out: LayerWeights = { q8: {}, f32: w.f32 };
  // xorshift32: deterministic, and good enough for perturbation noise.
  let s = seed | 0 || 1;
  const rand = () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return ((s >>> 0) / 4294967296) - 0.5;   // [-0.5, 0.5)
  };
  for (const [name, t] of Object.entries(w.q8)) {
    const f32 = dequantQ8(t.packed, t.rows, t.cols);
    const nb = t.cols / 32;
    const dv = new DataView(t.packed.buffer, t.packed.byteOffset, t.packed.byteLength);
    for (let r = 0; r < t.rows; r++) {
      for (let b = 0; b < nb; b++) {
        // One quantization step for this block is its scale: quants are integers.
        const step = halfToF32(dv.getUint16(r * nb * Q8_BYTES + b * Q8_BYTES, true));
        for (let i = 0; i < 32; i++) {
          const at = r * t.cols + b * 32 + i;
          f32[at] = Math.fround(f32[at] + rand() * step);
        }
      }
    }
    out.q8[name] = { rows: t.rows, cols: t.cols, packed: quantMatrixQ8(f32, t.rows, t.cols) };
  }
  return out;
}
