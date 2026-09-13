// CPU references for the non-matmul transformer ops, in strict f32.
//
// Same discipline as lib.ts: round after every operation and model the GPU's
// fused multiply-adds, so a correct kernel matches exactly instead of "closely".
// Where a transcendental is involved (cos, sin, exp) exactness is NOT achievable
// -- those are implementation-defined to a few ULP and Metal's differ from V8's
// -- so those references document a tolerance and say why.

import { fma32, type RopePairing } from "./lib.ts";

const fr = Math.fround;

/**
 * RMSNorm over `nVecs` vectors of width `n`: x / sqrt(mean(x^2) + eps) * g.
 *
 * The reduction order mirrors the kernel's: WG threads each sum a stride-WG
 * slice, then a pairwise tree. Summing serially instead disagrees in the last
 * bits, which then gets amplified by the reciprocal square root.
 */
export function rmsnormRef(
  x: Float32Array,
  g: Float32Array,
  n: number,
  nVecs: number,
  eps: number,
  wg = 128,
): Float32Array {
  const out = new Float32Array(n * nVecs);
  const part = new Float32Array(wg);
  for (let v = 0; v < nVecs; v++) {
    const base = v * n;
    part.fill(0);
    for (let t = 0; t < wg; t++) {
      let acc = 0;
      for (let i = t; i < n; i += wg) acc = fma32(x[base + i], x[base + i], acc);
      part[t] = acc;
    }
    const sum = treeSum(part);
    // 1/sqrt(v) as ONE rounding, not a rounded sqrt followed by a rounded
    // divide. Metal contracts it into a single reciprocal-sqrt, and the
    // difference is real: rounding twice costs 1.6 ULP here, which is what made
    // an earlier version of this reference disagree with a correct kernel by
    // 1.9e-7 at every size. Same lesson as fma32 -- match the hardware's
    // rounding count, not just its arithmetic.
    const scale = fr(1 / Math.sqrt(fr(fr(sum / n) + eps)));
    for (let i = 0; i < n; i++) out[base + i] = fr(fr(x[base + i] * scale) * g[i]);
  }
  return out;
}

/**
 * RoPE in place on a copy. `invFreq` must be precomputed by the caller so host and
 * device agree on it.
 *
 * `pairing` selects which two elements of a head rotate together -- "norm"
 * (GGUF/llama.cpp, adjacent) or "neox" (HuggingFace, halves). It is a parameter
 * because which one is correct depends on the weights, not the container: see
 * rope.wgsl. Qwen3-0.6B-Q8_0 needs "neox", measured against ONNX.
 *
 * cos/sin are the reason this one cannot be bit-exact: they are accurate to a
 * few ULP but Metal's and V8's implementations are different functions. The
 * error that leaks into the output is bounded by the input magnitude times a few
 * ULP, which is what test_ops.ts asserts.
 */
export function ropeRef(
  x: Float32Array,
  nTokens: number,
  nHeads: number,
  headDim: number,
  pos0: number,
  invFreq: Float32Array,
  pairing: RopePairing = "neox",
): Float32Array {
  const out = Float32Array.from(x);
  const half = headDim / 2;
  for (let t = 0; t < nTokens; t++) {
    for (let h = 0; h < nHeads; h++) {
      for (let j = 0; j < half; j++) {
        const theta = fr((pos0 + t) * invFreq[j]);
        const c = fr(Math.cos(theta)), s = fr(Math.sin(theta));
        const base = (t * nHeads + h) * headDim;
        const ia = base + (pairing === "norm" ? j * 2 : j);
        const ib = base + (pairing === "norm" ? j * 2 + 1 : j + half);
        const a = out[ia], b = out[ib];
        // a*c - b*s as one FMA: fma(-b, s, a*c).
        out[ia] = fr(-b * s + fr(a * c));
        out[ib] = fr(b * c + fr(a * s));
      }
    }
  }
  return out;
}

/** inv_freq[j] = 1 / base^(2j/headDim). Computed once per model, on the host. */
export function ropeInvFreq(headDim: number, base: number): Float32Array {
  const half = headDim / 2;
  const out = new Float32Array(half);
  for (let j = 0; j < half; j++) out[j] = fr(1 / Math.pow(base, (2 * j) / headDim));
  return out;
}

/** SwiGLU: silu(gate) * up, elementwise. silu(v) = v * sigmoid(v). */
export function swigluRef(gate: Float32Array, up: Float32Array): Float32Array {
  const out = new Float32Array(gate.length);
  for (let i = 0; i < gate.length; i++) {
    const g = gate[i];
    out[i] = fr(fr(fr(g / fr(1 + fr(Math.exp(-g)))) * up[i]));
  }
  return out;
}

/** Elementwise add, for residual connections. */
export function addRef(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = fr(a[i] + b[i]);
  return out;
}

/**
 * Causal grouped-query attention for ONE query position.
 *
 * q is [nHeads][headDim]; k and v are [nKeys][nKvHeads][headDim]. Query head h
 * reads kv head h / (nHeads/nKvHeads) -- that is what GQA means: 16 query heads
 * sharing 8 kv heads, two queries per kv head.
 *
 * Causality here is "attend to keys 0..nKeys-1", which for single-token decode is
 * the whole cache. The ONNX graph's Trilu builds an explicit mask because it
 * processes a whole prompt at once; for one query position the mask is just the
 * loop bound, so there is nothing to materialize.
 *
 * The softmax is the max-subtracted form, which is what the kernel does and what
 * anything numerically sane does. exp() is not bit-reproducible across
 * implementations, so this reference carries a tolerance.
 */
export function attnRef(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  nHeads: number,
  nKvHeads: number,
  headDim: number,
  nKeys: number,
): Float32Array {
  const out = new Float32Array(nHeads * headDim);
  const groupSize = nHeads / nKvHeads;
  const scale = fr(1 / Math.sqrt(headDim));
  const scores = new Float32Array(nKeys);
  for (let h = 0; h < nHeads; h++) {
    const kvh = Math.floor(h / groupSize);
    for (let t = 0; t < nKeys; t++) {
      let acc = 0;
      for (let i = 0; i < headDim; i++) {
        acc = fma32(q[h * headDim + i], k[(t * nKvHeads + kvh) * headDim + i], acc);
      }
      scores[t] = fr(acc * scale);
    }
    let max = -Infinity;
    for (let t = 0; t < nKeys; t++) max = Math.max(max, scores[t]);
    let sum = 0;
    for (let t = 0; t < nKeys; t++) {
      scores[t] = fr(Math.exp(fr(scores[t] - max)));
      sum = fr(sum + scores[t]);
    }
    const inv = fr(1 / sum);
    for (let i = 0; i < headDim; i++) {
      let acc = 0;
      for (let t = 0; t < nKeys; t++) {
        acc = fma32(scores[t], v[(t * nKvHeads + kvh) * headDim + i], acc);
      }
      out[h * headDim + i] = fr(acc * inv);
    }
  }
  return out;
}

function treeSum(a: Float32Array): number {
  const buf = Float32Array.from(a);
  for (let stride = buf.length >> 1; stride > 0; stride >>= 1) {
    for (let i = 0; i < stride; i++) buf[i] = fr(buf[i] + buf[i + stride]);
  }
  return buf[0];
}

/** Max relative error, with an absolute floor so near-zero entries behave. */
export function relErr(a: Float32Array, b: Float32Array, floor = 1e-6): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    worst = Math.max(worst, Math.abs(a[i] - b[i]) / Math.max(floor, Math.abs(b[i])));
  }
  return worst;
}

/**
 * Max absolute error relative to the SCALE of the data, rather than per element.
 *
 * The right metric for a rotation. RoPE computes a*cos - b*sin from two O(1)
 * inputs, so an output can land near zero through cancellation while carrying the
 * same ~1 ULP absolute error as its neighbours. Per-element relative error then
 * reports 7.8e-5 for that one entry and 1e-7 for the rest, which says nothing
 * about the kernel -- the measured absolute error across the whole tensor was
 * 1.79e-7, about 1.5 ULP of the inputs, uniformly.
 *
 * Cancellation-sensitive outputs get judged against the magnitude of what went
 * in, which is what an error budget for the next op actually cares about.
 */
export function absErrScaled(a: Float32Array, b: Float32Array, scale: number): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst / scale;
}

/** Largest magnitude in a tensor, for use as the scale in absErrScaled. */
export function amax(a: Float32Array): number {
  let m = 0;
  for (const v of a) m = Math.max(m, Math.abs(v));
  return m;
}
