// CPU references for the three ops that surround the layer stack: token
// embedding lookup, argmax over the logits, and prefill (N-query) attention.
//
// Same discipline as ops_ref.ts -- strict f32 with the GPU's rounding count
// modelled, so a correct kernel matches exactly instead of "closely". Where that
// is not achievable the reference says why:
//
//   embedRef    EXACT. A dequantize is one multiply per element, and the f64
//               product of an int8 and an f16 is exactly representable, so
//               fround of it is the f32 result bit-for-bit.
//   argmaxRef   EXACT by construction -- it is a comparison, not arithmetic. The
//               tie rule (lowest index wins) is the part that has to match, and
//               it is asserted on constructed ties rather than hoped for.
//   attnPrefillRef  NOT exact, and cannot be: the kernel's streaming softmax
//               rescales partial sums by exp(m_old - m_new) as it goes, so it
//               performs a different (though algebraically identical) sequence of
//               f32 operations than any two-pass softmax. This reference models
//               the STREAMING order specifically -- see the note on it -- which
//               is what makes the residual disagreement exp()'s ~15 ULP rather
//               than a summation-order artifact.

import { fma32, halfToF32, Q8_BYTES } from "./lib.ts";

const fr = Math.fround;

/**
 * Gather and dequantize `ids.length` rows out of a Q8_0 [rows][cols] matrix.
 *
 * Deliberately decodes from the 34-byte on-disk layout while the kernel reads the
 * split qs/scales layout, so this also re-checks that splitQ8 did not reorder
 * anything for the one tensor where a mistake would be invisible: every row of
 * token_embd is a plausible embedding, so a wrong row does not look wrong.
 */
export function embedRef(
  packed: Uint8Array, ids: number[], cols: number,
): Float32Array {
  const nb = cols / 32;
  const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  const out = new Float32Array(ids.length * cols);
  for (let t = 0; t < ids.length; t++) {
    const row = ids[t];
    for (let b = 0; b < nb; b++) {
      const base = row * nb * Q8_BYTES + b * Q8_BYTES;
      const scale = fr(halfToF32(dv.getUint16(base, true)));
      for (let i = 0; i < 32; i++) {
        let q = packed[base + 2 + i];
        if (q > 127) q -= 256;
        out[t * cols + b * 32 + i] = fr(q * scale);
      }
    }
  }
  return out;
}

/**
 * Index of the largest value, lowest index winning ties.
 *
 * The tie rule is the contract, not an implementation detail: numpy and torch
 * both return the lowest index, so an argmax that broke ties differently would
 * make a greedy-decode diff against ONNX report a spurious divergence on the
 * first prompt where two logits land on the same f32. Rare, but it happens.
 */
export function argmaxRef(x: Float32Array): number {
  let bi = 0, bv = x[0];
  for (let i = 1; i < x.length; i++) {
    if (x[i] > bv) { bv = x[i]; bi = i; }
  }
  return bi;
}

/**
 * Causal GQA attention over N query positions, computed with the STREAMING
 * softmax the prefill kernel uses.
 *
 * This does NOT reuse attnRef in a loop, and the difference is the point. attnRef
 * computes a row max, then exponentiates, then sums -- two passes. The kernel
 * cannot: it never holds a whole score row, so it carries a running max `m`, a
 * running denominator `l`, and a running output `acc`, and rescales `l` and `acc`
 * by exp(m_old - m_new) whenever a tile raises the max. Algebraically the two are
 * the same number; in f32 they are not, because the streaming form multiplies
 * every earlier contribution by a chain of correction factors.
 *
 * So this reference mirrors the kernel's tile structure exactly -- same TILE, same
 * reduction order within a tile, same rescale points. What is left over after
 * that is exp()'s implementation difference (Metal vs V8, ~15 ULP over the range a
 * max-subtracted softmax feeds it) and nothing else, which is the same floor
 * attention.wgsl already lives with. A reference written as "two-pass softmax per
 * query" would instead disagree by the accumulated rescale error and there would
 * be no way to tell that from a bug.
 *
 * q is [nQueries][nHeads][headDim]; k and v are [nKeys][nKvHeads][headDim] with
 * nKeys >= pos0 + nQueries. Query i attends to keys 0..pos0+i.
 */
export function attnPrefillRef(
  q: Float32Array, k: Float32Array, v: Float32Array,
  nHeads: number, nKvHeads: number, headDim: number,
  nQueries: number, pos0: number, tile = 64, wg = 128,
): Float32Array {
  const out = new Float32Array(nQueries * nHeads * headDim);
  const groupSize = nHeads / nKvHeads;
  const scale = fr(1 / Math.sqrt(headDim));
  const tileS = new Float32Array(tile);
  const part = new Float32Array(wg);
  const acc = new Float32Array(headDim);

  for (let qi = 0; qi < nQueries; qi++) {
    for (let h = 0; h < nHeads; h++) {
      const kvh = Math.floor(h / groupSize);
      const qbase = (qi * nHeads + h) * headDim;
      const nKeys = pos0 + qi + 1;
      let m = -Infinity, l = 0;
      acc.fill(0);

      for (let base = 0; base < nKeys; base += tile) {
        const nThis = Math.min(tile, nKeys - base);

        // Scores. The kernel's inner dot is a serial FMA chain over head_dim,
        // then one multiply by the scale.
        for (let j = 0; j < nThis; j++) {
          const kbase = ((base + j) * nKvHeads + kvh) * headDim;
          let s = 0;
          for (let i = 0; i < headDim; i++) s = fma32(q[qbase + i], k[kbase + i], s);
          tileS[j] = fr(s * scale);
        }

        // Tile max, via the same strided-then-tree reduction the kernel does.
        part.fill(-Infinity);
        for (let t = 0; t < wg; t++) {
          let tm = -Infinity;
          for (let j = t; j < nThis; j += wg) tm = Math.max(tm, tileS[j]);
          part[t] = tm;
        }
        const tileMax = treeReduce(part, Math.max);

        const mNew = Math.max(m, tileMax);
        // exp of an exact difference: both m and mNew are f32, so m - mNew is
        // exact and only the exp rounds.
        const corr = fr(Math.exp(fr(m - mNew)));

        part.fill(0);
        for (let t = 0; t < wg; t++) {
          let ts = 0;
          for (let j = t; j < nThis; j += wg) {
            const e = fr(Math.exp(fr(tileS[j] - mNew)));
            tileS[j] = e;
            ts = fr(ts + e);
          }
          part[t] = ts;
        }
        const tileSum = treeReduce(part, (a, b) => fr(a + b));

        // acc <- acc * corr + sum_j e_j * v[j]. The kernel's own order: one
        // multiply, then a serial FMA chain over the tile.
        for (let i = 0; i < headDim; i++) {
          let a = fr(acc[i] * corr);
          for (let j = 0; j < nThis; j++) {
            a = fma32(tileS[j], v[((base + j) * nKvHeads + kvh) * headDim + i], a);
          }
          acc[i] = a;
        }
        l = fma32(l, corr, tileSum);
        m = mNew;
      }

      for (let i = 0; i < headDim; i++) {
        out[(qi * nHeads + h) * headDim + i] = fr(acc[i] / l);
      }
    }
  }
  return out;
}

/** Pairwise tree reduction over `a`, the order a workgroup reduction produces. */
function treeReduce(a: Float32Array, f: (x: number, y: number) => number): number {
  const buf = Float32Array.from(a);
  for (let stride = buf.length >> 1; stride > 0; stride >>= 1) {
    for (let i = 0; i < stride; i++) buf[i] = f(buf[i], buf[i + stride]);
  }
  return buf[0];
}
