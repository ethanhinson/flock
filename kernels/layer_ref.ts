// CPU reference for one Qwen3 layer, built from the same per-op references the
// individual kernels were validated against.
//
// This is deliberately a *composition of already-trusted pieces* rather than a
// fresh implementation. Each op in it (cpuMatmulQ8F32, rmsnormRef, ropeRef,
// attnRef, swigluRef, addRef) has been checked against the GPU on its own, so if
// the composed layer disagrees, the disagreement is in the wiring -- which op
// feeds which, in what order, with which weights -- and not in the arithmetic.
// Wiring is what a whole-layer test is for.

import { coop, cpuMatmulQ8F32 } from "./lib.ts";
import { addRef, attnRef, rmsnormRef, ropeInvFreq, ropeRef, swigluRef } from "./ops_ref.ts";
import { attnPrefillRef } from "./head_ref.ts";
import type { LayerConfig, LayerWeights } from "./layer.ts";

const COOP_LANES = 64;   // must match q8_coop.wgsl

/** Mutable KV cache for the reference, mirroring the GPU layout. */
export interface RefCache { k: Float32Array[]; v: Float32Array[] }

export function newRefCache(): RefCache { return { k: [], v: [] }; }

/**
 * One decode step. `cache` is appended to, matching the GPU's behaviour, so a
 * multi-token test exercises the same cache-growth path on both sides.
 */
export function layerForwardRef(
  hidden: Float32Array, w: LayerWeights, cfg: LayerConfig, cache: RefCache,
): Float32Array {
  const { nHeads, nKvHeads, headDim, ffn, eps } = cfg;
  const invFreq = ropeInvFreq(headDim, cfg.ropeBase);
  const pos = cache.k.length;
  const mv = (name: string, x: Float32Array) => {
    const t = w.q8[name];
    return cpuMatmulQ8F32(t.packed, x, t.rows, t.cols, coop(COOP_LANES));
  };

  // --- attention -----------------------------------------------------------
  const nh = rmsnormRef(hidden, w.f32["attn_norm.weight"], cfg.hidden, 1, eps);
  let q = mv("attn_q.weight", nh);
  let k = mv("attn_k.weight", nh);
  const v = mv("attn_v.weight", nh);

  // Per-head RMSNorm: headDim-wide gain applied to each head independently.
  q = rmsnormRef(q, w.f32["attn_q_norm.weight"], headDim, nHeads, eps);
  k = rmsnormRef(k, w.f32["attn_k_norm.weight"], headDim, nKvHeads, eps);

  q = ropeRef(q, 1, nHeads, headDim, pos, invFreq, cfg.ropePairing);
  k = ropeRef(k, 1, nKvHeads, headDim, pos, invFreq, cfg.ropePairing);

  cache.k.push(k);
  cache.v.push(v);
  const nKeys = cache.k.length;
  // Flatten the cache into the [key][kv_head][head_dim] layout attnRef expects.
  const kFlat = new Float32Array(nKeys * nKvHeads * headDim);
  const vFlat = new Float32Array(nKeys * nKvHeads * headDim);
  for (let t = 0; t < nKeys; t++) {
    kFlat.set(cache.k[t], t * nKvHeads * headDim);
    vFlat.set(cache.v[t], t * nKvHeads * headDim);
  }

  const attn = attnRef(q, kFlat, vFlat, nHeads, nKvHeads, headDim, nKeys);
  const proj = mv("attn_output.weight", attn);
  let h = addRef(hidden, proj);

  // --- feed-forward --------------------------------------------------------
  const nh2 = rmsnormRef(h, w.f32["ffn_norm.weight"], cfg.hidden, 1, eps);
  const gate = mv("ffn_gate.weight", nh2);
  const up = mv("ffn_up.weight", nh2);
  const act = swigluRef(gate, up);
  const down = mv("ffn_down.weight", act);
  h = addRef(h, down);
  return h;
}

/**
 * One layer over N tokens at once -- the prefill reference.
 *
 * Also a composition of already-trusted pieces, and the same argument applies:
 * every op here has been diffed against its kernel on its own, so a disagreement
 * is in the wiring. Prefill's wiring has four things to get wrong and this
 * reference states each of them:
 *
 *  - The batched matvec's layout. x is [token][cols] and o is [token][rows], so a
 *    per-token slice of the input produces a per-token slice of the output. A
 *    kernel that got the two strides confused would still produce numbers of the
 *    right magnitude.
 *  - RMSNorm over N*nHeads independent head-wide vectors, not over the batch.
 *  - RoPE at position pos0 + i for token i, NOT pos0 for all of them.
 *  - Causal attention with a per-query bound.
 *
 * `cache` is appended to with all N tokens, matching the GPU, so a prefill
 * followed by decode steps exercises the same cache on both sides.
 */
export function layerPrefillRef(
  hidden: Float32Array, nTokens: number, w: LayerWeights, cfg: LayerConfig,
  cache: RefCache,
): Float32Array {
  const { nHeads, nKvHeads, headDim, ffn, eps, hidden: H } = cfg;
  const invFreq = ropeInvFreq(headDim, cfg.ropeBase);
  const pos0 = cache.k.length;
  // The batched matvec is the SAME arithmetic per token as the unbatched one --
  // the kernel's only change is where it reads x and writes o -- so the reference
  // runs the validated per-token matvec N times and concatenates. If the kernel
  // ever starts sharing work across tokens (a real tiled GEMM), this is the
  // reference that has to change with it.
  const mv = (name: string, x: Float32Array, cols: number) => {
    const t = w.q8[name];
    if (t.cols !== cols) throw new Error(`${name} cols ${t.cols} != ${cols}`);
    const out = new Float32Array(nTokens * t.rows);
    for (let i = 0; i < nTokens; i++) {
      out.set(
        cpuMatmulQ8F32(t.packed, x.subarray(i * cols, (i + 1) * cols), t.rows, t.cols,
          coop(COOP_LANES)),
        i * t.rows,
      );
    }
    return out;
  };

  // --- attention -----------------------------------------------------------
  const nh = rmsnormRef(hidden, w.f32["attn_norm.weight"], H, nTokens, eps);
  let q = mv("attn_q.weight", nh, H);
  let k = mv("attn_k.weight", nh, H);
  const v = mv("attn_v.weight", nh, H);

  // N*nHeads and N*nKvHeads independent headDim-wide normalizations.
  q = rmsnormRef(q, w.f32["attn_q_norm.weight"], headDim, nTokens * nHeads, eps);
  k = rmsnormRef(k, w.f32["attn_k_norm.weight"], headDim, nTokens * nKvHeads, eps);

  // Token i is rotated by position pos0 + i. ropeRef's nTokens argument is what
  // makes that per-token rather than uniform.
  q = ropeRef(q, nTokens, nHeads, headDim, pos0, invFreq, cfg.ropePairing);
  k = ropeRef(k, nTokens, nKvHeads, headDim, pos0, invFreq, cfg.ropePairing);

  const kvw = nKvHeads * headDim;
  for (let i = 0; i < nTokens; i++) {
    cache.k.push(k.slice(i * kvw, (i + 1) * kvw));
    cache.v.push(v.slice(i * kvw, (i + 1) * kvw));
  }
  const nKeys = cache.k.length;
  const kFlat = new Float32Array(nKeys * kvw);
  const vFlat = new Float32Array(nKeys * kvw);
  for (let t = 0; t < nKeys; t++) {
    kFlat.set(cache.k[t], t * kvw);
    vFlat.set(cache.v[t], t * kvw);
  }

  const attn = attnPrefillRef(q, kFlat, vFlat, nHeads, nKvHeads, headDim, nTokens, pos0);
  const proj = mv("attn_output.weight", attn, nHeads * headDim);
  let h = addRef(hidden, proj);

  // --- feed-forward --------------------------------------------------------
  const nh2 = rmsnormRef(h, w.f32["ffn_norm.weight"], H, nTokens, eps);
  const gate = mv("ffn_gate.weight", nh2, H);
  const up = mv("ffn_up.weight", nh2, H);
  const act = swigluRef(gate, up);
  const down = mv("ffn_down.weight", act, ffn);
  h = addRef(h, down);
  return h;
}
