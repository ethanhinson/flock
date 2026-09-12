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

  q = ropeRef(q, 1, nHeads, headDim, pos, invFreq);
  k = ropeRef(k, 1, nKvHeads, headDim, pos, invFreq);

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
