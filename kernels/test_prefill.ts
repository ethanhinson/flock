// Validate a PREFILL through one real Qwen3 layer, on the GPU, three ways.
//
//   deno run --unstable-webgpu --allow-all kernels/test_prefill.ts
//
// The three checks are deliberately different in kind, because prefill has failure
// modes that a single comparison cannot separate:
//
//  1. vs the CPU reference (layerPrefillRef). Catches arithmetic and layout: the
//     batched matvec's strides, RMSNorm's n_vecs, the causal bound.
//
//  2. vs N SEQUENTIAL DECODE STEPS through the same layer. This is the check that
//     matters most and it is the one a prefill implementation usually fails: the
//     two paths must produce the same hidden states, because they compute the same
//     function. They share no code -- different attention kernel, different softmax
//     (one-pass vs streaming), different dispatch shapes, different uniforms -- so
//     agreement between them is strong evidence rather than a tautology. A prefill
//     that applied RoPE at a single position, or read a stale KV cache, or dropped
//     the causal mask, disagrees here while looking perfectly healthy in isolation.
//
//  3. The KV CACHE after prefill must be bit-identical to the cache after the
//     equivalent decode steps. A prefill can produce the right hidden states and
//     still leave the cache wrong (wrong stride, wrong offset), which then
//     corrupts every subsequent decode token rather than the prefill itself --
//     the worst kind of bug to find from generated text.
//
// Then the mixed path: prefill a prompt, decode a few tokens on top of it, and
// compare against all-decode. That is what a real generate() does, and it is where
// the prefill/decode boundary gets exercised.

import { getDevice, ok, randVec, readBack, summary } from "./lib.ts";
import { absErrScaled, amax } from "./ops_ref.ts";
import { Layer, QWEN3_06B, type LayerConfig } from "./layer.ts";
import { layerForwardRef, layerPrefillRef, newRefCache } from "./layer_ref.ts";
import { realLayer } from "./real_weights.ts";

const cfg: LayerConfig = { ...QWEN3_06B, maxPrefill: 256 };
const dev = await getDevice();
const w = await realLayer(24);

/** A fresh Layer, so each case starts with an empty KV cache. */
const fresh = () => Layer.create(dev, w, cfg);

// ============================================== prefill vs the CPU reference
console.log("prefill vs CPU reference (real blk.24 weights)\n");

for (const n of [1, 2, 7, 13, 64, 65]) {
  const hidden = randVec(n * cfg.hidden, 0.05);
  const L = await fresh();
  L.encodePrefill(n, hidden);
  const got = await L.readOutput(n);
  const want = layerPrefillRef(hidden, n, w, cfg, newRefCache());
  const err = absErrScaled(got, want, amax(want));
  ok(`prefill N=${n} vs CPU reference`, err < 2e-6,
    `abs/scale ${err.toExponential(1)}  |max| ${amax(want).toFixed(2)}`);
}

// N=1 prefill must equal N=1 decode against the DECODE reference too -- the two
// references share no attention code, so this pins the streaming softmax against
// the one-pass one on real weights rather than on random q/k/v.
{
  const hidden = randVec(cfg.hidden, 0.05);
  const L = await fresh();
  L.encodePrefill(1, hidden);
  const got = await L.readOutput(1);
  const want = layerForwardRef(hidden, w, cfg, newRefCache());
  const err = absErrScaled(got, want, amax(want));
  ok("prefill N=1 matches the DECODE CPU reference", err < 2e-6,
    `abs/scale ${err.toExponential(1)} (streaming vs one-pass softmax, real weights)`);
}

// ================================= prefill vs N sequential decode steps (GPU)
console.log("\nprefill vs N sequential decode steps, both on GPU\n");

const kvBytes = cfg.nKvHeads * cfg.headDim * 4;

/** The KV cache contents for the first `nKeys` keys, as the GPU holds them. */
async function readCache(L: Layer, nKeys: number) {
  const b = L.cacheBuffers();
  return {
    k: await readBack(dev, b.k, nKeys * kvBytes),
    v: await readBack(dev, b.v, nKeys * kvBytes),
  };
}

for (const n of [2, 7, 13, 33]) {
  const hidden = randVec(n * cfg.hidden, 0.05);

  // Path A: one prefill of n tokens.
  const A = await fresh();
  A.encodePrefill(n, hidden);
  const outA = await A.readOutput(n);
  const cacheA = await readCache(A, n);

  // Path B: n decode steps. Each step consumes its own row of `hidden` -- which is
  // what makes this the same function and not a different one. (A real generate()
  // feeds each step the PREVIOUS step's output; here both paths are fed the same
  // externally-supplied hidden states so the comparison isolates the layer.)
  const B = await fresh();
  const outB = new Float32Array(n * cfg.hidden);
  for (let i = 0; i < n; i++) {
    B.encode(hidden.subarray(i * cfg.hidden, (i + 1) * cfg.hidden));
    outB.set(await B.readOutput(), i * cfg.hidden);
  }
  const cacheB = await readCache(B, n);

  const err = absErrScaled(outA, outB, amax(outB));
  ok(`prefill N=${n} == ${n} decode steps (hidden states)`, err < 2e-6,
    `abs/scale ${err.toExponential(1)}`);

  // The cache is written by the SAME kernels on both paths (rmsnorm then rope then
  // a buffer copy), differing only in batching, so bit-identity is the right bar
  // here -- not a tolerance. A non-zero difference would mean the batched norm or
  // rope computed something else, which is exactly what this is looking for.
  let kSame = 0, vSame = 0;
  for (let i = 0; i < cacheA.k.length; i++) {
    if (cacheA.k[i] === cacheB.k[i]) kSame++;
    if (cacheA.v[i] === cacheB.v[i]) vSame++;
  }
  ok(`prefill N=${n} leaves a BIT-IDENTICAL KV cache`,
    kSame === cacheA.k.length && vSame === cacheA.v.length,
    `k ${kSame}/${cacheA.k.length}, v ${vSame}/${cacheA.v.length}`);
  ok(`prefill N=${n} advanced nKeys correctly`, A.nKeys === n && B.nKeys === n,
    `prefill ${A.nKeys}, decode ${B.nKeys}`);
}

// ============================================ prefill then decode on top of it
console.log("\nprefill then decode (the path generate() takes)\n");

{
  const nPre = 9, nDec = 4;
  const hidden = randVec((nPre + nDec) * cfg.hidden, 0.05);

  // Path A: prefill nPre, then nDec decode steps.
  const A = await fresh();
  A.encodePrefill(nPre, hidden.subarray(0, nPre * cfg.hidden));
  const preOut = await A.readOutput(nPre);
  const decA: Float32Array[] = [];
  for (let i = 0; i < nDec; i++) {
    const at = (nPre + i) * cfg.hidden;
    A.encode(hidden.subarray(at, at + cfg.hidden));
    decA.push(await A.readOutput());
  }

  // Path B: all nPre + nDec as decode steps.
  const B = await fresh();
  const decB: Float32Array[] = [];
  for (let i = 0; i < nPre + nDec; i++) {
    B.encode(hidden.subarray(i * cfg.hidden, (i + 1) * cfg.hidden));
    const o = await B.readOutput();
    if (i >= nPre) decB.push(o);
  }

  let worst = 0;
  for (let i = 0; i < nDec; i++) {
    worst = Math.max(worst, absErrScaled(decA[i], decB[i], amax(decB[i])));
  }
  ok(`${nDec} decode steps on top of a ${nPre}-token prefill match all-decode`,
    worst < 2e-6, `abs/scale ${worst.toExponential(1)}`);
  ok("nKeys is consistent across the prefill/decode boundary",
    A.nKeys === nPre + nDec && B.nKeys === nPre + nDec, `${A.nKeys} vs ${B.nKeys}`);
  ok("the prefill's own output is finite and non-trivial",
    preOut.every(Number.isFinite) && amax(preOut) > 1,
    `|max| ${amax(preOut).toFixed(2)}`);
}

// RoPE position is the silent one: a prefill that rotated every token by the same
// angle is a self-consistent, WRONG model whose output magnitude looks perfectly
// healthy. So assert directly that token i was rotated by pos0 + i, by prefilling
// the SAME hidden state N times and requiring the outputs to differ.
//
// The POSITION SCALE matters for how this can be tested at all, and getting it
// wrong produced two bad assertions before this one. Qwen3's rope_base is 1e6, so
// at positions 0-5 every rotation angle is tiny:
//
//   positions 0..5, identical inputs:  |positional signal| ~1.0e-5
//   GPU vs CPU reference on the outputs: 2.1e-7 relative == 1.0e-5 ABSOLUTE
//
// The signal and the f32 noise floor are the same size -- both are ~1 ULP of an
// output near 48 -- so no metric built on positions 0-5 can discriminate. A
// ">90% of entries differ" threshold failed (86% is correct), and so did comparing
// the GPU's positional difference against the reference's (ratio 0.8-1.2, because
// it is one ULP against one ULP). That is README trap 4 in a new costume: the
// quantity being measured had been cancelled down to the noise.
//
// So the positional assertion is made where the angles are actually large: two
// prefills of the same tokens at pos0 = 0 versus pos0 = 900. There the rotation is
// a real rotation and the difference is O(1), which is a signal a rounding
// difference cannot fake.
{
  const n = 6;
  const one = randVec(cfg.hidden, 0.05);
  const rep = new Float32Array(n * cfg.hidden);
  for (let i = 0; i < n; i++) rep.set(one, i * cfg.hidden);

  const A = await fresh();
  A.encodePrefill(n, rep);
  const at0 = await A.readOutput(n);

  // Same n tokens, but starting 900 keys in. The cache ahead of them is junk as
  // far as this test cares -- what is being asserted is that the POSITION changed
  // the answer, and by how much.
  const B = await fresh();
  for (let i = 0; i < 900; i += 180) B.encodePrefill(180, randVec(180 * cfg.hidden, 0.05));
  B.encodePrefill(n, rep);
  const at900 = await B.readOutput(n);

  let worstDelta = 0;
  for (let i = 0; i < n * cfg.hidden; i++) {
    worstDelta = Math.max(worstDelta, Math.abs(at0[i] - at900[i]));
  }
  ok("position changes the output by O(1) at pos0=900 (RoPE angles are live)",
    worstDelta > 1, `max |diff| ${worstDelta.toFixed(2)} vs |out| ${amax(at0).toFixed(2)}`);

  // And within a prefill, token i must be rotated by pos0 + i rather than all by
  // pos0. At pos0=900 the between-token angles are large enough to see, so this
  // is the check that a batched RoPE actually used its token index.
  let tokDelta = 0;
  for (let i = 0; i < cfg.hidden; i++) {
    tokDelta = Math.max(tokDelta, Math.abs(at900[i] - at900[(n - 1) * cfg.hidden + i]));
  }
  ok("identical tokens WITHIN a prefill differ (RoPE uses the token index)",
    tokDelta > 1e-3, `max |diff| between token 0 and ${n - 1} at pos0=900: ${tokDelta.toExponential(2)}`);

  // The GPU still has to agree with the reference at pos0=0, which is where the
  // arithmetic is checked; the positional assertions above are structural.
  const ref = layerPrefillRef(rep, n, w, cfg, newRefCache());
  const err = absErrScaled(at0, ref, amax(ref));
  ok("repeated-token prefill matches the CPU reference", err < 2e-6,
    `abs/scale ${err.toExponential(1)}`);
}

// Chunking: two prefills of 5 must equal one prefill of 10. This is what lets a
// prompt longer than maxPrefill be fed in pieces, and it is a different code path
// from prefill-then-decode because both halves are batched.
{
  const n = 10;
  const hidden = randVec(n * cfg.hidden, 0.05);
  const A = await fresh();
  A.encodePrefill(n, hidden);
  const outA = await A.readOutput(n);

  const B = await fresh();
  B.encodePrefill(5, hidden.subarray(0, 5 * cfg.hidden));
  const firstHalf = await B.readOutput(5);
  B.encodePrefill(5, hidden.subarray(5 * cfg.hidden));
  const secondHalf = await B.readOutput(5);
  const outB = new Float32Array(n * cfg.hidden);
  outB.set(firstHalf, 0);
  outB.set(secondHalf, 5 * cfg.hidden);

  const err = absErrScaled(outA, outB, amax(outA));
  ok("one prefill of 10 == two chunked prefills of 5", err < 2e-6,
    `abs/scale ${err.toExponential(1)} (this is what chunks a prompt > maxPrefill)`);
}

// Guard rails, because silently computing on a too-large batch would corrupt
// memory rather than erroring.
{
  const L = await fresh();
  let threw = false;
  try { L.encodePrefill(cfg.maxPrefill + 1, randVec((cfg.maxPrefill + 1) * cfg.hidden)); }
  catch { threw = true; }
  ok("prefill beyond maxPrefill throws rather than overrunning the buffers", threw);
  let threw2 = false;
  try { L.encodePrefill(2, randVec(cfg.hidden)); } catch { threw2 = true; }
  ok("a hidden state of the wrong length throws", threw2);
}

Deno.exit(summary() ? 1 : 0);
