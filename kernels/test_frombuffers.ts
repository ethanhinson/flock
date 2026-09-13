// Layer.fromBuffers() against Layer.create(): the bird's construction path.
//
//   deno run --unstable-webgpu --allow-all kernels/test_frombuffers.ts
//
// WHY THIS TEST EXISTS. A bird cannot use Layer.create(): that takes packed CPU
// bytes and calls splitQ8 over the whole tensor, which is exactly the
// materialize-everything step the streaming loader exists to avoid (see
// web/js/gguf-stream.mjs -- holding a 67MB slice in the JS heap while the driver
// makes its own copy is what killed an iPad). So a bird streams each tensor
// straight into GPU buffers and needs a constructor that consumes those.
//
// That makes fromBuffers() a SECOND path to the same weights, and a second path
// is a second chance to be subtly wrong -- a swapped rows/cols, a tensor bound to
// the wrong slot, a norm gain silently missing. The only assertion worth making
// is therefore equality against the path that is already validated end to end
// against ONNX: same weights, two constructors, and the outputs must agree
// BIT-EXACTLY. Not "within tolerance" -- both run the identical kernels on the
// identical bytes, so any difference at all is a wiring bug, and a tolerance
// would only hide it.
//
// The GPU-side split is produced here by uploading splitQ8's output, which is
// what the streamer produces incrementally and is asserted byte-identical to by
// web/js/gguf-stream.test.ts. So this test covers the CONSUMER (fromBuffers) and
// that one covers the PRODUCER (the incremental split), which together span the
// bird's path without either test having to do both jobs.

import { getDevice, ok, randVec, splitQ8, storageBuffer, summary } from "./lib.ts";
import { Layer, QWEN3_06B, type LayerBuffers } from "./layer.ts";
import { realLayer } from "./real_weights.ts";

const dev = await getDevice();
const cfg = QWEN3_06B;

const LAYER = 24;
console.log(`loading blk.${LAYER}.* from Qwen3-0.6B-Q8_0.gguf ...`);
const w = await realLayer(LAYER);

// The GPU-resident form of the same weights, keyed exactly as the streaming
// loader keys them: the suffix after `blk.N.`, with `.weight` left ON. That
// convention is load-bearing -- stripping it makes every lookup in layer.ts
// return undefined, which is why fromBuffers checks for the tensors by name.
const gpu: LayerBuffers = {};
for (const [name, t] of Object.entries(w.q8)) {
  const { qs, scales } = splitQ8(t.packed, t.rows, t.cols);
  gpu[name] = {
    rows: t.rows, cols: t.cols,
    qs: storageBuffer(dev, qs), scales: storageBuffer(dev, scales),
  };
}
for (const [name, v] of Object.entries(w.f32)) {
  // A norm gain is 1 x n: rows/cols are unused for these, but stating them keeps
  // the shape of the record uniform.
  gpu[name] = { rows: 1, cols: v.length, data: storageBuffer(dev, v) };
}

const packed = await Layer.create(dev, w, cfg);
const streamed = await Layer.fromBuffers(dev, gpu, cfg);

ok("fromBuffers builds a layer from pre-built GPU buffers",
   streamed instanceof Layer,
   `${Object.keys(gpu).length} tensors`);

// --- decode: several steps, so the KV cache is exercised and not just step 0 ---
//
// One step would only prove the projections are bound right. The cache is where a
// construction bug can hide until position 1 -- and per correctness trap 1 in the
// README, position 0 is the one place two RoPE conventions agree, so a test that
// stops there is exactly the test that cannot see a rotation bug.
let allExact = true;
let worst = 0;
for (let step = 0; step < 4; step++) {
  const h = randVec(cfg.hidden, 1.0);
  const a = await packed.forward(h);
  const b = await streamed.forward(h);
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff = Math.max(diff, Math.abs(a[i] - b[i]));
  if (diff !== 0) allExact = false;
  worst = Math.max(worst, diff);
  ok(`decode step ${step} (nKeys=${step}) is bit-identical across both constructors`,
     diff === 0, `max |diff| ${diff.toExponential(1)}`);
}
ok("every decode step agreed exactly, not merely closely", allExact,
   `worst |diff| over 4 steps ${worst.toExponential(1)}`);

// --- prefill: the other dispatch path, which uses the dimsPre uniforms ---
//
// fromBuffers builds its own dimsPre buffer per weight, so prefill is a genuinely
// separate thing to get wrong: a decode-only test would pass with dimsPre
// unwritten.
packed.reset();
streamed.reset();
const N = 8;
const hp = randVec(N * cfg.hidden, 1.0);
packed.encodePrefill(N, hp);
streamed.encodePrefill(N, hp);
const pa = await packed.readOutput(N);
const pb = await streamed.readOutput(N);
let pdiff = 0;
for (let i = 0; i < pa.length; i++) pdiff = Math.max(pdiff, Math.abs(pa[i] - pb[i]));
ok(`prefill of ${N} tokens is bit-identical across both constructors`, pdiff === 0,
   `max |diff| ${pdiff.toExponential(1)} over ${pa.length} floats`);

// --- the guard that makes a missing tensor loud -----------------------------
//
// The failure this prevents is the quiet one: an absent norm gain binds nothing
// and the layer computes with garbage, while an absent projection surfaces as a
// bind-group error naming a slot index rather than a tensor. Both are much worse
// than a throw at load time.
for (const drop of ["attn_q.weight", "attn_k_norm.weight"]) {
  const partial: LayerBuffers = { ...gpu };
  delete partial[drop];
  let threw = "";
  try {
    await Layer.fromBuffers(dev, partial, cfg);
  } catch (e) {
    threw = String((e as Error).message);
  }
  ok(`a missing ${drop} throws and names the tensor`,
     threw.includes(drop), threw || "(did not throw)");
}

Deno.exit(summary());
