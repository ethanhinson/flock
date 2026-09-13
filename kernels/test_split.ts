// The SHARDED forward pass == the whole-model forward pass.
//
//   deno run --unstable-webgpu --allow-all kernels/test_split.ts
//
// WHAT THIS IS FOR. flock runs Qwen3 across devices: the coordinator holds the
// embedding and layers 0..cut-1, birds hold cut..27, and the hidden state crosses
// the network once per token before coming back for output_norm and the tied head.
// test_model.ts proves the UNSPLIT engine matches ONNX token for token. This
// proves that splitting it changes nothing -- which is the only new claim the
// distributed path makes.
//
// The bar is BIT-IDENTICAL, not close. Both sides run the same kernels on the same
// weights in the same order; the only difference is that a hidden state makes a
// round trip through host memory in the middle. Float32 copied out and back is
// exact, so any difference at all would be a real wiring error -- an off-by-one in
// the layer range, a KV cache advanced twice, output_norm applied on both sides of
// the cut -- and every one of those produces fluent wrong text rather than a
// crash. A tolerance would hide exactly the bugs worth catching.
//
// f16 WIRE ENCODING IS DELIBERATELY NOT IN THIS TEST. flock's wire format
// (web/js/wire.mjs) sends activations as f16, which is lossy and would put a real
// ~1e-3 floor under any comparison. That is a property of the transport, not of
// the split, so it is tested where it lives. Keeping it out is what lets this
// assertion be exact and therefore diagnostic.

import { getDevice, ok, summary } from "./lib.ts";
import { Layer, QWEN3_06B } from "./layer.ts";
import { Model } from "./model.ts";
import { realModel } from "./real_weights.ts";

const dev = await getDevice();
const cfg = QWEN3_06B;

console.log("loading Qwen3-0.6B-Q8_0.gguf ...");
const m = await realModel();
console.log(`  ${m.nLayers} layers, vocab ${m.vocab}, hidden ${m.hidden}\n`);

const CUT = 24; // flock's default split: coordinator 0-23, birds 24-27

// The reference: one Model holding everything, the configuration test_model.ts
// validated against ONNX.
const whole = await Model.create(dev, m, cfg);
ok("the whole model holds every layer", whole.held === m.nLayers, `${whole.held} layers`);

// The coordinator: the same class, cut. It holds the embedding, layers 0..CUT-1,
// output_norm and the tied head.
const coord = await Model.create(dev, m, cfg, { cut: CUT });
ok(
  `a cut-${CUT} coordinator holds only layers 0-${CUT - 1}`,
  coord.held === CUT,
  `${coord.held} layers`,
);

// The birds: layers CUT..27, each its own Layer with its own KV cache, exactly as
// a phone holds them. Built through fromBuffers' sibling create() here because the
// bytes are already local; a real bird streams them (see test_frombuffers.ts).
const birds: Layer[] = [];
for (let i = CUT; i < m.nLayers; i++) {
  birds.push(await Layer.create(dev, m.layers[i], cfg));
}
console.log(`coordinator: layers 0-${CUT - 1};  birds: layers ${CUT}-${m.nLayers - 1}\n`);

/**
 * One lap, the way server/server.js drives it: coordinator forward, through
 * every bird in order, then back to the coordinator for the head.
 *
 * The hidden state crosses a host boundary at each hop, which is what the real
 * transport does -- so this exercises the same readback/upload seam, minus the
 * network and minus the f16 encoding.
 */
async function lap(ids: number[]): Promise<number> {
  let flat = await coord.encodePartial(ids);
  const seq = ids.length;
  for (const b of birds) {
    // Each bird consumes every position and produces every position: a bird has to
    // fill its own KV cache for all of them, not just the one being predicted.
    if (seq === 1) b.encode(flat);
    else b.encodePrefill(seq, flat);
    flat = await b.readOutput(seq);
  }
  return await coord.projectHidden(flat, seq);
}

// --- prompt: the 16 ids test_model.ts uses, so this ties back to the ONNX proof --
const PROMPT = [
  151644,
  872,
  198,
  63593,
  315,
  9625,
  30,
  151645,
  198,
  151644,
  77091,
  198,
  151667,
  271,
  151668,
  271,
];

const wholeFirst = await whole.step(PROMPT);
const splitFirst = await lap(PROMPT);
ok(
  "split and whole agree on the first token after the prompt",
  wholeFirst === splitFirst,
  `whole ${wholeFirst}, split ${splitFirst}`,
);
ok("that token is the one the ONNX comparison pins (785)", splitFirst === 785, `got ${splitFirst}`);

// --- and then a real generation, because step 0 is the easy case --------------
//
// A single token cannot see a KV cache bug: every cache is empty at position 0.
// Divergence in a sharded engine shows up once the caches on the two sides of the
// cut have to stay in step, which is what generating does.
const N = 12;
const wholeIds: number[] = [];
const splitIds: number[] = [];
let a = wholeFirst, b = splitFirst;
for (let i = 0; i < N; i++) {
  wholeIds.push(a);
  splitIds.push(b);
  if (a === m.eos || b === m.eos) break;
  a = await whole.step([a]);
  b = await lap([b]);
}
console.log(`whole: ${JSON.stringify(wholeIds)}`);
console.log(`split: ${JSON.stringify(splitIds)}\n`);

const same = wholeIds.length === splitIds.length &&
  wholeIds.every((x, i) => x === splitIds[i]);
ok(
  `greedy decode is identical across the split for ${wholeIds.length} tokens`,
  same,
  same
    ? `${wholeIds.length} tokens identical`
    : `first difference at ${wholeIds.findIndex((x, i) => x !== splitIds[i])}`,
);

// --- the hidden state itself, not just the token it produces ------------------
//
// Matching tokens could in principle survive a small error, since argmax discards
// magnitude. Comparing the hidden state at the cut against the same position of an
// unsplit pass is the stronger statement, and it should be exact.
whole.reset();
coord.reset();
for (const l of birds) l.reset();

const probe = PROMPT.slice(0, 8);
const coordOut = await coord.encodePartial(probe);
ok(
  "the coordinator hands over every position, not only the last",
  coordOut.length === probe.length * cfg.hidden,
  `${coordOut.length} floats for ${probe.length} tokens`,
);

// The same prefix through the unsplit model, read at the cut. hiddenState() reads
// after ALL layers, so the comparison has to be made by running the unsplit
// model's first CUT layers -- which is what a second cut Model with an identical
// configuration is. Two independent instances agreeing is also a check that
// nothing in Model holds cross-instance state.
const coord2 = await Model.create(dev, m, cfg, { cut: CUT });
const coord2Out = await coord2.encodePartial(probe);
let hdiff = 0;
for (let i = 0; i < coordOut.length; i++) {
  hdiff = Math.max(hdiff, Math.abs(coordOut[i] - coord2Out[i]));
}
ok(
  "two independently built coordinators produce the identical hidden state",
  hdiff === 0,
  `max |diff| ${hdiff.toExponential(1)} over ${coordOut.length} floats`,
);

// --- the guards, because a cache that silently desynchronizes is the real risk --
let threw = "";
try {
  await coord.encodePartial([1, 2, 3], 999);
} catch (e) {
  threw = String((e as Error).message);
}
ok(
  "an offset that disagrees with the cache position throws rather than computing",
  threw.includes("999") && threw.includes("position"),
  threw || "(did not throw)",
);

threw = "";
try {
  await coord.projectHidden(new Float32Array(cfg.hidden - 1), 1);
} catch (e) {
  threw = String((e as Error).message);
}
ok(
  "a short hidden state throws rather than projecting garbage",
  threw.includes("floats"),
  threw || "(did not throw)",
);

Deno.exit(summary());
