// Diff the WGSL layers against the ONNX shard (layers 24-25 of the same model).
//
//   deno run --unstable-webgpu --allow-all kernels/test_onnx.ts
//
// shard0.onnx runs Qwen3 layers 24 and 25. This chains two WGSL Layers over the
// same hidden state and position and compares the final output.
//
// WHAT THIS COMPARISON MEASURES, precisely, because it is easy to over-read.
//
// The ONNX shard holds the ORIGINAL f32 weights exported from HuggingFace. The
// WGSL layers hold Q8_0 from the GGUF. So the two are not computing the same
// function and a tight tolerance is not the right assertion. The difference is
// dominated by quantization: Q8_0 keeps one f16 scale per 32 weights and rounds
// each weight to int8, which is ~0.2-0.4% per weight, and a layer is seven
// matvecs over 1024-3072 of them plus a softmax.
//
// So the test does two things instead of picking a tolerance:
//
//  1. It PREDICTS the quantization term rather than assuming it, by running the
//     CPU reference layer twice with identical arithmetic and only the weights
//     varying: once on the real Q8_0 blocks, once on blocks carrying one extra
//     round of the same-sized quantization noise. That difference has the same
//     statistics as the term separating Q8_0 weights from the f32 weights ONNX
//     holds. (Dequantizing and re-quantizing instead measures exactly 0, because
//     Q8_0 round-tripping is idempotent -- see dequant.ts.)
//
//  2. It asserts the GPU-vs-ONNX gap is no worse than that predicted term (with
//     margin), and that cosine similarity is near 1. A wiring bug fails both:
//     it is O(1), not O(quantization), and it destroys the direction rather than
//     perturbing the magnitude.
//
// onnxruntime-node is a native addon that will not load under Deno, and the
// kernels need Deno's WebGPU, so ONNX runs as a Node subprocess (onnx_truth.mjs)
// and the two halves exchange JSON.
//
// This test SKIPS rather than fails when the shard or onnxruntime is missing. The
// ONNX export is not part of the repo (it is gigabytes and nothing at runtime
// needs it); it is read from kernels/.ref/, or wherever FLOCK_ONNX_REF points.

import { getDevice, ok, randVec, summary } from "./lib.ts";
import { amax, relErr } from "./ops_ref.ts";
import { Layer, type LayerWeights, QWEN3_06B } from "./layer.ts";
import { layerForwardRef, newRefCache } from "./layer_ref.ts";
import { realLayer } from "./real_weights.ts";
import { requantWithNoise } from "./dequant.ts";

const cfg = QWEN3_06B;
const REPO = new URL("../", import.meta.url).pathname;

/** Where the ONNX shard and onnxruntime live. The export directory is
 *  configurable because it is large and untracked; onnxruntime is a devDependency. */
function findPaths() {
  const ref = (Deno.env.get("FLOCK_ONNX_REF") || `${REPO}kernels/.ref`).replace(/\/$/, "");
  const shard = `${ref}/shard0.onnx`;
  const ort = `${REPO}node_modules/onnxruntime-node/dist/index.js`;
  try {
    Deno.statSync(shard);
    Deno.statSync(ort);
    return { shard, ort };
  } catch {
    return null;
  }
}

const paths = findPaths();
if (!paths) {
  console.log("  skip  no ONNX reference: needs shard0.onnx under kernels/.ref/ (or");
  console.log("        FLOCK_ONNX_REF) and onnxruntime-node from `npm install`.");
  Deno.exit(0);
}
console.log(`shard:        ${paths.shard}`);
console.log(`onnxruntime:  ${paths.ort}\n`);

const dev = await getDevice();
const hidden = randVec(cfg.hidden, 0.05);

// --- ONNX ------------------------------------------------------------------
const tmp = await Deno.makeTempDir();
const inPath = `${tmp}/in.json`, outPath = `${tmp}/out.json`;
await Deno.writeTextFile(
  inPath,
  JSON.stringify({
    ort: paths.ort,
    shard: paths.shard,
    nTokens: 1,
    positions: [0],
    pastLen: 0,
    nLayers: 2,
    kvHeads: cfg.nKvHeads,
    headDim: cfg.headDim,
    hidden: Array.from(hidden),
  }),
);
const proc = new Deno.Command("node", {
  args: [new URL("./onnx_truth.mjs", import.meta.url).pathname, inPath, outPath],
  stdout: "piped",
  stderr: "piped",
});
const { code, stderr } = await proc.output();
if (code !== 0) {
  console.log("  skip  onnx_truth.mjs failed:");
  console.log("        " + new TextDecoder().decode(stderr).trim().split("\n")[0]);
  Deno.exit(0);
}
const onnx = JSON.parse(await Deno.readTextFile(outPath));
const onnxOut = Float32Array.from(onnx.output);

// --- WGSL: layers 24 and 25 chained ----------------------------------------
const w24 = await realLayer(24);
const w25 = await realLayer(25);
const l24 = await Layer.create(dev, w24, cfg);
const l25 = await Layer.create(dev, w25, cfg);
const mid = await l24.forward(hidden);
const gpuOut = await l25.forward(mid);

// --- how much of the gap is quantization? ----------------------------------
// Run the CPU reference twice with IDENTICAL arithmetic, varying only the weights:
// once on the real Q8_0 blocks, once on blocks perturbed by another round of
// quantization noise of the same size. The gap between those two outputs has the
// same statistics as the term that separates Q8_0 weights from the f32 weights
// ONNX holds. See dequant.ts for why the obvious dequantize-and-requantize
// approach measures exactly zero instead.
function refChain(a: LayerWeights, b: LayerWeights): Float32Array {
  const c1 = newRefCache(), c2 = newRefCache();
  return layerForwardRef(layerForwardRef(hidden, a, cfg, c1), b, cfg, c2);
}
const refQ8 = refChain(w24, w25);
const refF32 = refChain(requantWithNoise(w24, 12345), requantWithNoise(w25, 67890));

const quantTerm = relErr(refQ8, refF32, 1e-3);
const quantAbs = maxAbs(refQ8, refF32) / amax(refF32);

// --- compare ---------------------------------------------------------------
function maxAbs(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
}

const gpuVsOnnxAbs = maxAbs(gpuOut, onnxOut) / amax(onnxOut);
const gpuVsOnnxCos = cosine(gpuOut, onnxOut);
const gpuVsRefAbs = maxAbs(gpuOut, refQ8) / amax(refQ8);

console.log(`ONNX  output |max| ${amax(onnxOut).toFixed(3)}, first: ${onnxOut[0].toFixed(4)}`);
console.log(`WGSL  output |max| ${amax(gpuOut).toFixed(3)}, first: ${gpuOut[0].toFixed(4)}\n`);

console.log(`quantization term (same arithmetic, one extra round of Q8_0 noise):`);
console.log(
  `  abs/scale ${quantAbs.toExponential(2)}   per-element rel ${quantTerm.toExponential(2)}\n`,
);

// The GPU must match its own Q8_0 CPU reference tightly -- that is the ULP-level
// check test_layer.ts makes, repeated here over two chained layers.
ok(
  "WGSL matches its Q8_0 CPU reference over 2 chained layers",
  gpuVsRefAbs < 1e-6,
  `abs/scale ${gpuVsRefAbs.toExponential(1)}`,
);

// Against ONNX the bar is "no worse than quantization explains". 3x margin
// covers ORT using different kernels and summation orders than either of ours.
ok(
  "WGSL vs ONNX is within what Q8_0 quantization accounts for",
  gpuVsOnnxAbs < Math.max(3 * quantAbs, 1e-3),
  `abs/scale ${gpuVsOnnxAbs.toExponential(2)} vs quantization ${quantAbs.toExponential(2)}`,
);

// Direction is the structural check: quantization perturbs magnitudes, a wiring
// bug changes what is being computed.
ok(
  "WGSL and ONNX agree in direction (cosine ~ 1)",
  gpuVsOnnxCos > 0.999,
  `cosine ${gpuVsOnnxCos.toFixed(6)}`,
);

await Deno.remove(tmp, { recursive: true });
Deno.exit(summary() ? 1 : 0);
