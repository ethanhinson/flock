// Run web/shard0.onnx (Qwen3 layers 24-25) under onnxruntime-node and dump its
// output, so the WGSL layer can be diffed against the engine flock uses today.
//
//   node kernels/onnx_truth.mjs <in.json> <out.json>
//
// This is a separate Node process on purpose. onnxruntime-node is a native
// N-API addon and does not load under Deno, while the kernels need Deno's
// WebGPU, so the two halves cannot share one runtime. The test drives this as a
// subprocess and compares the JSON.
//
// IMPORTANT about what this comparison can and cannot show. The ONNX shard holds
// the ORIGINAL f32 weights exported from HuggingFace; the WGSL layer holds Q8_0
// from the GGUF. So a disagreement is the sum of:
//
//   - Q8_0 quantization error, which is the dominant term by orders of magnitude
//     (one f16 scale per 32 weights, weights rounded to int8: ~0.4% per weight)
//   - genuinely different arithmetic (ORT may use different kernels and orders)
//   - any actual bug
//
// A bug therefore does NOT show up as a tight-tolerance failure here; it shows up
// as a structural difference -- cosine similarity well below 1, or a disagreement
// far larger than quantization alone can explain. The test computes both, and
// also quantizes the ONNX weights itself to predict what the quantization term
// should be, so the comparison has something to be measured against.

import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error('usage: node kernels/onnx_truth.mjs <in.json> <out.json>');
  process.exit(2);
}

// onnxruntime-node is a native addon installed under node_modules, and the
// caller may pass an explicit path because a git worktree does not have its own
// node_modules. Import by URL so no resolution config is needed.
const req0 = JSON.parse(readFileSync(inPath, 'utf8'));
const ortPath = req0.ort ??
  new URL('../node_modules/onnxruntime-node/dist/index.js', import.meta.url).pathname;
if (!existsSync(ortPath)) {
  console.error(`onnxruntime-node not found at ${ortPath}. It is a devDependency of ` +
    `the repo and is not installed in a worktree; pass {"ort": "<path>"} in the input ` +
    `JSON to point at an existing install, or run npm install.`);
  process.exit(3);
}
const ort = (await import(`file://${ortPath}`)).default;

const req = req0;
const {
  shard = new URL('../web/shard0.onnx', import.meta.url).pathname,
  hidden,          // flat f32, length nTokens * 1024
  nTokens = 1,
  positions,       // int64 position ids, length nTokens
  kvHeads = 8,
  headDim = 128,
  nLayers = 2,
  pastLen = 0,
} = req;

const sess = await ort.InferenceSession.create(shard);

const feeds = {
  hidden: new ort.Tensor('float32', Float32Array.from(hidden), [1, nTokens, 1024]),
  position_ids: new ort.Tensor('int64', BigInt64Array.from(positions.map(BigInt)), [1, nTokens]),
};
// Empty KV cache is a zero-length tensor, not a missing input: the graph's
// concat needs something with the right rank.
for (let l = 0; l < nLayers; l++) {
  const n = pastLen * kvHeads * headDim;
  const shape = [1, kvHeads, pastLen, headDim];
  feeds[`past_k${l}`] = new ort.Tensor('float32', new Float32Array(n), shape);
  feeds[`past_v${l}`] = new ort.Tensor('float32', new Float32Array(n), shape);
  if (req[`past_k${l}`]) {
    feeds[`past_k${l}`] = new ort.Tensor('float32', Float32Array.from(req[`past_k${l}`]), shape);
    feeds[`past_v${l}`] = new ort.Tensor('float32', Float32Array.from(req[`past_v${l}`]), shape);
  }
}

const res = await sess.run(feeds);
const out = { output: Array.from(res.output.data), dims: res.output.dims };
for (let l = 0; l < nLayers; l++) {
  out[`new_k${l}`] = Array.from(res[`new_k${l}`].data);
  out[`new_v${l}`] = Array.from(res[`new_v${l}`].data);
  out[`new_k${l}_dims`] = res[`new_k${l}`].dims;
}
writeFileSync(outPath, JSON.stringify(out));
