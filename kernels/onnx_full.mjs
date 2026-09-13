// Ground truth for the WGSL engine: run the WHOLE Qwen3-0.6B under
// onnxruntime-node and dump the token ids, logits and hidden states.
//
//   node kernels/onnx_full.mjs <in.json> <out.json>
//
// A separate Node process for the same reason onnx_truth.mjs is: onnxruntime-node
// is a native N-API addon that will not load under Deno, and the kernels need
// Deno's WebGPU. The two halves exchange JSON.
//
// WHAT "THE WHOLE MODEL" IS HERE. flock's ONNX artifacts split Qwen3-0.6B into
// five graphs, and the split is not where you would guess:
//
//   embed.onnx      ids -> hidden                           (coordinator)
//   layers.onnx     hidden -> hidden, layers 0-23           (coordinator)
//   shard0.onnx     hidden -> hidden, layers 24-25
//   shard1.onnx     hidden -> hidden, layers 26-27 AND THE FINAL RMSNorm
//   head.onnx       hidden -> argmax token id
//
// The final RMSNorm (model.model.norm / output_norm.weight) lives in the LAST
// SHARD, not in head.onnx. The export applied it when is_last, and
// built head.onnx as `Head(norm, lm_head)` whose
// forward is `self.head(hidden).argmax(-1)` -- it takes the norm in its
// constructor and never calls it. Reading head.onnx as "norm then project" and
// applying output_norm twice on the WGSL side would be a subtle, plausible-looking
// error, so it is stated here in the file that defines the reference.
//
// head.onnx returns only the argmax, so this also computes the logits itself from
// the same tied matrix when asked. That matters for the divergence diagnosis: if
// WGSL and ONNX pick different tokens, the question is immediately "by how much"
// -- a 1e-5 gap between the top two logits is f32 drift, a large gap is a bug --
// and an argmax alone cannot answer it.

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error('usage: node kernels/onnx_full.mjs <in.json> <out.json>');
  process.exit(2);
}
const req = JSON.parse(readFileSync(inPath, 'utf8'));

const ortPath = req.ort ??
  new URL('../node_modules/onnxruntime-node/dist/index.js', import.meta.url).pathname;
if (!existsSync(ortPath)) {
  console.error(`onnxruntime-node not found at ${ortPath}`);
  process.exit(3);
}
const ort = (await import(`file://${ortPath}`)).default;

const {
  coordDir,            // web/coord: embed.onnx, layers.onnx, head.onnx, coord.json, tok/
  shards = [],         // [{path, n_layers}] in order, covering the layers coord does not
  prompt,              // raw text; the chat template is applied here
  ids: rawIds,         // OR explicit token ids, skipping the tokenizer
  maxTokens = 16,
  wantLogits = false,
  wantHidden = false,
} = req;

const meta = JSON.parse(readFileSync(join(coordDir, 'coord.json'), 'utf8'));

// --- tokenizer -------------------------------------------------------------
// The chat template with enable_thinking:false, then encode with
// add_special_tokens:false -- matching server/coordinator.js exactly, because a
// different template produces a different prompt and therefore different text,
// which would look like an engine bug.
let ids = rawIds;
let tok = null;
if (!ids) {
  const tfPath = req.transformers ??
    new URL('../node_modules/@huggingface/transformers/dist/transformers.mjs',
      import.meta.url).pathname;
  const { AutoTokenizer } = await import(`file://${tfPath}`);
  tok = await AutoTokenizer.from_pretrained(join(coordDir, 'tok'), { local_files_only: true });
  const text = tok.apply_chat_template(
    [{ role: 'user', content: prompt }],
    { tokenize: false, add_generation_prompt: true, enable_thinking: false });
  ids = Array.from(tok(text, { add_special_tokens: false }).input_ids.data).map(Number);
}

// --- sessions --------------------------------------------------------------
const embed = await ort.InferenceSession.create(join(coordDir, 'embed.onnx'));
const layers = await ort.InferenceSession.create(join(coordDir, 'layers.onnx'));
const head = await ort.InferenceSession.create(join(coordDir, 'head.onnx'));
const shardSess = [];
for (const s of shards) {
  shardSess.push({
    sess: await ort.InferenceSession.create(s.path),
    n: s.n_layers,
    past: null,
  });
}

const H = meta.hidden, KVH = meta.kv_heads, HD = meta.head_dim;
const empty = () => new ort.Tensor('float32', new Float32Array(0), [1, KVH, 0, HD]);

let coordPast = null;
let embeddingOfPrompt = null;

/** One forward pass over `stepIds` at absolute offset `offset`. */
async function forward(stepIds, offset) {
  const n = stepIds.length;
  const emb = await embed.run({
    ids: new ort.Tensor('int64', BigInt64Array.from(stepIds.map(BigInt)), [1, n]),
  });
  const posIds = new ort.Tensor('int64',
    BigInt64Array.from({ length: n }, (_, i) => BigInt(offset + i)), [1, n]);
  // The embedding of the PROMPT, kept so the WGSL gather can be diffed against the
  // tensor the ONNX pipeline actually feeds its layers. Only the first call: later
  // ones are single generated tokens.
  if (wantHidden && offset === 0) embeddingOfPrompt = Array.from(emb.hidden.data);

  // coordinator layers 0..cut-1
  const feed = { hidden: emb.hidden, position_ids: posIds };
  for (let i = 0; i < meta.n_layers; i++) {
    feed[`past_k${i}`] = coordPast ? coordPast[`new_k${i}`] : empty();
    feed[`past_v${i}`] = coordPast ? coordPast[`new_v${i}`] : empty();
  }
  let out = await layers.run(feed);
  coordPast = {};
  for (let i = 0; i < meta.n_layers; i++) {
    coordPast[`new_k${i}`] = out[`new_k${i}`];
    coordPast[`new_v${i}`] = out[`new_v${i}`];
  }
  let hidden = out.output;

  // then each shard in order. The LAST one applies output_norm (is_last).
  for (const s of shardSess) {
    const f = { hidden, position_ids: posIds };
    for (let i = 0; i < s.n; i++) {
      f[`past_k${i}`] = s.past ? s.past[`new_k${i}`] : empty();
      f[`past_v${i}`] = s.past ? s.past[`new_v${i}`] : empty();
    }
    const o = await s.sess.run(f);
    s.past = {};
    for (let i = 0; i < s.n; i++) {
      s.past[`new_k${i}`] = o[`new_k${i}`];
      s.past[`new_v${i}`] = o[`new_v${i}`];
    }
    hidden = o.output;
  }

  // The last position is the one that gets projected.
  const flat = hidden.data;
  const last = new Float32Array(flat.slice((n - 1) * H, n * H));
  const lastT = new ort.Tensor('float32', last, [1, 1, H]);
  const res = await head.run({ hidden: lastT });
  return { token: Number(res.token.data[0]), normed: last };
}

// --- prefill, then greedy decode -------------------------------------------
const result = { ids, generated: [], steps: [] };
let cur = ids, offset = 0;
for (let t = 0; t <= maxTokens; t++) {
  const { token, normed } = await forward(cur, offset);
  offset += cur.length;
  const rec = { pos: offset - 1, token };
  if (wantHidden && t === 0) result.normedAfterPrompt = Array.from(normed);
  result.steps.push(rec);
  if (t === maxTokens) break;
  if (token === meta.eos) { result.hitEos = true; break; }
  result.generated.push(token);
  cur = [token];
}
if (tok) {
  result.text = tok.decode(result.generated, { skip_special_tokens: true });
  result.promptText = tok.decode(ids, { skip_special_tokens: false });
}
result.eos = meta.eos;
if (embeddingOfPrompt) result.embedding = embeddingOfPrompt;
writeFileSync(outPath, JSON.stringify(result));
