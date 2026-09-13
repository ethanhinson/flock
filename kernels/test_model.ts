// THE DELIVERABLE: the whole WGSL engine against the whole ONNX engine.
//
//   deno run --unstable-webgpu --allow-all kernels/test_model.ts
//
// Generates text with 28 WGSL layers from GGUF Q8_0 weights -- embedding, prefill,
// decode, output_norm, tied LM head, argmax, all on the GPU -- and requires the
// TOKEN IDS to match what onnxruntime-node produces from the same prompt through
// embed.onnx / layers.onnx / shard0.onnx / shard1.onnx / head.onnx.
//
// Matching token ids is a much harder bar than matching a hidden state, and that
// is the point of setting it. The two engines do not compute the same function:
// ONNX holds the original f32 weights and this holds Q8_0, so every logit differs
// by ~1e-2 relative. Greedy decode is nonetheless reproducible whenever the top
// two logits are further apart than that -- and when they are NOT, the engines are
// entitled to disagree. So the test does not merely compare ids; when they diverge
// it reports the MARGIN between the top two logits at the divergence point, which
// is what distinguishes "f32/quantization drift on a near-tie" from a bug.
//
// It also diffs the intermediate tensors, so a failure localizes instead of just
// saying "different text":
//
//   embedding        vs embed.onnx        -- the gather and splitQ8 on 165 MB
//   28-layer hidden  vs the shard chain   -- the layer stack and the KV cache
//   logits           vs a CPU projection  -- output_norm and the tied head
//   token ids        vs head.onnx         -- the whole thing
//
// SKIPS rather than fails when the ONNX artifacts or onnxruntime are missing. The
// export is not part of the repo (622 MB of embed/head weights alone, and nothing
// at runtime needs it); it is read from kernels/.ref/, or wherever FLOCK_ONNX_REF
// points, laid out flat: embed.onnx, layers.onnx, head.onnx, coord.json, tok/,
// shard0.onnx, shard1.onnx, each with its .data file beside it.

import { getDevice, ok, summary } from "./lib.ts";
import { absErrScaled, amax } from "./ops_ref.ts";
import { QWEN3_06B } from "./layer.ts";
import { Model } from "./model.ts";
import { realModel } from "./real_weights.ts";

const REPO = new URL("../", import.meta.url).pathname;
const PROMPT = "Capital of France?";
const MAX_TOKENS = 16;

/** The ONNX export directory (untracked, configurable) and the two npm packages
 *  the Node-side reference runner needs. */
function findPaths() {
  const ref = (Deno.env.get("FLOCK_ONNX_REF") || `${REPO}kernels/.ref`).replace(/\/$/, "");
  const ort = `${REPO}node_modules/onnxruntime-node/dist/index.js`;
  const transformers = `${REPO}node_modules/@huggingface/transformers/dist/transformers.node.mjs`;
  const need = [
    `${ref}/embed.onnx`,
    `${ref}/layers.onnx`,
    `${ref}/head.onnx`,
    `${ref}/coord.json`,
    `${ref}/shard0.onnx`,
    `${ref}/shard1.onnx`,
    ort,
    transformers,
  ];
  try {
    for (const f of need) Deno.statSync(f);
  } catch {
    return null;
  }
  return {
    coordDir: ref,
    shards: [
      { path: `${ref}/shard0.onnx`, n_layers: 2 },
      { path: `${ref}/shard1.onnx`, n_layers: 2 },
    ],
    ort,
    transformers,
  };
}

const paths = findPaths();
if (!paths) {
  console.log("  skip  no ONNX reference: needs the export under kernels/.ref/ (or");
  console.log("        FLOCK_ONNX_REF) and onnxruntime-node from `npm install`.");
  Deno.exit(0);
}

// --- ONNX ground truth ------------------------------------------------------
console.log(`prompt: ${JSON.stringify(PROMPT)}`);
console.log(`onnx:   ${paths.coordDir} + 2 shards\n`);

const tmp = await Deno.makeTempDir();
const inPath = `${tmp}/in.json`, outPath = `${tmp}/out.json`;
await Deno.writeTextFile(
  inPath,
  JSON.stringify({
    ...paths,
    prompt: PROMPT,
    maxTokens: MAX_TOKENS,
    wantHidden: true,
  }),
);
const proc = new Deno.Command("node", {
  args: [new URL("./onnx_full.mjs", import.meta.url).pathname, inPath, outPath],
  stdout: "piped",
  stderr: "piped",
});
const { code, stderr } = await proc.output();
if (code !== 0) {
  console.log("  skip  onnx_full.mjs failed:");
  console.log(
    "        " + new TextDecoder().decode(stderr).trim().split("\n").slice(-3).join("\n        "),
  );
  Deno.exit(0);
}
const onnx = JSON.parse(await Deno.readTextFile(outPath));
const promptIds: number[] = onnx.ids;

console.log(`ONNX prompt:    ${promptIds.length} tokens`);
console.log(`ONNX generated: ${JSON.stringify(onnx.generated)}`);
console.log(`ONNX text:      ${JSON.stringify(onnx.text)}\n`);

// --- WGSL engine ------------------------------------------------------------
const dev = await getDevice();
const t0 = performance.now();
const weights = await realModel();
const M = await Model.create(dev, weights, { ...QWEN3_06B, maxPrefill: 64 });
console.log(
  `WGSL model loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s: ` +
    `${weights.nLayers} layers, vocab ${weights.vocab}\n`,
);

ok(
  "GGUF metadata matches the ONNX export",
  weights.nLayers === 28 && weights.vocab === 151936 && weights.hidden === 1024 &&
    weights.eos === onnx.eos,
  `${weights.nLayers} layers, vocab ${weights.vocab}, eos ${weights.eos}`,
);

const cos = (a: Float32Array, b: Float32Array) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / Math.sqrt(na * nb);
};

// --- the embedding, against embed.onnx --------------------------------------
// The gather is bit-exact against its own CPU reference in test_head.ts; what this
// adds is that it gathers the RIGHT rows for real prompt ids, including the special
// tokens above 151643 that only a templated prompt reaches. The residual here is
// pure Q8_0: embed.onnx holds f32 weights.
{
  const got = await M.embed(promptIds);
  const ref = Float32Array.from(onnx.embedding);
  const c = cos(got, ref);
  ok(
    "WGSL embedding matches embed.onnx on the real prompt ids",
    c > 0.9999,
    `cosine ${c.toFixed(6)}  abs/scale ${
      absErrScaled(got, ref, amax(ref)).toExponential(2)
    } (Q8_0 vs f32)`,
  );
  let worstTok = 1, worstAt = -1;
  for (let t = 0; t < promptIds.length; t++) {
    const ct = cos(got.subarray(t * 1024, (t + 1) * 1024), ref.subarray(t * 1024, (t + 1) * 1024));
    if (ct < worstTok) {
      worstTok = ct;
      worstAt = t;
    }
  }
  ok(
    "every prompt token's embedding row is right (not just the average)",
    worstTok > 0.9999,
    `worst token ${worstAt} (id ${promptIds[worstAt]}) cosine ${worstTok.toFixed(6)}`,
  );
}

// --- the hidden state after all 28 layers + output_norm ---------------------
// onnx.normedAfterPrompt is exactly what head.onnx was FED: the last shard's
// output, with output_norm already applied (the export applied it when
// is_last; head.onnx never calls the norm it holds). So the WGSL tensor to compare
// is normedState -- output_norm applied once, in the same place.
{
  M.reset();
  const got = await M.normedState(promptIds);
  const ref = Float32Array.from(onnx.normedAfterPrompt);
  const c = cos(got, ref);
  const e = absErrScaled(got, ref, amax(ref));
  console.log(`after 28 layers + output_norm (last prompt position):`);
  console.log(`  WGSL |max| ${amax(got).toFixed(3)}  ONNX |max| ${amax(ref).toFixed(3)}`);
  // The bar is quantization-sized, not ULP-sized: 28 layers of Q8_0 weights against
  // 28 layers of f32 weights. What a wiring bug would do is destroy the DIRECTION,
  // which is why cosine carries the assertion and the magnitude only corroborates.
  ok(
    "WGSL and ONNX agree on the hidden state after 28 layers (cosine ~ 1)",
    c > 0.999,
    `cosine ${c.toFixed(6)}  abs/scale ${e.toExponential(2)}`,
  );
  ok(
    "the magnitudes agree to within quantization",
    Math.abs(amax(got) - amax(ref)) / amax(ref) < 0.05,
    `|max| ${amax(got).toFixed(2)} vs ${amax(ref).toFixed(2)}`,
  );
}

// --- the logits and the argmax ---------------------------------------------
{
  M.reset();
  const logits = await M.logits(promptIds);
  ok(
    "logits are the full vocabulary and all finite",
    logits.length === weights.vocab && logits.every(Number.isFinite),
    `${logits.length} logits, |max| ${amax(logits).toFixed(2)}`,
  );

  // Rank order is what greedy decode consumes, so compare THAT rather than values:
  // a uniform scale error would leave every id right and every value wrong.
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  const want = onnx.generated[0];
  ok(
    "WGSL argmax over the logits picks the token ONNX picked",
    order[0] === want,
    `WGSL ${order[0]}, ONNX ${want}`,
  );

  // The margin is the evidence about how reproducible this is at all. A greedy
  // decode is only deterministic across two different weight quantizations when the
  // top two logits are further apart than the quantization noise.
  const margin = logits[order[0]] - logits[order[1]];
  console.log(
    `\ntop-5 WGSL logits: ${
      order.slice(0, 5).map((i) => `${i}:${logits[i].toFixed(3)}`).join("  ")
    }`,
  );
  console.log(`margin between top two: ${margin.toFixed(4)}\n`);
  // M.logits() ran the head, so the GPU's argmax output is sitting there for the
  // SAME logits the host just scanned. Comparing those isolates the reduction --
  // including its tie rule -- from everything upstream of it.
  const reduced = await M.currentToken();
  ok(
    "the GPU argmax reduction agrees with a host scan of the same logits",
    reduced === order[0],
    `reduction ${reduced}, host scan ${order[0]}`,
  );
}

// --- greedy decode, the whole thing ----------------------------------------
{
  M.reset();
  const t1 = performance.now();
  const gen: number[] = [];
  let next = await M.step(promptIds);
  for (let i = 0; i < MAX_TOKENS; i++) {
    if (next === weights.eos) break;
    gen.push(next);
    next = await M.step([next]);
  }
  const ms = performance.now() - t1;

  console.log(`WGSL generated: ${JSON.stringify(gen)}`);
  console.log(`ONNX generated: ${JSON.stringify(onnx.generated)}`);
  console.log(
    `  ${gen.length} tokens in ${ms.toFixed(0)} ms ` +
      `(${(gen.length / (ms / 1000)).toFixed(1)} tok/s including the prefill)\n`,
  );

  const n = Math.min(gen.length, onnx.generated.length);
  let firstDiff = -1;
  for (let i = 0; i < n; i++) {
    if (gen[i] !== onnx.generated[i]) {
      firstDiff = i;
      break;
    }
  }
  if (firstDiff === -1 && gen.length !== onnx.generated.length) firstDiff = n;

  ok(
    "WGSL greedy decode produces the SAME TOKEN IDS as ONNX",
    firstDiff === -1 && gen.length === onnx.generated.length,
    firstDiff === -1
      ? `${gen.length} tokens identical`
      : `diverged at token ${firstDiff}: WGSL ${gen[firstDiff]} vs ONNX ${
        onnx.generated[firstDiff]
      }`,
  );

  // If they diverged, say WHY with evidence rather than assuming drift. The margin
  // at the divergence point is the number that decides it: a margin smaller than
  // the ~1e-2 relative logit difference Q8_0 accounts for means the two engines were
  // entitled to disagree; a large margin means something is wrong.
  if (firstDiff >= 0) {
    M.reset();
    const prefix = [...promptIds, ...onnx.generated.slice(0, firstDiff)];
    const logits = await M.logits(prefix);
    const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
    const margin = logits[order[0]] - logits[order[1]];
    const onnxChoice = onnx.generated[firstDiff];
    console.log(`divergence at token ${firstDiff}:`);
    console.log(
      `  WGSL top-3: ${order.slice(0, 3).map((i) => `${i}:${logits[i].toFixed(4)}`).join("  ")}`,
    );
    console.log(`  ONNX chose ${onnxChoice}, whose WGSL logit is ${logits[onnxChoice].toFixed(4)}`);
    console.log(`  margin between WGSL's top two: ${margin.toExponential(2)}`);
    console.log(`  |logit| scale: ${amax(logits).toFixed(2)}`);
    const rel = margin / amax(logits);
    console.log(`  margin / scale: ${rel.toExponential(2)}`);
    console.log(
      rel < 2e-2
        ? "  -> a near-tie. Q8_0 vs f32 weights differ by ~1e-2 relative per logit,\n" +
          "     so the two engines are entitled to disagree here. This is drift."
        : "  -> NOT a near-tie. A margin this large is not explained by quantization;\n" +
          "     something is wrong.",
    );
  }

  ok(
    "WGSL stopped at EOS like ONNX did",
    (next === weights.eos) === Boolean(onnx.hitEos),
    `WGSL eos ${next === weights.eos}, ONNX eos ${Boolean(onnx.hitEos)}`,
  );
}

// --- prefill-vs-decode equivalence, on the WHOLE model ---------------------
// test_prefill.ts proves this for one layer. Repeating it for all 28 plus the head
// is what rules out a chaining or cache bug that a single layer cannot show.
{
  M.reset();
  const viaPrefill = await M.step(promptIds);
  M.reset();
  // The same prompt one token at a time. Every intermediate token id is discarded;
  // only the cache state matters, and the last step's answer must match.
  let last = 0;
  for (let i = 0; i < promptIds.length; i++) last = await M.step([promptIds[i]]);
  ok(
    "the whole model: prefill then head == 28 decode steps then head",
    viaPrefill === last,
    `prefill ${viaPrefill}, decode ${last}`,
  );
}

// ============================ how long does agreement last, and why does it end?
//
// The short prompts above match to EOS, but that is a 9-token answer. A 60-token
// generation does diverge, and the value of this section is that it pins down where
// and proves what -- "probably f32 drift" is not an acceptable answer when the
// margin can simply be measured.
//
// The test asserts two things and neither is "the ids match":
//   1. Agreement lasts a long time (tens of tokens), not a handful.
//   2. At the FIRST divergence, the top-two margin is smaller than the
//      quantization difference -- i.e. the two engines were not distinguishable
//      there. That is the claim "this is drift" cashed out as a number.
{
  const longPrompt = "Explain in three sentences why the sky is blue.";
  const lIn = `${tmp}/in3.json`, lOut = `${tmp}/out3.json`;
  await Deno.writeTextFile(
    lIn,
    JSON.stringify({
      ...paths,
      prompt: longPrompt,
      maxTokens: 80,
      wantHidden: false,
    }),
  );
  const p3 = await new Deno.Command("node", {
    args: [new URL("./onnx_full.mjs", import.meta.url).pathname, lIn, lOut],
    stdout: "piped",
    stderr: "piped",
  }).output();
  if (p3.code === 0) {
    const o3 = JSON.parse(await Deno.readTextFile(lOut));
    const refIds: number[] = o3.generated;

    // Generate with WGSL, recording each step's top-two margin so a divergence can
    // be diagnosed from the same run rather than reconstructed.
    M.reset();
    const gen: number[] = [];
    const margins: number[] = [];
    let next = 0;
    for (let i = 0; i <= refIds.length + 5; i++) {
      const lg = i === 0 ? await M.logits(o3.ids) : await M.logits([next]);
      let b0 = -Infinity, i0 = 0, b1 = -Infinity;
      for (let j = 0; j < lg.length; j++) {
        if (lg[j] > b0) {
          b1 = b0;
          b0 = lg[j];
          i0 = j;
        } else if (lg[j] > b1) b1 = lg[j];
      }
      next = i0;
      margins.push((b0 - b1) / amax(lg));
      if (next === weights.eos) break;
      gen.push(next);
    }

    let at = -1;
    for (let i = 0; i < Math.min(gen.length, refIds.length); i++) {
      if (gen[i] !== refIds[i]) {
        at = i;
        break;
      }
    }
    console.log(`\nlong prompt ${JSON.stringify(longPrompt)} (${o3.ids.length} prompt tokens):`);
    console.log(`  ONNX generated ${refIds.length} tokens, WGSL ${gen.length}`);

    // The measured margin floor over the run, which is the context any single
    // divergence has to be read against.
    const sorted = [...margins].sort((a, b) => a - b);
    console.log(
      `  margin/scale over ${margins.length} steps: min ${sorted[0].toExponential(2)}` +
        `  median ${sorted[sorted.length >> 1].toExponential(2)}` +
        `  max ${sorted[sorted.length - 1].toExponential(2)}`,
    );

    ok(
      "long-generation agreement lasts many tokens before any divergence",
      at === -1 || at >= 20,
      at === -1 ? `identical for all ${gen.length}` : `first divergence at token ${at}`,
    );

    if (at >= 0) {
      // Re-run on the ONNX prefix so both engines are in the SAME state, and read
      // the two candidates' logits directly. This is the diagnosis.
      M.reset();
      const lg = await M.logits([...o3.ids, ...refIds.slice(0, at)]);
      const order = Array.from(lg.keys()).sort((a, b) => lg[b] - lg[a]);
      const scale = amax(lg);
      const gap = (lg[order[0]] - lg[refIds[at]]) / scale;
      const rank = order.indexOf(refIds[at]) + 1;
      console.log(
        `  divergence at token ${at}: WGSL ${order[0]} (${lg[order[0]].toFixed(5)}) ` +
          `vs ONNX ${refIds[at]} (${lg[refIds[at]].toFixed(5)})`,
      );
      console.log(`    gap/scale ${gap.toExponential(3)};  ONNX's token is WGSL's rank #${rank}`);

      // THE assertion. The hidden state feeding this projection differs from ONNX's
      // by ~1.2e-2 relative (measured above, and it is quantization). If the gap
      // between the two candidate logits is smaller than that, the engines were not
      // distinguishable at this step and either answer is correct for its weights.
      const QUANT = 1.22e-2;
      ok(
        "the first divergence is a near-tie that quantization fully explains",
        gap < QUANT,
        `gap/scale ${gap.toExponential(2)} < quantization ${QUANT.toExponential(2)} ` +
          `(${(QUANT / gap).toFixed(1)}x margin)`,
      );
      // A wiring bug does not leave the reference's choice at rank 2; it scatters it
      // into the tail of 151936. Rank is the structural half of the diagnosis.
      ok(
        "ONNX's choice is still at the very top of WGSL's ranking",
        rank <= 3,
        `rank #${rank} of ${lg.length}`,
      );
      ok(
        "the divergence happens at the run's tightest margin, not a typical one",
        gap <= sorted[Math.max(0, Math.floor(margins.length * 0.15))],
        `gap ${gap.toExponential(2)} vs p15 margin ` +
          `${sorted[Math.max(0, Math.floor(margins.length * 0.15))].toExponential(2)}`,
      );
    }
  }
}

// A second prompt, so the result is not one lucky string.
{
  const alt = "What is 2 + 2?";
  const altIn = `${tmp}/in2.json`, altOut = `${tmp}/out2.json`;
  await Deno.writeTextFile(
    altIn,
    JSON.stringify({
      ...paths,
      prompt: alt,
      maxTokens: 12,
      wantHidden: false,
    }),
  );
  const p2 = new Deno.Command("node", {
    args: [new URL("./onnx_full.mjs", import.meta.url).pathname, altIn, altOut],
    stdout: "piped",
    stderr: "piped",
  });
  const r2 = await p2.output();
  if (r2.code === 0) {
    const o2 = JSON.parse(await Deno.readTextFile(altOut));
    M.reset();
    const gen: number[] = [];
    let next = await M.step(o2.ids);
    for (let i = 0; i < 12; i++) {
      if (next === weights.eos) break;
      gen.push(next);
      next = await M.step([next]);
    }
    const same = gen.length === o2.generated.length &&
      gen.every((v, i) => v === o2.generated[i]);
    console.log(`\nsecond prompt ${JSON.stringify(alt)}:`);
    console.log(`  WGSL ${JSON.stringify(gen)}`);
    console.log(`  ONNX ${JSON.stringify(o2.generated)}`);
    console.log(`  ONNX text: ${JSON.stringify(o2.text)}`);
    ok(
      `a second prompt also matches token-for-token`,
      same,
      same
        ? `${gen.length} tokens identical`
        : `first difference at ${gen.findIndex((v, i) => v !== o2.generated[i])}`,
    );
  }
}

await Deno.remove(tmp, { recursive: true });
Deno.exit(summary() ? 1 : 0);
