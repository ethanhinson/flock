// A row-wise sharded LM head picks the same token as the unsharded one.
//
//   deno run --unstable-webgpu --allow-all kernels/test_shard_head.ts
//
// THIS IS THE MOTIVATING CASE, not a synthetic one. The LM head is the largest
// single tensor in any of these models and the one that cannot be split by layer
// range: Qwen3-0.6B's is tied to token_embd (151936 x 1024, 155.6 MB repacked)
// and Qwen3-14B's `output.weight` is 638 MB with tie_word_embeddings: False. A
// device whose adapter caps maxStorageBufferBindingSize at the 128 MiB default
// cannot bind either one.
//
// WHAT IS ASSERTED, in increasing strength:
//
//   1. The sharded logits are BIT-IDENTICAL to the unsharded logits, all 151936
//      of them -- not just the argmax. Row-wise makes this achievable (see
//      test_shard.ts), and it matters because argmax discards magnitude: two
//      engines can agree on a token while disagreeing about everything else.
//   2. The sharded argmax picks the same token as argmaxRef over the unsharded
//      logits. argmaxRef is head_ref.ts's, unchanged -- the reference this suite
//      already uses for the unsharded head.
//   3. The tie rule survives the split, on CONSTRUCTED ties rather than hoped-for
//      ones, including a tie that straddles a shard boundary. That is the case a
//      sharded argmax can get wrong while looking right on random data.
//   4. A whole greedy generation over the real model agrees token for token, and
//      the sharded head runs under a binding limit the unsharded one cannot.

import {
  getDevice, ok, quantMatrixQ8, randVec, readBack, splitQ8, summary,
} from "./lib.ts";
import { argmaxRef } from "./head_ref.ts";
import { QWEN3_06B } from "./layer.ts";
import { Model } from "./model.ts";
import { realModel } from "./real_weights.ts";
import { memShardPlan, reduceShardArgmax, shardRanges } from "./shard_ref.ts";
import { ShardedHead } from "./shard.ts";

const dev = await getDevice();

/** Run a sharded head over one hidden state and return (logits, token id). */
async function runHead(
  head: ShardedHead, h: Float32Array,
): Promise<{ logits: Float32Array; token: number }> {
  head.setInput(h);
  const enc = dev.createCommandEncoder();
  head.encode(enc);
  dev.queue.submit([enc.finish()]);
  const logits = await readBack(dev, head.logits(), head.vocab * 4);
  // The per-shard winners. In a real deployment these 8 bytes per shard are what
  // crosses the network; the 608 KB of logits above is read only by this test.
  const parts = [];
  for (const b of head.argmaxBuffers()) {
    const v = await readBack(dev, b.oVal, 4);
    const i = new Uint32Array((await readBack(dev, b.oIdx, 4)).buffer);
    parts.push({ value: v[0], index: i[0], rowOffset: b.rowOffset });
  }
  return { logits, token: reduceShardArgmax(parts) };
}

// ------------------------------------------------- synthetic, every shard count
//
// A small vocab first, so every shard count including awkward ones runs quickly
// and the exactness claim is checked densely rather than once.
console.log("-- synthetic head, vocab 4096 x hidden 256\n");
{
  const vocab = 4096, hidden = 256;
  const w = randVec(vocab * hidden, 0.05);
  const packed = quantMatrixQ8(w, vocab, hidden);
  const split = splitQ8(packed, vocab, hidden);
  const h = randVec(hidden, 0.5);

  // The N = 1 head IS the unsharded head: one shard covering every row, through
  // the same kernel. Using it as the reference rather than q8_coop.wgsl would be
  // circular, so the reference is built from the whole tensor through the
  // UNSHARDED path -- Model's own matvec pipeline is not reachable standalone, so
  // this uses N=1 for the logits baseline and argmaxRef (an independent CPU
  // implementation) for the token. test_shard.ts already pins the N=1 kernel
  // against cpuMatmulQ8F32 bit-for-bit at these shapes.
  const base = await ShardedHead.create(dev, split, vocab, hidden, { shards: 1 });
  const ref = await runHead(base, h);
  const refToken = argmaxRef(ref.logits);
  ok("the 1-shard head's own argmax agrees with argmaxRef", ref.token === refToken,
     `sharded ${ref.token}, argmaxRef ${refToken}`);

  for (const n of [2, 3, 4, 5, 7, 8]) {
    const head = await ShardedHead.create(dev, split, vocab, hidden, { shards: n });
    const got = await runHead(head, h);
    let diff = 0;
    for (let i = 0; i < vocab; i++) diff = Math.max(diff, Math.abs(got.logits[i] - ref.logits[i]));
    ok(`N=${n}: all ${vocab} logits bit-identical to unsharded`, diff === 0,
       `max |diff| ${diff.toExponential(1)}  (rows ${shardRanges(vocab, n, 1).map((r) => r.count).join("/")})`);
    ok(`N=${n}: picks the same token as argmaxRef`, got.token === refToken,
       `sharded ${got.token}, ref ${refToken}`);
    head.destroy();
  }
  base.destroy();
}

// ------------------------------------------------------ ties, constructed
//
// The tie rule is "lowest index wins" and it is a CONTRACT, not an implementation
// detail: numpy and torch both return the lowest index, so a sharded argmax that
// broke ties differently would make a divergence diff against ONNX report a
// spurious disagreement the first time two logits land on the same f32. Rare, but
// not hypothetical.
//
// Constructed rather than hoped for. Random logits essentially never tie, so a
// test on random data proves nothing about the rule. The planted case that matters
// is a tie that STRADDLES A SHARD BOUNDARY: two equal maxima in different shards,
// where a host-side reduction that compared shards in the wrong order, or added
// the row offset after comparing instead of before, would pick the higher index.
console.log("\n-- tie-breaking across a shard boundary\n");
{
  const vocab = 2048, hidden = 64;
  const n = 4;
  const ranges = shardRanges(vocab, n, 1);
  // Rows whose weight is a single 1.0 in column 0 produce logit = x[0] * 1.0, so
  // planting equal rows plants exactly equal logits -- no reliance on random
  // values happening to collide.
  const cases: { name: string; rows: number[]; want: number }[] = [
    // A tie between shard 1 and shard 3: the lower global index must win, which
    // is in the EARLIER shard.
    { name: "tie across shards 1 and 3", rows: [ranges[1].start + 5, ranges[3].start + 9],
      want: ranges[1].start + 5 },
    // A tie in the LAST shard against the FIRST: shard 0 must win even though the
    // reduction walks shards in increasing order and would otherwise be free to
    // keep the later one.
    { name: "tie across shards 0 and 3", rows: [ranges[0].start + 1, ranges[3].end - 1],
      want: ranges[0].start + 1 },
    // A tie WITHIN one shard, which exercises argmax.wgsl's own rule under the
    // shard's shifted indexing rather than the host reduction.
    { name: "tie inside shard 2", rows: [ranges[2].start + 3, ranges[2].start + 400],
      want: ranges[2].start + 3 },
    // Exactly at a boundary: the last row of shard 1 against the first of shard 2.
    { name: "tie at the shard 1/2 boundary", rows: [ranges[1].end - 1, ranges[2].start],
      want: ranges[1].end - 1 },
  ];
  const x = new Float32Array(hidden);
  x[0] = 7.0;
  for (const c of cases) {
    const w = new Float32Array(vocab * hidden);
    // Every other row gets a small positive weight so the planted rows are the
    // maxima but the rest are not all identically zero (which would make the whole
    // vector a tie and test nothing).
    for (let r = 0; r < vocab; r++) w[r * hidden] = 0.1 + (r % 17) * 0.01;
    for (const r of c.rows) w[r * hidden] = 1.0;
    const packed = quantMatrixQ8(w, vocab, hidden);
    const split = splitQ8(packed, vocab, hidden);
    const head = await ShardedHead.create(dev, split, vocab, hidden, { shards: n });
    const got = await runHead(head, x);
    // First confirm the tie is real: a "tie test" on logits that are not actually
    // equal proves nothing, and quantization could have separated them.
    const tied = got.logits[c.rows[0]] === got.logits[c.rows[1]];
    const refToken = argmaxRef(got.logits);
    ok(`${c.name}: the planted logits really are equal in f32`, tied,
       `${got.logits[c.rows[0]]} vs ${got.logits[c.rows[1]]}`);
    ok(`${c.name}: lowest index wins`, got.token === c.want && refToken === c.want,
       `sharded ${got.token}, argmaxRef ${refToken}, want ${c.want}`);
    head.destroy();
  }
}

// ------------------------------------------------- the real tied head, 151936 rows
console.log("\n-- Qwen3-0.6B's real tied head, 151936 x 1024\n");
const m = await realModel();
const cfg = QWEN3_06B;
const embdSplit = splitQ8(m.embd.packed, m.embd.rows, m.embd.cols);

// The unsharded engine, to produce the hidden state and the reference logits. This
// is the configuration test_model.ts validates token for token against ONNX, so
// the reference here is the one with the ONNX proof behind it.
const whole = await Model.create(dev, m, cfg);
const PROMPT = [151644, 872, 198, 63593, 315, 9625, 30, 151645, 198,
                151644, 77091, 198, 151667, 271, 151668, 271];

const refLogits = await whole.logits(PROMPT);
const refToken = argmaxRef(refLogits);
const wholeToken = await whole.currentToken();
ok("the unsharded engine's GPU argmax agrees with argmaxRef on the real logits",
   wholeToken === refToken, `GPU ${wholeToken}, argmaxRef ${refToken}`);
ok("and that token is the one the ONNX comparison pins (785)", refToken === 785,
   `got ${refToken}`);

// The hidden state AFTER output_norm: the sharded head's job starts there, since
// output_norm is one 1024-element op that does not need splitting.
//
// reset() FIRST, and this is not defensive tidiness. `logits(PROMPT)` above already
// pushed the prompt through and left `pos` at 16; calling normedState(PROMPT) on
// that same instance would run the prompt a SECOND time at positions 16..31, with
// the first pass's keys still in the cache. The result is a perfectly valid hidden
// state for a different input, and the logits comparison below then fails by 5.6 at
// EVERY shard count -- identically, because the discrepancy is in the input and not
// in the split. Which is the diagnostic: an error whose magnitude does not move
// with N is not a sharding error. Caught exactly this way.
whole.reset();
const normed = await whole.normedState(PROMPT);

for (const n of [2, 4, 8]) {
  const head = await ShardedHead.create(dev, embdSplit, m.vocab, m.embd.cols, { shards: n });
  const got = await runHead(head, normed);
  let diff = 0;
  for (let i = 0; i < m.vocab; i++) diff = Math.max(diff, Math.abs(got.logits[i] - refLogits[i]));
  ok(`real head N=${n}: all 151936 logits bit-identical to the unsharded engine`,
     diff === 0, `max |diff| ${diff.toExponential(1)}`);
  ok(`real head N=${n}: picks token ${refToken}, the same as argmaxRef`,
     got.token === refToken, `sharded ${got.token}, ref ${refToken}`);
  const bytes = head.shardBytes();
  console.log(`     shard bytes: ${bytes.map((b) => (b / 1e6).toFixed(1)).join(" / ")} MB` +
              `  (unsharded ${((m.vocab * m.embd.cols + m.vocab * (m.embd.cols / 32) * 2) / 1e6).toFixed(1)} MB)`);
  head.destroy();
}

// --------------------------- a whole generation, token for token, through shards
//
// One token cannot see much: it is the easy case, and the logits comparison above
// already covers it densely. A generation is the statement that nothing
// accumulates -- that the sharded head fed back into the layer stack stays in step
// with the unsharded engine over many steps.
console.log("\n-- greedy generation, unsharded head vs a 4-way sharded head\n");
{
  const N = 12;
  const shardHead = await ShardedHead.create(
    dev, embdSplit, m.vocab, m.embd.cols, { shards: 4 });

  // Two independent engines so the KV caches cannot alias. Each runs the SAME
  // layer stack; only the head differs -- unsharded on `a`, 4-way sharded on `b`.
  const a = await Model.create(dev, m, cfg);
  const b = await Model.create(dev, m, cfg);
  const idsA: number[] = [], idsB: number[] = [];

  // `normedState(ids)` runs the layers for `ids` (advancing b's KV cache exactly
  // once) and returns the last position's state after output_norm. Calling it once
  // per step and feeding the result to the sharded head is the whole loop -- the
  // bug worth guarding against here is advancing b's cache twice per token, which
  // would make b's attention read keys for positions it never emitted.
  let ta = await a.step(PROMPT);
  let tb = (await runHead(shardHead, await b.normedState(PROMPT))).token;

  for (let i = 0; i < N; i++) {
    idsA.push(ta); idsB.push(tb);
    if (ta === m.eos || tb === m.eos) break;
    ta = await a.step([ta]);
    const nb = await b.normedState([tb]);
    tb = (await runHead(shardHead, nb)).token;
  }
  console.log(`  unsharded head: ${JSON.stringify(idsA)}`);
  console.log(`  4-way sharded:  ${JSON.stringify(idsB)}\n`);
  const same = idsA.length === idsB.length && idsA.every((x, i) => x === idsB[i]);
  ok(`greedy decode is identical with a 4-way sharded head for ${idsA.length} tokens`,
     same, same ? `${idsA.length} tokens identical`
                : `first difference at ${idsA.findIndex((x, i) => x !== idsB[i])}`);
  shardHead.destroy();
}

// ------------------------------ the head fits where the unsharded one does not
//
// The capacity claim, on the real tensor. 155.6 MB of qs against the 128 MiB
// default limit: the unsharded head must be refused and the sharded one must run.
{
  const LIMIT = 128 * 1024 * 1024;
  const unshardedPlan = memShardPlan(
    { name: "token_embd.weight", rows: m.vocab, cols: m.embd.cols }, 1, "row", LIMIT);
  ok("the unsharded tied head does NOT fit the 128 MiB default binding limit",
     !unshardedPlan.fits,
     `${(unshardedPlan.maxBindingBytes / 1e6).toFixed(1)} MB > ${(LIMIT / 1e6).toFixed(1)} MB`);
  ok(`memShardPlan says ${unshardedPlan.minShardsForLimit} shards are enough`,
     unshardedPlan.minShardsForLimit === 2, `${unshardedPlan.minShardsForLimit}`);

  // And it actually runs under that limit, producing the right token. This is the
  // whole deliverable in one assertion: a head that a default-limits device cannot
  // bind at all, running correctly split.
  const head = await ShardedHead.create(
    dev, embdSplit, m.vocab, m.embd.cols,
    { shards: unshardedPlan.minShardsForLimit!, bindingLimit: LIMIT });
  const got = await runHead(head, normed);
  ok(`a ${unshardedPlan.minShardsForLimit}-way head runs under a 128 MiB limit and picks ${refToken}`,
     got.token === refToken, `got ${got.token}`);
  head.destroy();

  let threw = "";
  try {
    await ShardedHead.create(dev, embdSplit, m.vocab, m.embd.cols,
      { shards: 1, bindingLimit: LIMIT });
  } catch (e) { threw = String((e as Error).message); }
  ok("a 1-shard head under a 128 MiB limit is refused before upload",
     threw.includes("over the"), threw || "(did not throw)");
}

Deno.exit(summary() ? 1 : 0);
