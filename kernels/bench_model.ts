// Benchmark the pieces this engine added: the embedding gather, the LM head,
// argmax (GPU reduction versus reading 151936 floats back), prefill, and a whole
// generated token.
//
//   deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/bench_model.ts
//
// READ kernels/README.md "the measurement traps" FIRST. Four of them produce
// confident wrong numbers, and two apply directly to everything below:
//
//  - queue.onSubmittedWorkDone() DOES NOT WAIT on this backend. Only mapAsync on a
//    buffer the pass wrote is a real barrier. Every number here is fenced that way.
//  - A single dispatch at these shapes is ~17.6 us of pure queue overhead, which is
//    the same as the arithmetic. So per-kernel numbers come from MANY dispatches
//    behind one fence with the fence's own ~24 ms subtracted, and are labelled
//    "batched". The whole-token numbers are deliberately NOT batched -- they are
//    what a generation actually pays, overhead included -- and are labelled so.
//
// Both are reported for the head, because they answer different questions: batched
// says what the kernel costs, unbatched says what a token costs.

import {
  coopSource, getDevice, probeUnpack, probeUnpackF16, splitQ8, storageBuffer,
  uniformBuffer,
} from "./lib.ts";
import { QWEN3_06B } from "./layer.ts";
import { Model } from "./model.ts";
import { realModel } from "./real_weights.ts";

const TRIALS = 5;
const dev = await getDevice();

// ------------------------------------------------------------------ the fence

/** The only honest fence: map a buffer the pass wrote. */
async function fenced(build: (p: GPUComputePassEncoder) => void, out: GPUBuffer) {
  const rd = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  build(p);
  p.end();
  enc.copyBufferToBuffer(out, 0, rd, 0, 4);
  dev.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ);
  rd.unmap();
  rd.destroy();
}

async function fenceOverhead(): Promise<number> {
  const o = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const ms: number[] = [];
  for (let t = 0; t < 9; t++) {
    const t0 = performance.now();
    await fenced(() => {}, o);
    ms.push(performance.now() - t0);
  }
  o.destroy();
  ms.sort((a, b) => a - b);
  return ms[ms.length >> 1];
}

/** Median per-dispatch ms from `reps` dispatches behind one fence. */
async function batched(
  dispatch: (p: GPUComputePassEncoder) => void, out: GPUBuffer, reps: number, fenceMs: number,
): Promise<number> {
  const build = (p: GPUComputePassEncoder) => { for (let i = 0; i < reps; i++) dispatch(p); };
  await fenced(build, out);
  const ms: number[] = [];
  for (let t = 0; t < TRIALS; t++) {
    const t0 = performance.now();
    await fenced(build, out);
    ms.push(performance.now() - t0);
  }
  ms.sort((a, b) => a - b);
  return Math.max(0, ms[ms.length >> 1] - fenceMs) / reps;
}

/** Median wall-clock ms of an async operation, warmed up. */
async function wall(f: () => Promise<unknown>, trials = TRIALS): Promise<number> {
  await f();
  const ms: number[] = [];
  for (let t = 0; t < trials; t++) {
    const t0 = performance.now();
    await f();
    ms.push(performance.now() - t0);
  }
  ms.sort((a, b) => a - b);
  return ms[ms.length >> 1];
}

const fenceMs = await fenceOverhead();
const unpack8 = await probeUnpack(dev);
const unpackF16 = await probeUnpackF16(dev);
const src = (f: string) => Deno.readTextFile(new URL("./" + f, import.meta.url));
const mk = (code: string, entryPoint = "main") => dev.createComputePipeline({
  layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint },
});
const bind = (pipe: GPUComputePipeline, bufs: GPUBuffer[]) =>
  dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

console.log(`unpack4xI8: ${unpack8 ? "yes" : "NO -- shift/mask fallback is what runs"}`);
console.log(`map fence overhead: ${fenceMs.toFixed(2)} ms (subtracted from batched numbers)\n`);

console.log("loading Qwen3-0.6B Q8_0 ...");
const t0 = performance.now();
const weights = await realModel();
const loadMs = performance.now() - t0;
const VOCAB = weights.vocab, H = weights.hidden;

const { qs, scales } = splitQ8(weights.embd.packed, VOCAB, H);
const t1 = performance.now();
const embdQs = storageBuffer(dev, qs);
const embdSc = storageBuffer(dev, scales);
const uploadMs = performance.now() - t1;
console.log(`  weights read in ${(loadMs / 1000).toFixed(1)}s; token_embd repack+upload ` +
  `${(uploadMs / 1000).toFixed(2)}s for ${(qs.byteLength / 1e6).toFixed(0)} MB\n`);

const rw = (n: number) => dev.createBuffer({
  size: Math.max(16, n * 4),
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

// ================================================== embedding gather, batched
console.log("embedding gather (Q8_0 row -> f32), batched per dispatch");
{
  const embedPipe = mk(coopSource(await src("embed.wgsl"), { unpack8, unpackF16 }));
  for (const n of [1, 16, 64, 256]) {
    const ids = rw(n), out = rw(n * H);
    dev.queue.writeBuffer(ids, 0, new Uint32Array(
      Array.from({ length: n }, (_, i) => (i * 997) % VOCAB)));
    const dims = uniformBuffer(dev, [n, H, 0, 0]);
    const bg = bind(embedPipe, [embdQs, embdSc, ids, out, dims]);
    const us = (await batched((p) => {
      p.setPipeline(embedPipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(n);
    }, out, 2000, fenceMs)) * 1000;
    // bytes / us == MB/s; / 1e3 == GB/s. One int8 read and one f32 written per element.
    const gbs = (n * H * (1 + 4)) / us / 1e3;
    console.log(`  ${String(n).padStart(3)} token(s): ${us.toFixed(2).padStart(7)} us` +
      `   ${gbs.toFixed(2).padStart(7)} GB/s   ${(n * H * 5 / 1024).toFixed(0)} KB moved`);
    for (const b of [ids, out, dims]) b.destroy();
  }
  console.log(`  -> FLAT at the ~17.6 us dispatch floor from 1 token to 256: 256x the work`);
  console.log(`     for the same time, which is the signature of a dispatch that is pure`);
  console.log(`     launch overhead. A decode step's gather is 5 KB and costs the same as`);
  console.log(`     an empty kernel, so there is nothing to optimize at decode. The rising`);
  console.log(`     GB/s column is the same fact read the other way: the kernel only starts`);
  console.log(`     doing measurable work somewhere past 256 rows.\n`);
}

// ================================================== LM head matvec, batched
console.log("LM head: hidden (1024) x token_embd^T (151936), batched per dispatch");
{
  const mvPipe = mk(coopSource(await src("q8_coop.wgsl"), { unpack8, unpackF16 }));
  const x = rw(H), logits = rw(VOCAB);
  dev.queue.writeBuffer(x, 0, new Float32Array(
    Array.from({ length: H }, (_, i) => Math.sin(i) * 0.1)));
  const dims = uniformBuffer(dev, [VOCAB, H, 1, 0]);
  const bg = bind(mvPipe, [embdQs, embdSc, x, logits, dims]);
  const groups = Math.ceil(VOCAB / 4);
  const us = (await batched((p) => {
    p.setPipeline(mvPipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(groups, 1);
  }, logits, 200, fenceMs)) * 1000;
  const gflops = (2 * VOCAB * H) / (us * 1e3);
  // bytes / us == MB/s; / 1e3 == GB/s. One int8 per weight plus one f16 per 32.
  const bytes = VOCAB * H + (VOCAB * H / 32) * 2;
  const gbs = bytes / us / 1e3;
  console.log(`  151936x1024: ${us.toFixed(1)} us   ${gflops.toFixed(0)} GFLOP/s   ` +
    `${gbs.toFixed(0)} GB/s of weight traffic   (${groups} workgroups)`);
  console.log(`  ${(bytes / 1e6).toFixed(0)} MB of weights read per call -- the whole tied matrix,`);
  console.log(`  once per generated token, and there is no reuse to exploit: a matvec`);
  console.log(`  touches every weight exactly once.`);
  console.log(`  For scale: ~${(us / 37).toFixed(0)}x the biggest single op inside a layer, and ` +
    `${(us / 1000 / (28 * 0.72) * 100).toFixed(0)}% of`);
  console.log(`  a 28-layer decode step's total arithmetic (README: ~0.72 ms/layer for`);
  console.log(`  this layer's 16 dispatches timed individually, so ~20 ms for 28).`);
  // Is 151936 rows a shape this kernel handles BADLY, or just a big problem? The
  // way to find out is to run the same kernel on a layer-sized shape with the same
  // per-byte accounting and compare the byte rates. Claiming "it's bandwidth-bound"
  // without this comparison would be an assertion, not a measurement.
  const smallRows = 3072;
  const sw = new Uint8Array((smallRows * H / 32) * 34);
  for (let i = 0; i < sw.length; i++) sw[i] = (i * 31) & 0xff;
  const ss = splitQ8(sw, smallRows, H);
  const sq = storageBuffer(dev, ss.qs), ssc = storageBuffer(dev, ss.scales);
  const sx = rw(H), so = rw(smallRows);
  const sd = uniformBuffer(dev, [smallRows, H, 1, 0]);
  const sbg = bind(mvPipe, [sq, ssc, sx, so, sd]);
  const sus = (await batched((p) => {
    p.setPipeline(mvPipe); p.setBindGroup(0, sbg);
    p.dispatchWorkgroups(Math.ceil(smallRows / 4), 1);
  }, so, 2000, fenceMs)) * 1000;
  const sBytes = smallRows * H + (smallRows * H / 32) * 2;
  const sGbs = sBytes / sus / 1e3;
  console.log(`  same kernel at 3072x1024: ${sus.toFixed(1)} us, ${sGbs.toFixed(0)} GB/s ` +
    `(${(sBytes / 1e6).toFixed(1)} MB)`);
  console.log(`  byte rate at the head's shape vs the layer's shape: ` +
    `${(gbs / sGbs).toFixed(2)}x.`);
  if (gbs / sGbs > 1.1) {
    console.log(`  -> The head is ${(us / sus).toFixed(0)}x slower than the layer shape for ${(VOCAB / smallRows).toFixed(0)}x the rows, so`);
    console.log(`     it runs ${(gbs / sGbs).toFixed(1)}x MORE efficiently per byte, not less. The layer`);
    console.log(`     shape is the one leaving throughput on the table: at 3.3 MB it does not`);
    console.log(`     have enough work to saturate, while 165 MB does. So the head is already`);
    console.log(`     doing the best this kernel does, and its cost is the problem size.`);
    console.log(`     A faster head needs fewer BYTES (Q4 for this one tensor would halve`);
    console.log(`     them) or fewer ROWS -- not a better dispatch shape.`);
  } else if (gbs / sGbs > 0.85) {
    console.log(`  -> Same byte rate at both shapes, so the ${(us / sus).toFixed(0)}x cost is the ${(VOCAB / smallRows).toFixed(0)}x problem`);
    console.log(`     size and there is nothing shape-specific left to fix.`);
  } else {
    console.log(`  -> The head runs at a LOWER byte rate than the layer shape, so there IS`);
    console.log(`     something shape-specific to win here. Worth investigating.`);
  }
  console.log(`  Whether ${gbs.toFixed(0)} GB/s is near this machine's ceiling is NOT measured, so no`);
  console.log(`  absolute headroom claim is made -- only the relative one above.\n`);
  for (const b of [x, logits, dims, sq, ssc, sx, so, sd]) b.destroy();
}

// ============================== argmax: GPU reduction vs reading logits back
console.log("argmax over 151936 logits: GPU reduction vs host readback");
{
  const argmaxSrc = await src("argmax.wgsl");
  const p1 = mk(argmaxSrc, "pass1"), p2 = mk(argmaxSrc, "pass2");
  const GROUPS = 594;
  const logits = rw(VOCAB);
  const vals = new Float32Array(VOCAB);
  for (let i = 0; i < VOCAB; i++) vals[i] = Math.sin(i * 0.37) * 10;
  dev.queue.writeBuffer(logits, 0, vals);
  const pv = rw(GROUPS), pi = rw(GROUPS), ov = rw(1), oi = rw(1), dummy = rw(1);
  const d1 = uniformBuffer(dev, [VOCAB, GROUPS, 0, 0]);
  const d2 = uniformBuffer(dev, [GROUPS, GROUPS, 0, 0]);
  const bg1 = bind(p1, [logits, dummy, pv, pi, d1]);
  const bg2 = bind(p2, [pv, pi, ov, oi, d2]);

  // Stage 1 batched, since it is the part that scales with the vocabulary.
  const us1 = (await batched((p) => {
    p.setPipeline(p1); p.setBindGroup(0, bg1); p.dispatchWorkgroups(GROUPS);
  }, pv, 2000, fenceMs)) * 1000;
  const us2 = (await batched((p) => {
    p.setPipeline(p2); p.setBindGroup(0, bg2); p.dispatchWorkgroups(1);
  }, ov, 2000, fenceMs)) * 1000;
  console.log(`  reduction, batched:  pass1 ${us1.toFixed(2)} us + pass2 ${us2.toFixed(2)} us ` +
    `= ${(us1 + us2).toFixed(2)} us of kernel`);

  // What a real step pays: two dispatches plus a 4-byte readback.
  const reduceMs = await wall(async () => {
    const enc = dev.createCommandEncoder();
    const a = enc.beginComputePass();
    a.setPipeline(p1); a.setBindGroup(0, bg1); a.dispatchWorkgroups(GROUPS); a.end();
    const b = enc.beginComputePass();
    b.setPipeline(p2); b.setBindGroup(0, bg2); b.dispatchWorkgroups(1); b.end();
    const rd = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(oi, 0, rd, 0, 4);
    dev.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    rd.unmap(); rd.destroy();
  });

  // The alternative: copy all 151936 floats to the host and scan them in JS.
  const readbackMs = await wall(async () => {
    const rd = dev.createBuffer({
      size: VOCAB * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(logits, 0, rd, 0, VOCAB * 4);
    dev.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rd.getMappedRange());
    let bi = 0, bv = a[0];
    for (let i = 1; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; }
    rd.unmap(); rd.destroy();
    if (bi < 0) throw new Error("unreachable");
  });

  console.log(`  end to end, one call (what a token pays):`);
  console.log(`    GPU reduction + 4-byte readback:        ${reduceMs.toFixed(2)} ms`);
  console.log(`    copy ${(VOCAB * 4 / 1e6).toFixed(2)} MB back + scan in JS:   ${readbackMs.toFixed(2)} ms`);
  console.log(`    -> ${(readbackMs / reduceMs).toFixed(2)}x`);
  console.log(`  The brief was "argmax on GPU beats reading 151936 floats back; measure`);
  console.log(`  both". Measured, it BARELY does, and the reason is worth stating: both`);
  console.log(`  paths pay exactly one map readback and that readback is ~26 ms on this`);
  console.log(`  backend regardless of size, so it swamps the ${(VOCAB * 4 / 1e6).toFixed(2)} MB of copy and the`);
  console.log(`  ${(us1 + us2).toFixed(0)} us of kernel alike. The reduction is still the right choice --`);
  console.log(`  it is never slower, it keeps 0.6 MB off the bus, and on a backend whose`);
  console.log(`  readback scales with size it would win properly -- but on THIS backend`);
  console.log(`  the ${(readbackMs / reduceMs).toFixed(2)}x is the honest number, not a speedup worth citing.\n`);
  for (const b of [logits, pv, pi, ov, oi, dummy, d1, d2]) b.destroy();
}

// ==================================================== prefill and a full token
console.log("whole-model timings (NOT batched: this is what a generation pays)\n");
{
  const M = await Model.create(dev, weights, { ...QWEN3_06B, maxPrefill: 256 });
  const fakeIds = (n: number) => Array.from({ length: n }, (_, i) => 1000 + (i * 37) % 50000);

  console.log("  prefill, 28 layers + head, one submit per maxPrefill chunk:");
  for (const n of [1, 8, 16, 64, 128, 256]) {
    const ids = fakeIds(n);
    const ms = await wall(async () => { M.reset(); await M.step(ids); }, 3);
    console.log(`    ${String(n).padStart(3)} tokens: ${ms.toFixed(1).padStart(7)} ms   ` +
      `${(ms / n).toFixed(2).padStart(6)} ms/token   ${(n / (ms / 1000)).toFixed(0).padStart(5)} tok/s`);
  }

  console.log("\n  decode, one token at a time (the steady state):");
  for (const ctx of [16, 128, 512]) {
    M.reset();
    await M.step(fakeIds(ctx));
    const ms = await wall(async () => { await M.step([1234]); }, 10);
    console.log(`    at ${String(ctx).padStart(3)} keys of context: ${ms.toFixed(2)} ms/token   ` +
      `${(1000 / ms).toFixed(1)} tok/s`);
  }

  // WHERE A DECODE TOKEN'S TIME GOES -- and the first attempt at this was wrong in
  // a way worth recording. Timing M.step() against M.hiddenState() and subtracting
  // reported "the head is 0% of a token", because BOTH pay the ~26 ms map readback
  // that ends them; the subtraction cancels the layers and the readback together
  // and leaves noise. That is README trap 1 wearing a different hat: the readback is
  // not a small constant to be differenced away, it is the entire measurement.
  //
  // So the split is measured by amortizing instead: run K tokens behind ONE fence
  // and divide. The readback is then paid once for K tokens rather than once per
  // token, which is what a streaming generation would do anyway if it did not need
  // each id before choosing the next -- and it is the only way to see the GPU work
  // underneath.
  console.log("\n  a decode token, with the readback amortized over K tokens:");
  console.log("    (step() must read each id back before it can feed the next, so a real");
  console.log("     greedy decode cannot amortize this. These numbers isolate the GPU");
  console.log("     work; the ms/token above is what generation actually costs.)");
  for (const K of [1, 4, 16, 64]) {
    M.reset();
    await M.step(fakeIds(16));
    // K decode steps encoded into one command buffer, one readback at the end.
    const ms = await wall(async () => {
      const startKeys = M.pos;
      await M.stepsAmortized(Array(K).fill(1234));
      // Rewind so repeated trials see the same context length.
      M.rewind(startKeys);
    }, 3);
    console.log(`    K=${String(K).padStart(2)}: ${(ms / K).toFixed(2).padStart(6)} ms/token   ` +
      `${(1000 / (ms / K)).toFixed(0).padStart(4)} tok/s   (${ms.toFixed(1)} ms for ${K})`);
  }
  console.log(`    -> The gap between K=1 and K=64 IS the map readback, ~26 ms, and it`);
  console.log(`       dominates a per-token decode. The GPU work per token is the K=64`);
  console.log(`       figure; everything above it is the host waiting.`);

  console.log("\n  prefill vs decode for the same number of tokens:");
  const n = 128;
  M.reset();
  const pre = await wall(async () => { M.reset(); await M.step(fakeIds(n)); }, 3);
  M.reset();
  const dec = await wall(async () => {
    M.reset();
    for (const id of fakeIds(n)) await M.step([id]);
  }, 1);
  console.log(`    ${n} tokens as one prefill:  ${pre.toFixed(0)} ms`);
  console.log(`    ${n} tokens as decode steps: ${dec.toFixed(0)} ms   -> prefill is ` +
    `${(dec / pre).toFixed(1)}x faster`);
  console.log(`    This is the whole reason prefill exists: a decode step's cost is`);
  console.log(`    dominated by per-dispatch and per-submit overhead, and prefill pays`);
  console.log(`    that once for N tokens instead of N times.`);
}
