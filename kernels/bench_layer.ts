// How long does one layer's decode step take, and where does the time go?
//
//   deno run --unstable-webgpu --allow-all kernels/bench_layer.ts
//
// This is the number that decides whether the WGSL engine is viable, so it is
// measured the same careful way bench.ts is -- see the long note there about
// onSubmittedWorkDone() not waiting on Deno's wgpu, and about the ~17.6 us
// per-dispatch floor. layerForward already ends in a map readback, which is a
// real fence, so timing it end to end is honest without extra machinery.
//
// The interesting question is not just the total but the split: a decode step is
// 7 matvecs plus 9 small dispatches, and bench.ts measured the batched cost of
// those matvecs at 24-37 us each. If the total is far above the sum, the time is
// going to dispatch overhead and queue round-trips rather than arithmetic -- which
// changes what is worth optimizing next.

import { getDevice, randVec } from "./lib.ts";
import { Layer, QWEN3_06B } from "./layer.ts";
import { realLayer } from "./real_weights.ts";

const dev = await getDevice();
const cfg = QWEN3_06B;
const w = await realLayer(24);
const layer = await Layer.create(dev, w, cfg);

const TRIALS = 30;
const x = randVec(cfg.hidden, 0.05);

// Warm up: first call pays pipeline compilation and buffer residency.
for (let i = 0; i < 3; i++) { layer.reset(); await layer.forward(x); }

// Time steady-state decode at a few cache depths. Attention is the only part
// whose cost grows with the cache, so this says how much.
console.log("one layer, one token, real Qwen3-0.6B blk.24 weights\n");
console.log("cache depth   ms/step   tok/s (this layer)   est. 28-layer tok/s");
for (const depth of [1, 128, 512, 1024]) {
  layer.reset();
  // Fill the cache to `depth` without timing it.
  for (let i = 0; i < depth - 1; i++) await layer.forward(x);

  const ms: number[] = [];
  for (let t = 0; t < TRIALS; t++) {
    // Hold the depth fixed by rewinding the counter: the cache contents stay,
    // only the write position repeats, which is what keeps nKeys constant.
    layer.nKeys = depth - 1;
    const t0 = performance.now();
    await layer.forward(x);
    ms.push(performance.now() - t0);
  }
  ms.sort((a, b) => a - b);
  const med = ms[ms.length >> 1];
  console.log(
    `${String(depth).padStart(11)}   ${med.toFixed(3).padStart(7)}   ` +
    `${(1000 / med).toFixed(1).padStart(18)}   ${(1000 / (med * 28)).toFixed(2).padStart(19)}`,
  );
}

// Where does the time actually go? forward() ends in a map readback, and the
// readback is the thing being measured above -- not the layer. Compare against
// encode(), which queues the identical 16 dispatches and does not wait.
{
  const timeOf = async (fn: () => Promise<void> | void) => {
    for (let i = 0; i < 3; i++) { layer.nKeys = 0; await fn(); }
    const ms: number[] = [];
    for (let t = 0; t < TRIALS; t++) {
      layer.nKeys = 0;
      const t0 = performance.now();
      await fn();
      ms.push(performance.now() - t0);
    }
    ms.sort((a, b) => a - b);
    return ms[ms.length >> 1];
  };

  layer.reset();
  const withReadback = await timeOf(() => layer.forward(x).then(() => {}));
  // N encodes behind ONE readback, each submitting its own command buffer.
  const N = 28;
  const chained = await timeOf(async () => {
    for (let i = 0; i < N; i++) { layer.nKeys = i; layer.encode(i === 0 ? x : undefined); }
    await layer.readOutput();
  });
  // N encodes sharing ONE command buffer. This is what a shard should do, and it
  // is a large win on top of sharing the fence: submitting per layer means the
  // driver validates and dispatches N times instead of once.
  const batched = await timeOf(async () => {
    const enc = dev.createCommandEncoder();
    for (let i = 0; i < N; i++) { layer.nKeys = i; layer.encode(i === 0 ? x : undefined, enc); }
    dev.queue.submit([enc.finish()]);
    await layer.readOutput();
  });

  // Matvec figures from bench.ts, measured batched and therefore excluding
  // per-dispatch overhead. Sum of the seven this layer issues.
  const matvecMs = (30.5 + 24.7 + 24.7 + 24.7 + 36.9 + 36.9 + 30.0) / 1000;
  console.log("\ncost per layer, by how much the host batches:");
  console.log(`  forward()          own command buffer, own readback   ${withReadback.toFixed(3)} ms`);
  console.log(`  encode() x${N}      own command buffer, one readback    ${(chained / N).toFixed(3)} ms` +
    `   (${(withReadback / (chained / N)).toFixed(1)}x)`);
  console.log(`  encode() x${N}      ONE command buffer, one readback    ${(batched / N).toFixed(3)} ms` +
    `   (${(withReadback / (batched / N)).toFixed(1)}x)`);
  console.log(`\n  floor for reference:`);
  console.log(`    sum of this layer's 16 dispatches, timed individually: ~0.72 ms`);
  console.log(`    the 7 matvecs alone (bench.ts):                        ${matvecMs.toFixed(3)} ms`);
  console.log(`\n  Two things cost more than the arithmetic, and both are host-side:`);
  console.log(`  the map readback (~${(withReadback - chained / N).toFixed(0)} ms, so never per layer -- keep the hidden`);
  console.log(`  state on the GPU and read once per shard), and per-layer command`);
  console.log(`  buffer submission (${((chained - batched) / N).toFixed(2)} ms/layer, so share one encoder).`);
  console.log(`\n  A 28-layer pass projects to ${(1000 / batched).toFixed(1)} tok/s batched, versus`);
  console.log(`  ${(1000 / (28 * withReadback)).toFixed(2)} tok/s with a readback and a submit per layer.`);
}
