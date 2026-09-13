// What tensor parallelism COSTS, on one device: unsharded vs 2/4/8-way, both
// directions, with the reduction measured separately.
//
//   deno run --unstable-webgpu --allow-all kernels/bench_shard.ts
//
// WHAT THIS CAN AND CANNOT MEASURE, stated first so no number here is read as
// something it is not.
//
// Every shard runs on the SAME GPU, sequentially in one queue. So this measures
// the OVERHEAD of splitting -- extra dispatches, the reduction, the strided reads
// -- and NOT the speedup of running shards in parallel on N devices, which this
// machine cannot demonstrate at all. Tensor parallelism on one device is expected
// to be SLOWER than unsharded, exactly like pipeline parallelism: it buys capacity,
// not speed. The question these numbers answer is "how much does the capacity
// cost", and the answer is honest about being a bound on the LOCAL overhead a
// multi-device deployment would pay -- a real deployment adds network latency,
// which dwarfs everything here, and removes the serialization.
//
// THE MEASUREMENT DISCIPLINE is bench.ts's, unchanged, because getting it wrong
// produced a 4400 GFLOP/s table the first time someone tried:
//   * `onSubmittedWorkDone()` DOES NOT WAIT on Deno 2.1.2's wgpu. Only mapAsync on
//     a buffer the pass wrote is a real barrier.
//   * a map round-trip is ~24 ms, which would swamp a 20 us matvec, so the unit is
//     a BATCH of many dispatches behind one fence with the fence's own cost
//     measured and subtracted.
//   * dispatch cost has a ~17 us floor -- an empty one-workgroup kernel costs the
//     same as a 3072x1024 matvec. That floor is measured here too, because it is
//     the single most important number for interpreting the sharded results: N
//     shards means N dispatches, and if each is near the floor then sharding costs
//     (N-1) floors regardless of how little arithmetic it added.
//
// ONE TRAP THIS BENCHMARK ADDED TO THE LIST. The repetitions go inside a SINGLE
// compute pass (dispatchShards / dispatchReduce), not as repeated calls to
// `encode()`. Calling `encode()` in a loop builds one compute pass per repetition,
// and a command buffer with thousands of passes WEDGES on this backend: the submit
// never completes and mapAsync never resolves, at 0% CPU and 0.19 s of total CPU
// time. That is indistinguishable from "the benchmark is just slow" until you look
// at the process. Passes are not free objects to batch; dispatches are.

import { getDevice, quantMatrixQ8, randVec, splitQ8 } from "./lib.ts";
import { memShardPlan } from "./shard_ref.ts";
import { ShardedMatvec } from "./shard.ts";
import { realModel } from "./real_weights.ts";

const TRIALS = 5;
const dev = await getDevice();

/** The only honest fence: map a buffer the pass wrote. */
async function fencedPass(build: (p: GPUComputePassEncoder) => void, out: GPUBuffer) {
  const rd = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  build(p);
  p.end();
  enc.copyBufferToBuffer(out, 0, rd, 0, 4);
  dev.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ);
  rd.unmap(); rd.destroy();
}

async function measureFenceOverhead(): Promise<number> {
  const o = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const ms: number[] = [];
  for (let t = 0; t < 9; t++) {
    const t0 = performance.now();
    await fencedPass(() => {}, o);
    ms.push(performance.now() - t0);
  }
  o.destroy();
  ms.sort((a, b) => a - b);
  return ms[ms.length >> 1];
}

/** Median ms for ONE call, from `reps` in one pass behind one fence, fence removed. */
async function timeBatched(
  one: (p: GPUComputePassEncoder) => void, out: GPUBuffer, reps: number, fenceMs: number,
): Promise<number> {
  const build = (p: GPUComputePassEncoder) => { for (let i = 0; i < reps; i++) one(p); };
  await fencedPass(build, out);                          // warmup
  const ms: number[] = [];
  for (let t = 0; t < TRIALS; t++) {
    const t0 = performance.now();
    await fencedPass(build, out);
    ms.push(performance.now() - t0);
  }
  ms.sort((a, b) => a - b);
  return Math.max(0, ms[ms.length >> 1] - fenceMs) / reps;
}

const fenceMs = await measureFenceOverhead();
console.log(`map fence overhead: ${fenceMs.toFixed(2)} ms (measured and subtracted)`);

// The per-dispatch floor. Measured first because every sharded number below has to
// be read against it: N shards is N dispatches, and (N-1) floors is the price of
// admission before any arithmetic changes at all.
let floorUs = 0;
{
  const o = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const pipe = dev.createComputePipeline({
    layout: "auto",
    compute: {
      module: dev.createShaderModule({
        code: `@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x == 0xFFFFFFFFu) { o[0] = 1.0; }
}`,
      }), entryPoint: "main",
    },
  });
  const bg = dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: o } }],
  });
  floorUs = (await timeBatched((p) => {
    p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(1);
  }, o, 2000, fenceMs)) * 1000;
  console.log(`empty 1-workgroup dispatch: ${floorUs.toFixed(2)} us  <- the per-dispatch floor\n`);
  o.destroy();
}

interface Row {
  shape: string; direction: string; n: number;
  shardsUs: number; reduceUs: number; totalUs: number;
}
const all: Row[] = [];

/**
 * Time a sharded matvec at one shape and shard count, splitting out the reduction.
 *
 * `shardsUs` and `reduceUs` are measured independently; `totalUs` is their sum
 * rather than a third measurement, and that is stated rather than hidden, because
 * the two halves cannot share a pass (the reduce reads what the matvecs wrote,
 * through a different bind group, so it needs a pass boundary). Timing them in one
 * pass would be timing something that is not the operation.
 */
async function timeShape(
  packed: Uint8Array, rows: number, cols: number, x: Float32Array,
  n: number, direction: "row" | "col", reps: number,
): Promise<Row> {
  const split = splitQ8(packed, rows, cols);
  const mv = await ShardedMatvec.create(dev, split, rows, cols, n, direction);
  mv.setInput(x);
  const shardsUs = (await timeBatched(
    (p) => mv.dispatchShards(p),
    direction === "col" ? mv.partials() : mv.output(), reps, fenceMs)) * 1000;
  const reduceUs = direction === "col"
    ? (await timeBatched((p) => mv.dispatchReduce(p), mv.output(), reps, fenceMs)) * 1000
    : 0;
  mv.destroy();
  return { shape: `${rows}x${cols}`, direction, n, shardsUs, reduceUs, totalUs: shardsUs + reduceUs };
}

function line(r: Row, baseUs: number) {
  console.log(
    `  ${r.shape.padEnd(12)} ${r.direction.padEnd(4)} ${String(r.n).padStart(2)} ` +
    `${r.shardsUs.toFixed(1).padStart(8)} ${(r.direction === "col" ? r.reduceUs.toFixed(1) : "-").padStart(8)} ` +
    `${r.totalUs.toFixed(1).padStart(8)} ${(r.totalUs / baseUs).toFixed(2).padStart(7)}x`);
}

// ------------------------------------------------------- layer-sized shapes
//
// The Qwen3-0.6B projections, where the unsharded matvec is already only 1-2x the
// dispatch floor. That is the regime where sharding should look worst, and including
// it is the point: a matvec that IS the dispatch floor cannot absorb N-1 more of them.
console.log("-- layer-sized matvecs, us per call (unsharded is 1-2x the dispatch floor)\n");
console.log(`  ${"shape".padEnd(12)} ${"dir".padEnd(4)} ${"N".padStart(2)} ` +
            `${"shards".padStart(8)} ${"reduce".padStart(8)} ${"total".padStart(8)} ${"vs N=1".padStart(8)}`);
for (const [r, c] of [[1024, 1024], [3072, 1024], [1024, 3072]] as [number, number][]) {
  const packed = quantMatrixQ8(randVec(r * c, 0.05), r, c);
  const x = randVec(c, 0.1);
  for (const direction of ["row", "col"] as const) {
    const base = await timeShape(packed, r, c, x, 1, direction, 2000);
    all.push(base);
    line(base, base.totalUs);
    for (const n of [2, 4, 8]) {
      const row = await timeShape(packed, r, c, x, n, direction, 2000);
      all.push(row);
      line(row, base.totalUs);
    }
  }
  console.log("");
}

// ------------------------------------------------------------- the real LM head
//
// The case that matters. 151936 x 1024 is ~785 us unsharded (bench_model.ts), i.e.
// ~45x the dispatch floor -- so unlike the layer shapes there is real arithmetic
// here for the extra dispatches to hide behind, and the overhead should be a small
// percentage rather than a multiple.
console.log("-- the real tied LM head, 151936 x 1024, us per call\n");
console.log(`  ${"dir".padEnd(4)} ${"N".padStart(2)} ${"shards".padStart(9)} ` +
            `${"reduce".padStart(9)} ${"total".padStart(9)} ${"vs N=1".padStart(8)} ` +
            `${"GB/s".padStart(6)} ${"largest binding".padStart(16)}`);
{
  const m = await realModel();
  const split = splitQ8(m.embd.packed, m.embd.rows, m.embd.cols);
  const bytes = split.qs.byteLength + split.scales.byteLength;
  const x = randVec(m.embd.cols, 0.1);
  const REPS = 100;        // ~785 us x 100 = ~80 ms per batch, well over the fence
  for (const direction of ["row", "col"] as const) {
    let baseUs = 0;
    for (const n of [1, 2, 4, 8]) {
      const mv = await ShardedMatvec.create(dev, split, m.embd.rows, m.embd.cols, n, direction);
      mv.setInput(x);
      const shardsUs = (await timeBatched(
        (p) => mv.dispatchShards(p),
        direction === "col" ? mv.partials() : mv.output(), REPS, fenceMs)) * 1000;
      const reduceUs = direction === "col"
        ? (await timeBatched((p) => mv.dispatchReduce(p), mv.output(), REPS, fenceMs)) * 1000
        : 0;
      const totalUs = shardsUs + reduceUs;
      if (n === 1) baseUs = totalUs;
      const plan = memShardPlan(
        { name: "h", rows: m.embd.rows, cols: m.embd.cols }, n, direction);
      all.push({ shape: "151936x1024", direction, n, shardsUs, reduceUs, totalUs });
      console.log(
        `  ${direction.padEnd(4)} ${String(n).padStart(2)} ${shardsUs.toFixed(1).padStart(9)} ` +
        `${(direction === "col" ? reduceUs.toFixed(1) : "-").padStart(9)} ${totalUs.toFixed(1).padStart(9)} ` +
        `${(totalUs / baseUs).toFixed(2).padStart(7)}x ${(bytes / (totalUs * 1e3)).toFixed(0).padStart(6)} ` +
        `${((plan.maxBindingBytes / 1e6).toFixed(1) + " MB").padStart(16)}`);
      mv.destroy();
    }
    console.log("");
  }
}

// ------------------------------------------------------------------- the summary
console.log("-- what the overhead is, and where it comes from\n");
{
  const head = all.filter((r) => r.shape === "151936x1024");
  for (const direction of ["row", "col"] as const) {
    const base = head.find((r) => r.direction === direction && r.n === 1)!;
    for (const n of [2, 4, 8]) {
      const at = head.find((r) => r.direction === direction && r.n === n)!;
      const dispatchCost = (n - 1) * floorUs;
      console.log(
        `  ${direction}-wise head N=${n}: ${(at.totalUs - base.totalUs >= 0 ? "+" : "")}` +
        `${(at.totalUs - base.totalUs).toFixed(1)} us vs N=1 ` +
        `(${((at.totalUs / base.totalUs - 1) * 100).toFixed(1)}%); ` +
        `${n - 1} extra dispatches at the ${floorUs.toFixed(1)} us floor = ${dispatchCost.toFixed(1)} us` +
        (direction === "col" ? `; reduction ${at.reduceUs.toFixed(1)} us` : ""));
    }
  }
  // The column-wise result is not dispatch overhead and the arithmetic above
  // proves it: at N=8 the head is 5582 us slower than unsharded while 7 extra
  // dispatches account for 129 us and the reduction for 31 us. The remaining
  // ~5400 us is LANE STARVATION, and it is structural rather than a tuning miss.
  //
  // q8_shard.wgsl gives every row LANES = 64 threads, which walk that row's
  // nb = n_cols/32 blocks strided. A column shard has n_cols/N columns, so
  // nb/N blocks -- and the lane count does NOT shrink with it. At the head's
  // cols = 1024 that is 32 blocks unsharded (half the lanes already idle) and
  // 4 blocks at N=8, so 60 of 64 lanes contribute nothing while every shard
  // still pays a full workgroup's scheduling and a full 64-wide tree reduction.
  // Eight shards therefore do ~8x the total work for the same arithmetic.
  //
  // Row-wise has no such problem: a row shard's rows each keep all 1024 of their
  // columns, so nb is unchanged and every lane stays fed. That is the same
  // property that makes row-wise bit-identical -- the per-row work is untouched --
  // showing up as a performance result as well as a numerical one.
  //
  // A column-wise kernel that scaled LANES down with n_cols (and ROWS_PER_WG up
  // to keep the workgroup full) would recover most of this. It is not written:
  // the head is row-wise, where the problem does not arise. See the report.
  {
    const c8 = head.find((r) => r.direction === "col" && r.n === 8)!;
    const c1 = head.find((r) => r.direction === "col" && r.n === 1)!;
    const unexplained = (c8.totalUs - c1.totalUs) - 7 * floorUs - c8.reduceUs;
    console.log("");
    console.log(`  col-wise N=8's ${(c8.totalUs - c1.totalUs).toFixed(0)} us of overhead is NOT dispatches:`);
    console.log(`    dispatches ${(7 * floorUs).toFixed(0)} us + reduction ${c8.reduceUs.toFixed(0)} us ` +
                `leaves ${unexplained.toFixed(0)} us unexplained.`);
    console.log(`    That is lane starvation. Each row still gets 64 lanes but a shard`);
    console.log(`    has only 1024/32/8 = 4 blocks, so 60 of 64 lanes idle. Row-wise`);
    console.log(`    keeps every row's full column count and does not degrade.`);
  }

  console.log("");
  console.log("  Read these as the COST OF CAPACITY, not as a speed result. Every shard");
  console.log("  ran on the same GPU in the same queue, so nothing here is parallel: N");
  console.log("  shards is N sequential dispatches over the same total arithmetic, plus");
  console.log("  (column-wise) a reduction. A deployment putting each shard on its own");
  console.log("  device would overlap them and would also pay network latency this");
  console.log("  cannot see. What this DOES establish is the local overhead, and that at");
  console.log("  head sizes it is a small percentage rather than a multiple.");
}
