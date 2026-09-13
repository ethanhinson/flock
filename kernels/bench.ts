// Benchmark the matvec kernels at the shapes Qwen3-0.6B actually uses.
//
//   deno run --unstable-webgpu --allow-all kernels/bench.ts
//
// READ THIS BEFORE TRUSTING ANY GPU NUMBER FROM DENO.
//
// `queue.onSubmittedWorkDone()` is the obvious way to wait for a dispatch, and
// on Deno 2.1.2's wgpu backend it DOES NOT WAIT. Calibrated against a kernel
// with a deliberately long dependent loop:
//
//     10,000,000 iterations   fence says 0.13 ms   map readback says 476 ms
//
// Timing anything with that fence measures how fast the host can fill a command
// buffer. It produced a first draft of this file that reported 4400 GFLOP/s for
// a 1024-column Q8_0 matvec -- several times the machine's memory bandwidth, and
// "faster" for 3x the work. Only mapAsync on a buffer the pass wrote is a real
// barrier, so that is what is used here.
//
// A map round-trip costs ~24 ms by itself, which would swamp a 20 us matvec. So
// the unit of measurement is a BATCH: N dispatches into one pass, one map at the
// end, with the map's own cost measured separately and subtracted. N is chosen
// so the batch runs long enough that the residual error in that subtraction does
// not matter.
//
// The second trap, found the same way: dispatch cost has a hard floor of ~16.5 us
// on this backend -- an empty kernel with one workgroup costs the same as a
// 3072x1024 matvec. So a single matvec at these shapes is ENTIRELY dispatch
// overhead, and per-shape "speedups" measured that way are noise. The table
// below reports both numbers: the batched per-matvec time (which measures the
// kernel) and the floor (which says how much of a real single-token forward pass
// is queue overhead instead of arithmetic).
//
// Reported GFLOP/s counts 2 flops per weight, the conventional GEMV count. It
// ignores dequantize work, so it understates what the kernel does.

import {
  coopSource,
  getDevice,
  probeUnpack,
  probeUnpackF16,
  quantMatrixQ4,
  quantMatrixQ8,
  randVec,
  splitQ4,
  splitQ8,
  storageBuffer,
  uniformBuffer,
} from "./lib.ts";

const TRIALS = 5; // timed batches per variant; the median is reported

// attn and ffn projections of Qwen3-0.6B (hidden 1024, ffn 3072).
const SHAPES: [number, number, string][] = [
  [1024, 1024, "attn_k / attn_v"],
  [2048, 1024, "attn_q / attn_output"],
  [3072, 1024, "ffn_gate / ffn_up"],
  [1024, 3072, "ffn_down"],
];

const dev = await getDevice();
const unpack8 = await probeUnpack(dev);
const unpackF16 = await probeUnpackF16(dev);

interface Variant {
  name: string;
  setup(packed: Uint8Array, x: Float32Array, rows: number, cols: number): Run;
}
interface Run {
  dispatch(p: GPUComputePassEncoder): void;
  out: GPUBuffer;
  free(): void;
}

/** q8_matmul.wgsl: one thread per row, byte-extracted from the 34-byte layout. */
function refVariant(code: string, wg: number): Variant {
  const pipe = dev.createComputePipeline({
    layout: "auto",
    compute: { module: dev.createShaderModule({ code }), entryPoint: "main" },
  });
  return {
    name: "reference",
    setup(packed, x, rows, cols) {
      const bufs = [
        storageBuffer(dev, packed),
        storageBuffer(dev, x),
        dev.createBuffer({
          size: rows * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        }),
        uniformBuffer(dev, [rows, cols, cols / 32, 0]),
      ];
      const bg = dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const groups = Math.ceil(rows / wg);
      return {
        dispatch(p) {
          p.setPipeline(pipe);
          p.setBindGroup(0, bg);
          p.dispatchWorkgroups(groups);
        },
        out: bufs[2],
        free() {
          for (const b of bufs) b.destroy();
        },
      };
    },
  };
}

/** q8_coop.wgsl / q4_coop.wgsl: split buffers, vec4 dots, workgroup reduction. */
function coopVariant(
  name: string,
  code: string,
  rowsPerWg: number,
  split: (p: Uint8Array, r: number, c: number) => { qs: Uint8Array; scales: Uint8Array },
): Variant {
  const pipe = dev.createComputePipeline({
    layout: "auto",
    compute: { module: dev.createShaderModule({ code }), entryPoint: "main" },
  });
  return {
    name,
    setup(packed, x, rows, cols) {
      const { qs, scales } = split(packed, rows, cols);
      const bufs = [
        storageBuffer(dev, qs),
        storageBuffer(dev, scales),
        storageBuffer(dev, x),
        dev.createBuffer({
          size: rows * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        }),
        uniformBuffer(dev, [rows, cols, 1, 0]),
      ];
      const bg = dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const groups = Math.ceil(rows / rowsPerWg);
      return {
        dispatch(p) {
          p.setPipeline(pipe);
          p.setBindGroup(0, bg);
          p.dispatchWorkgroups(groups);
        },
        out: bufs[3],
        free() {
          for (const b of bufs) b.destroy();
        },
      };
    },
  };
}

/**
 * The only honest fence available: copy a buffer the pass wrote into a MAP_READ
 * buffer and await mapAsync. Anything cheaper returns before the GPU is done.
 */
async function fencedSubmit(build: (p: GPUComputePassEncoder) => void, out: GPUBuffer) {
  const rd = dev.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
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

/** Cost of the fence itself, so it can be subtracted from a batch. */
async function measureFenceOverhead(): Promise<number> {
  const o = dev.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const ms: number[] = [];
  for (let t = 0; t < 9; t++) {
    const t0 = performance.now();
    await fencedSubmit(() => {}, o);
    ms.push(performance.now() - t0);
  }
  o.destroy();
  ms.sort((a, b) => a - b);
  return ms[ms.length >> 1];
}

/**
 * Median ms for ONE matvec, from a batch of `reps` dispatches behind one fence.
 * Returns the per-dispatch figure with the fence's own cost removed.
 */
async function timeBatched(run: Run, reps: number, fenceMs: number): Promise<number> {
  const build = (p: GPUComputePassEncoder) => {
    for (let i = 0; i < reps; i++) run.dispatch(p);
  };
  await fencedSubmit(build, run.out); // warmup
  const ms: number[] = [];
  for (let t = 0; t < TRIALS; t++) {
    const t0 = performance.now();
    await fencedSubmit(build, run.out);
    ms.push(performance.now() - t0);
  }
  ms.sort((a, b) => a - b);
  return Math.max(0, ms[ms.length >> 1] - fenceMs) / reps;
}

const read = (f: string) => Deno.readTextFile(new URL("./" + f, import.meta.url));
const q8src = await read("q8_coop.wgsl");
const q4src = await read("q4_coop.wgsl");

// Each decode spelling is its own timed variant, so "what did the hardware
// unpacks buy" is a measurement rather than an assumption. Variants this backend
// cannot compile are simply absent from the table.
const decodePaths: { suffix: string; unpack8: boolean; unpackF16: boolean }[] = [];
if (unpack8) decodePaths.push({ suffix: "coop", unpack8: true, unpackF16 });
decodePaths.push({ suffix: unpack8 ? "coop-shift" : "coop", unpack8: false, unpackF16 });
if (unpackF16) decodePaths.push({ suffix: "coop-manualf16", unpack8, unpackF16: false });

const q8Variants: Variant[] = [
  refVariant(await read("q8_matmul.wgsl"), 64),
  ...decodePaths.map((p) => coopVariant(p.suffix, coopSource(q8src, p), 4, splitQ8)),
];
const q4Variants: Variant[] = decodePaths.map(
  (p) => coopVariant(p.suffix, coopSource(q4src, p), 4, splitQ4),
);

const fenceMs = await measureFenceOverhead();
// Enough dispatches that the batch runs for tens of ms, so the fence
// subtraction is a small correction rather than the whole measurement.
const REPS = 2000;

console.log(
  `unpack4xI8/unpack4xU8: ${unpack8 ? "yes" : "NO -- shift/mask fallback is what runs here"}`,
);
console.log(`unpack2x16float:       ${unpackF16 ? "yes" : "no"}`);
console.log(`map fence overhead:    ${fenceMs.toFixed(2)} ms (subtracted)`);
console.log(`${REPS} dispatches per fenced batch, median of ${TRIALS} batches\n`);

interface Row {
  quant: string;
  shape: string;
  variant: string;
  us: number;
}
const rows: Row[] = [];

for (
  const [quant, variants, quantize] of [
    ["Q8_0", q8Variants, quantMatrixQ8],
    ["Q4_0", q4Variants, quantMatrixQ4],
  ] as [string, Variant[], typeof quantMatrixQ8][]
) {
  if (!variants.length) continue;
  for (const [r, c, label] of SHAPES) {
    const w = randVec(r * c, 0.05);
    const packed = quantize(w, r, c);
    const x = randVec(c, 0.1);
    for (const v of variants) {
      const run = v.setup(packed, x, r, c);
      const us = (await timeBatched(run, REPS, fenceMs)) * 1000;
      run.free();
      rows.push({ quant, shape: `${r}x${c}`, variant: v.name, us });
      const gflops = (2 * r * c) / (us * 1e3);
      console.log(
        `${quant}  ${`${r}x${c}`.padEnd(10)} ${v.name.padEnd(15)} ` +
          `${us.toFixed(2).padStart(8)} us  ${gflops.toFixed(1).padStart(7)} GFLOP/s   ${label}`,
      );
    }
  }
  console.log("");
}

// Speedups, computed from the numbers just measured rather than asserted.
const byShape = new Map<string, Row[]>();
for (const r of rows) {
  const k = `${r.quant} ${r.shape}`;
  byShape.set(k, [...(byShape.get(k) ?? []), r]);
}
console.log("speedup vs reference kernel:");
for (const [k, rs] of byShape) {
  const base = rs.find((r) => r.variant === "reference");
  if (!base) continue;
  for (const r of rs) {
    if (r === base) continue;
    console.log(`  ${k.padEnd(16)} ${r.variant.padEnd(15)} ${(base.us / r.us).toFixed(2)}x`);
  }
}

// Context for the numbers above: how much of a real single-token matvec is queue
// overhead rather than arithmetic? A batched dispatch hides this; a forward pass
// issuing one dispatch per projection does not.
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
      }),
      entryPoint: "main",
    },
  });
  const bg = dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: o } }],
  });
  const run: Run = {
    dispatch(p) {
      p.setPipeline(pipe);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(1);
    },
    out: o,
    free() {
      o.destroy();
    },
  };
  const us = (await timeBatched(run, REPS, fenceMs)) * 1000;
  console.log(`\nempty 1-workgroup dispatch: ${us.toFixed(2)} us`);
  console.log("  This is the per-dispatch floor. Any matvec cheaper than it is");
  console.log("  invisible in a one-dispatch-per-op forward pass.");
  run.free();
}
