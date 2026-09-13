// Validate the cooperative kernels (q8_coop.wgsl, q4_coop.wgsl) against a
// strict-f32 CPU reference and against real GGUF weights, on BOTH decode paths.
//
//   deno run --unstable-webgpu --allow-all kernels/test_coop.ts
//
// Two things this test is built around:
//
// The reference uses coop(LANES), not SERIAL. A cooperative kernel sums in a
// different order than one-thread-per-row: 64 independent partials, tree-reduced.
// That is a *more* accurate f32 summation, so comparing it against the serial
// reference flags a correct kernel. Match the shape and the agreement is exact.
//
// Both decode spellings are tested wherever the device supports them. The
// hardware-unpack path and the shift/mask path must produce bit-identical
// output -- that is the whole justification for validating on one and shipping
// the other. On Deno's wgpu only the fallback compiles (no unpack4xI8), so the
// builtin path is reported as skipped rather than silently assumed.

import {
  coop,
  coopSource,
  cpuMatmulQ4F32,
  cpuMatmulQ8F32,
  getDevice,
  maxRelErr,
  ok,
  probeUnpack,
  probeUnpackF16,
  quantMatrixQ4,
  quantMatrixQ8,
  randVec,
  readBack,
  splitQ4,
  splitQ8,
  storageBuffer,
  summary,
  uniformBuffer,
} from "./lib.ts";
import { absErrScaled, amax } from "./ops_ref.ts";
import { realQ8Tensor } from "./real_weights.ts";

const LANES = 64, ROWS_PER_WG = 4;

const dev = await getDevice();
const unpack8 = await probeUnpack(dev);
const unpackF16 = await probeUnpackF16(dev);
console.log(
  `unpack4xI8/unpack4xU8: ${unpack8 ? "yes" : "no"}   unpack2x16float: ${
    unpackF16 ? "yes" : "no"
  }\n`,
);

// Every combination this device can actually compile. The f16 axis is tested
// separately from the 8-bit axis because they are independently available --
// Deno has one and not the other.
const paths: { label: string; unpack8: boolean; unpackF16: boolean }[] = [];
for (const u8 of unpack8 ? [true, false] : [false]) {
  for (const uf of unpackF16 ? [true, false] : [false]) {
    paths.push({
      label: `i8=${u8 ? "builtin" : "shift"} f16=${uf ? "builtin" : "manual"}`,
      unpack8: u8,
      unpackF16: uf,
    });
  }
}
if (!unpack8) {
  console.log("  skip  unpack4xI8 path: not implemented by this backend\n");
}

function makePipeline(src: string, opts: { unpack8: boolean; unpackF16: boolean }) {
  const code = coopSource(src, opts);
  return dev.createComputePipeline({
    layout: "auto",
    compute: { module: dev.createShaderModule({ code }), entryPoint: "main" },
  });
}

async function run(
  pipe: GPUComputePipeline,
  qs: Uint8Array,
  scales: Uint8Array,
  x: Float32Array,
  rows: number,
  cols: number,
) {
  const bufs = [
    storageBuffer(dev, qs),
    storageBuffer(dev, scales),
    storageBuffer(dev, x),
    dev.createBuffer({ size: rows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
    uniformBuffer(dev, [rows, cols, 1, 0]),
  ];
  const bg = dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(pipe);
  p.setBindGroup(0, bg);
  p.dispatchWorkgroups(Math.ceil(rows / ROWS_PER_WG));
  p.end();
  dev.queue.submit([enc.finish()]);
  const out = await readBack(dev, bufs[3], rows * 4);
  for (const b of bufs) b.destroy();
  return out;
}

// Shapes: the Qwen3-0.6B projections, plus two that are deliberately awkward.
// 30 rows is not a multiple of ROWS_PER_WG, so the last workgroup has idle rows
// that must still reach every barrier; 64x32 is a single block per row, so most
// of the 64 lanes contribute nothing and the tree reduction is summing zeros.
const SHAPES: [number, number][] = [
  [64, 32],
  [30, 1024],
  [128, 1024],
  [1024, 1024],
  [2048, 1024],
  [3072, 1024],
  [1024, 3072],
];

const q8src = await Deno.readTextFile(new URL("./q8_coop.wgsl", import.meta.url));
const q4src = await Deno.readTextFile(new URL("./q4_coop.wgsl", import.meta.url));

for (const path of paths) {
  console.log(`-- ${path.label}`);
  const q8pipe = makePipeline(q8src, path);
  const q4pipe = makePipeline(q4src, path);

  for (const [rows, cols] of SHAPES) {
    const w = randVec(rows * cols, 0.05);
    const x = randVec(cols);

    const p8 = quantMatrixQ8(w, rows, cols);
    const s8 = splitQ8(p8, rows, cols);
    const g8 = await run(q8pipe, s8.qs, s8.scales, x, rows, cols);
    const c8 = cpuMatmulQ8F32(p8, x, rows, cols, coop(LANES));
    const e8 = maxRelErr(g8, c8);
    ok(`Q8_0 ${rows}x${cols}`, e8 === 0, `rel err ${e8.toExponential(1)}`);

    const p4 = quantMatrixQ4(w, rows, cols);
    const s4 = splitQ4(p4, rows, cols);
    const g4 = await run(q4pipe, s4.qs, s4.scales, x, rows, cols);
    const c4 = cpuMatmulQ4F32(p4, x, rows, cols, coop(LANES));
    const e4 = maxRelErr(g4, c4);
    ok(`Q4_0 ${rows}x${cols}`, e4 === 0, `rel err ${e4.toExponential(1)}`);
  }

  // Real weights. Synthetic data is uniform, so every block gets a similar
  // scale and the quants fill the int8 range evenly; real rows have scales that
  // vary by orders of magnitude and blocks that are nearly all zero.
  for (const name of ["blk.24.attn_k.weight", "blk.24.ffn_gate.weight", "blk.24.ffn_down.weight"]) {
    const t = await realQ8Tensor(name);
    const x = randVec(t.cols, 0.1);
    const s = splitQ8(t.packed, t.rows, t.cols);
    const gpu = await run(q8pipe, s.qs, s.scales, x, t.rows, t.cols);
    const cpu = cpuMatmulQ8F32(t.packed, x, t.rows, t.cols, coop(LANES));
    const err = maxRelErr(gpu, cpu);
    ok(`real ${name} ${t.rows}x${t.cols}`, err === 0, `rel err ${err.toExponential(1)}`);
  }
  console.log("");
}

// The coop kernel and the reference kernel are different summation orders of the
// same sum, so they will NOT agree bit-for-bit -- but they must agree to f32's
// noise floor. This is the check that catches a coop kernel reading the wrong
// weights entirely, which a reference sharing its own indexing could not.
//
// Judged on absolute error against the output scale, not per-element relative
// error. A matvec output is a sum of 1024 signed terms, so some rows land near
// zero by cancellation and report a large relative error over ordinary rounding
// noise -- measured 3.9e-5 on one random draw and 2.2e-4 on the next, from the
// same correct kernel. The absolute figure is stable.
{
  const rows = 1024, cols = 1024;
  const w = randVec(rows * cols, 0.05), x = randVec(cols);
  const packed = quantMatrixQ8(w, rows, cols);
  const s = splitQ8(packed, rows, cols);
  const pipe = makePipeline(q8src, paths[0]);
  const gpu = await run(pipe, s.qs, s.scales, x, rows, cols);
  const serial = cpuMatmulQ8F32(packed, x, rows, cols);
  const err = absErrScaled(gpu, serial, amax(serial));
  ok(
    "coop agrees with the serial kernel to f32 noise",
    err > 0 && err < 1e-6,
    `abs err / scale ${err.toExponential(1)} (nonzero is expected: different summation order)`,
  );
}

// Repacking must be lossless: same bytes, different arrangement.
{
  const rows = 37, cols = 1024;
  const w = randVec(rows * cols, 0.05);
  const packed = quantMatrixQ8(w, rows, cols);
  const { qs, scales } = splitQ8(packed, rows, cols);
  const nb = cols / 32;
  let bad = 0;
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < nb; b++) {
      const src = r * nb * 34 + b * 34;
      if (scales[(r * nb + b) * 2] !== packed[src]) bad++;
      if (scales[(r * nb + b) * 2 + 1] !== packed[src + 1]) bad++;
      for (let i = 0; i < 32; i++) {
        if (qs[r * cols + b * 32 + i] !== packed[src + 2 + i]) bad++;
      }
    }
  }
  ok("splitQ8 preserves every byte", bad === 0, `${bad} mismatches`);
}

Deno.exit(summary() ? 1 : 0);
