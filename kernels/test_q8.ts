// Validate q8_matmul.wgsl (the reference kernel) against a STRICT f32 CPU
// reference, then against REAL GGUF bytes range-fetched from HuggingFace.
//
// The assertion is equality, not a tolerance. Once the reference models what the
// hardware actually does -- f32 rounding after every step, and multiply-adds
// contracted into single-rounding FMAs -- the GPU and the CPU agree exactly, so
// there is no error budget to argue about and any real bug shows up as a
// non-zero number. lib.ts explains how; this file just uses it.
//
// The middle test keeps the trap that motivated all of this on the record: the
// same GPU output compared against an f64 accumulator is off by ~6e-5, which is
// f32 summation losing that much on its own, not the kernel being wrong.
//
//   deno run --unstable-webgpu --allow-all --config kernels/deno.json kernels/test_q8.ts

import {
  SERIAL, cpuMatmulQ8F32, getDevice, maxRelErr, ok, quantMatrixQ8, randVec,
  readBack, storageBuffer, summary, uniformBuffer, Q8_BLOCK,
} from "./lib.ts";
import { realQ8Tensor } from "./real_weights.ts";

const dev = await getDevice();
const code = await Deno.readTextFile(new URL("./q8_matmul.wgsl", import.meta.url));
const pipe = dev.createComputePipeline({
  layout: "auto",
  compute: { module: dev.createShaderModule({ code }), entryPoint: "main" },
});

async function gpuMatmul(packed: Uint8Array, x: Float32Array, rows: number, cols: number) {
  const wbuf = storageBuffer(dev, packed);
  const xbuf = storageBuffer(dev, x);
  const obuf = dev.createBuffer({
    size: rows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const dbuf = uniformBuffer(dev, [rows, cols, cols / Q8_BLOCK, 0]);
  const bg = dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: wbuf } }, { binding: 1, resource: { buffer: xbuf } },
      { binding: 2, resource: { buffer: obuf } }, { binding: 3, resource: { buffer: dbuf } },
    ],
  });
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(pipe); p.setBindGroup(0, bg);
  p.dispatchWorkgroups(Math.ceil(rows / 64)); p.end();
  dev.queue.submit([enc.finish()]);
  const out = await readBack(dev, obuf, rows * 4);
  for (const b of [wbuf, xbuf, obuf, dbuf]) b.destroy();
  return out;
}

// --- synthetic -------------------------------------------------------------
// The reference kernel is one thread per row with a single serial accumulator,
// so the CPU reference uses SERIAL to match that summation order exactly.
for (const [rows, cols] of [[64, 32], [128, 1024], [1024, 1024], [3072, 1024], [1024, 3072]]) {
  const w = randVec(rows * cols, 0.05);
  const packed = quantMatrixQ8(w, rows, cols);
  const x = randVec(cols);
  const cpu = cpuMatmulQ8F32(packed, x, rows, cols, SERIAL);
  const gpu = await gpuMatmul(packed, x, rows, cols);
  const err = maxRelErr(gpu, cpu);
  ok(`${rows}x${cols} vs strict f32 CPU`, err === 0, `max rel err ${err.toExponential(1)}`);
}

// --- the trap, demonstrated ------------------------------------------------
// Same GPU output, same data, compared against an f64 accumulator instead.
// This number is the phantom bug: it is the CPU reference being *more*
// accurate than f32, not the kernel being wrong.
{
  const rows = 1024, cols = 1024;
  const w = randVec(rows * cols, 0.05);
  const packed = quantMatrixQ8(w, rows, cols);
  const x = randVec(cols);
  const gpu = await gpuMatmul(packed, x, rows, cols);
  const strict = cpuMatmulQ8F32(packed, x, rows, cols, SERIAL);
  const f64 = cpuMatmulF64(packed, x, rows, cols);
  const eStrict = maxRelErr(gpu, strict), eF64 = maxRelErr(gpu, f64);
  ok("f32 reference is tighter than f64 reference", eStrict < eF64 / 10,
    `strict ${eStrict.toExponential(1)} vs f64 ${eF64.toExponential(1)}`);
}

function cpuMatmulF64(packed: Uint8Array, x: Float32Array, rows: number, cols: number) {
  const nb = cols / 32, out = new Float32Array(rows);
  const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    for (let b = 0; b < nb; b++) {
      const base = r * nb * 34 + b * 34;
      const h = dv.getUint16(base, true);
      const s = (h & 0x8000 ? -1 : 1) * (1 + (h & 0x3ff) / 1024) * Math.pow(2, ((h >>> 10) & 0x1f) - 15);
      let sum = 0;
      for (let i = 0; i < 32; i++) {
        let q = packed[base + 2 + i]; if (q > 127) q -= 256;
        sum += q * x[b * 32 + i];
      }
      acc += s * sum;
    }
    out[r] = acc;
  }
  return out;
}

// --- real weights ----------------------------------------------------------
{
  const t = await realQ8Tensor("blk.24.attn_k.weight");
  const x = randVec(t.cols, 0.1);   // activations in flock are small
  const cpu = cpuMatmulQ8F32(t.packed, x, t.rows, t.cols, SERIAL);
  const gpu = await gpuMatmul(t.packed, x, t.rows, t.cols);
  const err = maxRelErr(gpu, cpu);
  ok(`real ${t.name} ${t.rows}x${t.cols}`, err === 0, `max rel err ${err.toExponential(1)}`);
}

Deno.exit(summary() ? 1 : 0);
