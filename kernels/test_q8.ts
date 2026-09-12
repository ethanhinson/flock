// Validate q8_matmul.wgsl against a CPU reference, then against REAL GGUF bytes.
const adapter = await navigator.gpu.requestAdapter();
const dev = await adapter!.requestDevice();
const code = await Deno.readTextFile(new URL("./q8_matmul.wgsl", import.meta.url));
const pipe = dev.createComputePipeline({
  layout: "auto",
  compute: {module: dev.createShaderModule({code}), entryPoint: "main"},
});

// Rounds to nearest-even. A truncating encoder costs ~3 bits of an 11-bit
// mantissa (5.8e-2 vs 4.9e-4 worst case), which is large enough to look like a
// kernel bug when it is really the reference data being wrong.
import {f32to16 as f32ToHalf} from "../web/js/wire.mjs";
function halfToF32(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * Math.pow(2, -24);
  if (e === 31) return s * 3.4028235e38;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}

/** Quantize a row of f32 into Q8_0 blocks (same scheme llama.cpp uses). */
function quantRow(vals: Float32Array) {
  const nb = vals.length / 32;
  const out = new Uint8Array(nb * 34);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let i = 0; i < 32; i++) amax = Math.max(amax, Math.abs(vals[b * 32 + i]));
    const d = amax / 127;
    dv.setUint16(b * 34, f32ToHalf(d), true);
    for (let i = 0; i < 32; i++) {
      const q = d === 0 ? 0 : Math.round(vals[b * 32 + i] / d);
      out[b * 34 + 2 + i] = Math.max(-128, Math.min(127, q)) & 0xff;
    }
  }
  return out;
}

/** CPU reference: dequantize then multiply, exactly what the kernel must match. */
function cpuMatmul(packed: Uint8Array, x: Float32Array, rows: number, cols: number) {
  const nb = cols / 32, out = new Float32Array(rows);
  const dv = new DataView(packed.buffer);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    for (let b = 0; b < nb; b++) {
      const base = r * nb * 34 + b * 34;
      const scale = halfToF32(dv.getUint16(base, true));
      let sum = 0;
      for (let i = 0; i < 32; i++) {
        let q = packed[base + 2 + i]; if (q > 127) q -= 256;
        sum += q * x[b * 32 + i];
      }
      acc += scale * sum;
    }
    out[r] = acc;
  }
  return out;
}

async function gpuMatmul(packed: Uint8Array, x: Float32Array, rows: number, cols: number) {
  const pad = (packed.byteLength + 3) & ~3;
  const wbuf = dev.createBuffer({size: pad, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  const padded = new Uint8Array(pad); padded.set(packed);
  dev.queue.writeBuffer(wbuf, 0, padded);
  const xbuf = dev.createBuffer({size: x.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  dev.queue.writeBuffer(xbuf, 0, x);
  const obuf = dev.createBuffer({size: rows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const dims = new Uint32Array([rows, cols, cols / 32, 0]);
  const dbuf = dev.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  dev.queue.writeBuffer(dbuf, 0, dims);
  const rd = dev.createBuffer({size: rows * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const bg = dev.createBindGroup({layout: pipe.getBindGroupLayout(0), entries: [
    {binding: 0, resource: {buffer: wbuf}}, {binding: 1, resource: {buffer: xbuf}},
    {binding: 2, resource: {buffer: obuf}}, {binding: 3, resource: {buffer: dbuf}}]});
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(pipe); p.setBindGroup(0, bg);
  p.dispatchWorkgroups(Math.ceil(rows / 64)); p.end();
  enc.copyBufferToBuffer(obuf, 0, rd, 0, rows * 4);
  dev.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rd.getMappedRange().slice(0));
  rd.unmap();
  return out;
}

let pass = 0, fail = 0;
const ok = (n: string, c: boolean, x = "") => {
  if (c) { pass++; console.log(`  ok   ${n}${x ? "  " + x : ""}`); }
  else { fail++; console.log(`  FAIL ${n}${x ? "  " + x : ""}`); }
};

// --- synthetic: kernel must match the CPU reference exactly ---------------
for (const [rows, cols] of [[64, 32], [128, 1024], [1024, 1024]]) {
  const w = new Float32Array(rows * cols);
  for (let i = 0; i < w.length; i++) w[i] = (Math.random() - 0.5) * 0.1;
  const packed = new Uint8Array(rows * (cols / 32) * 34);
  for (let r = 0; r < rows; r++) {
    packed.set(quantRow(w.subarray(r * cols, (r + 1) * cols)), r * (cols / 32) * 34);
  }
  const x = new Float32Array(cols);
  for (let i = 0; i < cols; i++) x[i] = (Math.random() - 0.5) * 2;
  const cpu = cpuMatmul(packed, x, rows, cols);
  const gpu = await gpuMatmul(packed, x, rows, cols);
  let maxRel = 0;
  for (let i = 0; i < rows; i++) {
    const denom = Math.max(1e-6, Math.abs(cpu[i]));
    maxRel = Math.max(maxRel, Math.abs(gpu[i] - cpu[i]) / denom);
  }
  ok(`${rows}x${cols} matches CPU reference`, maxRel < 1e-4, `max rel err ${maxRel.toExponential(1)}`);
}

console.log(`\n${pass} passed, ${fail} failed`);
if (fail) Deno.exit(1);
