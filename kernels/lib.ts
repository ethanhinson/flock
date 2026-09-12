// Shared plumbing for the kernel tests and benchmarks: quantization, CPU
// references, GGUF repacking, and the WebGPU boilerplate.
//
// The single most important thing in this file is `cpuMatmulQ8F32`. JS numbers
// are f64, so the obvious CPU reference accumulates in double precision and
// then disagrees with the GPU by ~1e-4 at 1024 columns -- not because either is
// wrong, but because f32 summation of 1024 terms loses that much on its own.
// Comparing an f32 kernel against an f64 reference makes a correct kernel look
// broken.
//
// Rounding after every operation is necessary but not sufficient. A naive
// strict-f32 reference (fround(acc + fround(q*x))) is still WORSE than the f64
// one, because the GPU contracts every multiply-add into a single FMA -- one
// rounding, not two. Model that and the agreement goes from "1e-4, hopefully
// fine" to bit-for-bit identical at every size tested. `fma32` below is how,
// and every reference here is built on it.
//
// The FMA emulation is exact rather than approximate: for f32 inputs the f64
// product a*b is exact (24+24 <= 53 mantissa bits), so fround(a*b + c) in f64
// performs exactly one rounding of the true value -- which is the definition of
// a f32 FMA.

import { f32to16 } from "../web/js/wire.mjs";

export const Q8_BLOCK = 32;
export const Q8_BYTES = 34;   // f16 scale + 32 int8
export const Q4_BLOCK = 32;
export const Q4_BYTES = 18;   // f16 scale + 16 packed nibble pairs

export function halfToF32(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * Math.pow(2, -24);
  if (e === 31) return s * 3.4028235e38;   // kernels clamp inf to max finite
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}

// ---------------------------------------------------------------- quantizing

/** Quantize one row of f32 into Q8_0 blocks, exactly as llama.cpp does. */
export function quantRowQ8(vals: Float32Array): Uint8Array {
  const nb = vals.length / Q8_BLOCK;
  const out = new Uint8Array(nb * Q8_BYTES);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let i = 0; i < Q8_BLOCK; i++) amax = Math.max(amax, Math.abs(vals[b * 32 + i]));
    const d = amax / 127;
    dv.setUint16(b * Q8_BYTES, f32to16(d), true);
    for (let i = 0; i < Q8_BLOCK; i++) {
      const q = d === 0 ? 0 : Math.round(vals[b * 32 + i] / d);
      out[b * Q8_BYTES + 2 + i] = Math.max(-128, Math.min(127, q)) & 0xff;
    }
  }
  return out;
}

/**
 * Quantize one row into Q4_0. The layout is the ggml one and it is NOT the
 * obvious one: byte i holds weight i in the low nibble and weight i+16 in the
 * high nibble, so a single byte spans two halves of the block. Getting this
 * wrong produces plausible-looking garbage, which is why the reference decoder
 * below mirrors it literally rather than being "simplified".
 */
export function quantRowQ4(vals: Float32Array): Uint8Array {
  const nb = vals.length / Q4_BLOCK;
  const out = new Uint8Array(nb * Q4_BYTES);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nb; b++) {
    // Q4_0 is symmetric around 8 with a signed scale: pick the extreme of
    // largest magnitude (not the largest absolute value) so the sign survives.
    let amax = 0, max = 0;
    for (let i = 0; i < Q4_BLOCK; i++) {
      const v = vals[b * 32 + i];
      if (Math.abs(v) > amax) { amax = Math.abs(v); max = v; }
    }
    const d = max / -8;
    const id = d === 0 ? 0 : 1 / d;
    dv.setUint16(b * Q4_BYTES, f32to16(d), true);
    for (let i = 0; i < 16; i++) {
      const x0 = vals[b * 32 + i] * id + 8.5;
      const x1 = vals[b * 32 + i + 16] * id + 8.5;
      const q0 = Math.min(15, Math.max(0, Math.floor(x0)));
      const q1 = Math.min(15, Math.max(0, Math.floor(x1)));
      out[b * Q4_BYTES + 2 + i] = q0 | (q1 << 4);
    }
  }
  return out;
}

/** Pack a whole rows x cols f32 matrix into the on-disk Q8_0 layout. */
export function quantMatrixQ8(w: Float32Array, rows: number, cols: number): Uint8Array {
  const rowBytes = (cols / Q8_BLOCK) * Q8_BYTES;
  const out = new Uint8Array(rows * rowBytes);
  for (let r = 0; r < rows; r++) {
    out.set(quantRowQ8(w.subarray(r * cols, (r + 1) * cols)), r * rowBytes);
  }
  return out;
}

export function quantMatrixQ4(w: Float32Array, rows: number, cols: number): Uint8Array {
  const rowBytes = (cols / Q4_BLOCK) * Q4_BYTES;
  const out = new Uint8Array(rows * rowBytes);
  for (let r = 0; r < rows; r++) {
    out.set(quantRowQ4(w.subarray(r * cols, (r + 1) * cols)), r * rowBytes);
  }
  return out;
}

// ---------------------------------------------------------------- repacking
//
// The file layout interleaves a 2-byte scale with 32 quants every 34 bytes, so
// nothing is u32-aligned and a kernel has to extract bytes with shift/mask. The
// split layout below is what makes vectorised decoding possible: `qs` is a
// contiguous i8 array (8 quants per u32 word, load-and-unpack4xI8) and `scales`
// is f16 packed two-per-word (unpack2x16float). Repacking is a one-time upload
// cost and it is what buys the kernel its speed.

export interface SplitQ8 { qs: Uint8Array; scales: Uint8Array }

export function splitQ8(packed: Uint8Array, rows: number, cols: number): SplitQ8 {
  const nb = cols / Q8_BLOCK, rowBytes = nb * Q8_BYTES;
  const qs = new Uint8Array(rows * cols);
  const scales = new Uint8Array(rows * nb * 2);
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < nb; b++) {
      const src = r * rowBytes + b * Q8_BYTES;
      const si = (r * nb + b) * 2;
      scales[si] = packed[src];
      scales[si + 1] = packed[src + 1];
      qs.set(packed.subarray(src + 2, src + 2 + 32), r * cols + b * 32);
    }
  }
  return { qs, scales };
}

export function splitQ4(packed: Uint8Array, rows: number, cols: number): SplitQ8 {
  const nb = cols / Q4_BLOCK, rowBytes = nb * Q4_BYTES;
  const qs = new Uint8Array(rows * nb * 16);
  const scales = new Uint8Array(rows * nb * 2);
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < nb; b++) {
      const src = r * rowBytes + b * Q4_BYTES;
      const si = (r * nb + b) * 2;
      scales[si] = packed[src];
      scales[si + 1] = packed[src + 1];
      qs.set(packed.subarray(src + 2, src + 18), (r * nb + b) * 16);
    }
  }
  return { qs, scales };
}

// ------------------------------------------------------------ CPU references

const fr = Math.fround;

/**
 * One f32 fused multiply-add: round(a*b + c) with a SINGLE rounding.
 *
 * Exact, not approximate -- see the FMA note at the top of this file. This is
 * the primitive that makes the CPU references match the GPU bit-for-bit rather
 * than merely within tolerance.
 */
export function fma32(a: number, b: number, c: number): number {
  return fr(a * b + c);
}

/**
 * How a kernel adds its terms up. f32 addition is not associative, so a
 * reference that sums in a different order than the kernel disagrees with it by
 * more than either is wrong by -- at cols=3072 that disagreement reaches 5e-4,
 * which is well into "looks like a real bug" territory.
 *
 *  "serial"  one thread per row, one accumulator, 32 FMAs per block, then one
 *            more FMA to fold in the scale. This is q8_matmul.wgsl.
 *  "coop"    `lanes` threads per row; thread t owns blocks t, t+lanes, ...; each
 *            block is two dot(vec4) calls, summed, scaled, and FMA'd into that
 *            thread's accumulator. The partials are then tree-reduced pairwise
 *            in workgroup memory. This is q8_coop.wgsl / q4_coop.wgsl.
 */
export type Reduction = { kind: "serial" } | { kind: "coop"; lanes: number };

export const SERIAL: Reduction = { kind: "serial" };
export const coop = (lanes: number): Reduction => ({ kind: "coop", lanes });

/**
 * dot(vec4, vec4) as the Metal backend emits it: one exact product, then three
 * FMAs. Verified bit-for-bit against the GPU -- a version that rounds each
 * product separately does NOT match.
 */
function dot4(a: number[], b: number[]): number {
  let s = fr(a[0] * b[0]);
  s = fma32(a[1], b[1], s);
  s = fma32(a[2], b[2], s);
  s = fma32(a[3], b[3], s);
  return s;
}

/**
 * The shared reference driver. `decode(base)` yields a block's scale and its 32
 * dequantized-but-unscaled integer weights; only the summation order differs
 * between kernel shapes, not the decode, so Q8_0 and Q4_0 share this.
 *
 * `pair(g)` gives the two 4-element column groups that word-pair g of a block
 * covers. Q8_0 words are contiguous (0..3 and 4..7); Q4_0 packs two nibbles per
 * byte so one word spans cols g*4..g*4+3 and g*4+16..g*4+19.
 */
function matmulF32(
  rows: number, cols: number, blockBytes: number, red: Reduction,
  decode: (base: number) => { scale: number; q: number[] },
  x: Float32Array,
  rowBase: (r: number) => number,
  pair: (g: number) => [number[], number[]],
): Float32Array {
  const nb = cols / 32, out = new Float32Array(rows);
  const lanes = red.kind === "coop" ? red.lanes : 1;
  const part = new Float32Array(lanes);
  const pairs = [0, 1, 2, 3].map(pair);
  for (let r = 0; r < rows; r++) {
    part.fill(0);
    for (let b = 0; b < nb; b++) {
      const { scale, q } = decode(rowBase(r) + b * blockBytes);
      if (red.kind === "serial") {
        let s = 0;
        for (let i = 0; i < 32; i++) s = fma32(q[i], x[b * 32 + i], s);
        part[0] = fma32(scale, s, part[0]);
      } else {
        // `sc * (dot(d0, xa) + dot(d1, xb))` folded into the accumulator, once
        // per word pair -- four times per 32-weight block.
        const lane = b % lanes;
        for (const [ia, ib] of pairs) {
          const d0 = dot4(ia.map((i) => q[i]), ia.map((i) => x[b * 32 + i]));
          const d1 = dot4(ib.map((i) => q[i]), ib.map((i) => x[b * 32 + i]));
          part[lane] = fma32(scale, fr(d0 + d1), part[lane]);
        }
      }
    }
    out[r] = red.kind === "serial" ? part[0] : treeSum(part);
  }
  return out;
}

export function cpuMatmulQ8F32(
  packed: Uint8Array, x: Float32Array, rows: number, cols: number, red: Reduction = SERIAL,
): Float32Array {
  const nb = cols / Q8_BLOCK;
  const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  return matmulF32(rows, cols, Q8_BYTES, red, (base) => {
    const q = new Array(32);
    for (let i = 0; i < 32; i++) {
      let v = packed[base + 2 + i]; if (v > 127) v -= 256;
      q[i] = v;
    }
    return { scale: fr(halfToF32(dv.getUint16(base, true))), q };
  }, x, (r) => r * nb * Q8_BYTES,
    (g) => [[0, 1, 2, 3].map((k) => g * 8 + k), [0, 1, 2, 3].map((k) => g * 8 + 4 + k)]);
}

export function cpuMatmulQ4F32(
  packed: Uint8Array, x: Float32Array, rows: number, cols: number, red: Reduction = SERIAL,
): Float32Array {
  const nb = cols / Q4_BLOCK;
  const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  return matmulF32(rows, cols, Q4_BYTES, red, (base) => {
    // Byte i holds weight i in its low nibble and weight i+16 in its high one,
    // so the 16 bytes cover the block's two halves in parallel, not in order.
    const q = new Array(32);
    for (let i = 0; i < 16; i++) {
      const byte = packed[base + 2 + i];
      q[i] = (byte & 0xf) - 8;
      q[i + 16] = (byte >> 4) - 8;
    }
    return { scale: fr(halfToF32(dv.getUint16(base, true))), q };
  }, x, (r) => r * nb * Q4_BYTES,
    (g) => [[0, 1, 2, 3].map((k) => g * 4 + k), [0, 1, 2, 3].map((k) => g * 4 + 16 + k)]);
}

/** Pairwise f32 tree sum, the same order a workgroup reduction produces. */
function treeSum(a: Float32Array): number {
  const buf = Float32Array.from(a);
  for (let stride = buf.length >> 1; stride > 0; stride >>= 1) {
    for (let i = 0; i < stride; i++) buf[i] = fr(buf[i] + buf[i + stride]);
  }
  return buf[0];
}

// ---------------------------------------------------------------- comparison

export function maxRelErr(a: Float32Array, b: Float32Array): number {
  let worst = 0;
  // A fixed epsilon floor keeps near-zero outputs from reporting a huge
  // relative error over pure rounding noise.
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(1e-6, Math.abs(b[i]));
    worst = Math.max(worst, Math.abs(a[i] - b[i]) / denom);
  }
  return worst;
}

// -------------------------------------------------------------------- WebGPU

/**
 * A device with the storage-buffer limits RAISED to what the adapter actually
 * supports, not the spec defaults.
 *
 * This is required, not an optimization. `maxStorageBufferBindingSize` defaults to
 * 128 MiB, and Qwen3-0.6B's `token_embd.weight` -- which is both the embedding
 * table and, tied, the LM head -- is 151936 x 1024 int8 = 155.6 MB once repacked
 * into the split layout the kernels read. Binding it on a default device fails
 * with "range 155582464 exceeds max_*_buffer_binding_size limit 134217728", and
 * because a failed bind group is a validation error rather than an exception at
 * the call site, the symptom is a kernel that silently writes zeros. Measured on
 * this machine the adapter allows 4 GiB bindings and 22 GB buffers, so asking for
 * the maximum costs nothing.
 *
 * A device is requested with raised limits first and falls back to the defaults if
 * that is refused, because a browser or a smaller GPU may cap them lower -- and
 * every kernel except the two that touch token_embd fits inside 128 MiB anyway.
 * Note the Deno quirk this has to work around: a failed `requestDevice` still
 * INVALIDATES the adapter ("The adapter cannot be reused, as it has been
 * invalidated by a device creation"), so the fallback has to request a fresh one.
 */
export async function getDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  let dev: GPUDevice;
  try {
    dev = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
  } catch {
    const again = await navigator.gpu.requestAdapter();
    if (!again) throw new Error("no WebGPU adapter");
    dev = await again.requestDevice();
  }
  dev.addEventListener?.("uncapturederror", (e) => {
    console.error("uncaptured GPU error:", (e as GPUUncapturedErrorEvent).error);
  });
  return dev;
}

/**
 * Does this device's WGSL support a given builtin? Compiling a probe shader is
 * the only reliable answer -- these are language builtins, not device features,
 * so nothing is advertised in `device.features`, and they were made mandatory
 * only after shipping implementations existed.
 *
 * Deno 2.1's wgpu backend has no `getCompilationInfo`, so the error scope alone
 * has to carry the verdict. It does: a missing identifier is a validation error.
 */
export async function probeWGSL(dev: GPUDevice, body: string): Promise<boolean> {
  dev.pushErrorScope("validation");
  const m = dev.createShaderModule({
    code: `@compute @workgroup_size(1) fn p() { ${body} }`,
  });
  let msgs = false;
  if (typeof m.getCompilationInfo === "function") {
    const info = await m.getCompilationInfo();
    msgs = info.messages.some((x) => x.type === "error");
  }
  const err = await dev.popErrorScope();
  return !err && !msgs;
}

/**
 * The packed-byte path needs BOTH unpack4xI8 (8 int8 -> 2 vec4<f32> in two
 * instructions) and unpack2x16float (f16 decode in one). They are probed
 * together because the kernel uses them together.
 *
 * Measured support, for the record:
 *   Deno 2.1.2 / wgpu / Metal   unpack2x16float yes, unpack4xI8 NO
 *   Chrome / Safari 26          both yes (per swarmllm, not verified here)
 *
 * So on Deno the fallback shift/mask path is what actually runs, and any claim
 * about what the builtins buy has to come from a browser, not from here.
 */
export function probeUnpack(dev: GPUDevice): Promise<boolean> {
  return probeWGSL(dev, `let v = vec4<f32>(unpack4xU8(0x0F0F0F0Fu))
      + vec4<f32>(unpack4xI8(1u))
      + vec4<f32>(unpack2x16float(0u), 0.0, 0.0);`);
}

/** unpack2x16float alone -- available more widely than the 8-bit unpacks. */
export function probeUnpackF16(dev: GPUDevice): Promise<boolean> {
  return probeWGSL(dev, `let v = unpack2x16float(0u);`);
}

/**
 * Fill in the $-placeholders in a cooperative kernel's source.
 *
 * WGSL has no preprocessor and the 8-bit unpack builtins are not implemented
 * everywhere (Deno's wgpu has unpack2x16float but not unpack4xI8), so the two
 * decode spellings live here and the host picks one per device. They produce
 * bit-identical values -- the only difference is one instruction versus three
 * ALU ops per element -- which is what makes it safe to validate on the
 * fallback path and still ship the fast one.
 */
export function coopSource(src: string, opts: { unpack8: boolean; unpackF16: boolean }): string {
  const i8x4 = opts.unpack8
    // unpack4xI8 sign-extends and converts 4 bytes in one instruction.
    ? "vec4<f32>(unpack4xI8(w))"
    // Fallback: shift each byte into the top of a u32 and arithmetic-shift back
    // down. `extractBits` would read better but goes through slow polyfill
    // paths on some backends, so the shift pair is deliberate.
    : `vec4<f32>(
    f32(bitcast<i32>(w << 24u) >> 24u),
    f32(bitcast<i32>(w << 16u) >> 24u),
    f32(bitcast<i32>(w << 8u) >> 24u),
    f32(bitcast<i32>(w) >> 24u))`;
  // Q4_0 stores two weights per byte, biased by 8. `v` is the word already
  // masked down to one nibble per byte, so both spellings see the same input.
  const nib = opts.unpack8
    ? "vec4<f32>(unpack4xU8(v)) - vec4<f32>(8.0)"
    : `vec4<f32>(
    f32(v & 0xFFu),
    f32((v >> 8u) & 0xFFu),
    f32((v >> 16u) & 0xFFu),
    f32(v >> 24u)) - vec4<f32>(8.0)`;
  const f16 = opts.unpackF16
    ? "unpack2x16float(sc[i >> 1u])[i & 1u]"
    : "half_to_f32((sc[i >> 1u] >> ((i & 1u) * 16u)) & 0xFFFFu)";
  return src
    .replaceAll("$DECODE_I8X4", i8x4)
    .replaceAll("$DECODE_NIBX4", nib)
    .replaceAll("$DECODE_F16", f16)
    // The manual f16 decoder is only referenced on the fallback path; leaving a
    // dead copy in every shader would be noise, so it is spliced in on demand.
    .replace("$HALF_TO_F32", opts.unpackF16 ? "" : HALF_TO_F32_WGSL);
}

/**
 * Size attention.wgsl's workgroup scores array to the KV capacity actually
 * needed. See the note in that file: the array is a hard occupancy limit, and
 * over-declaring it made the kernel 12x more expensive than every other dispatch
 * in the layer at n_keys=1.
 */
export function attnSource(src: string, maxKeys: number): string {
  if (maxKeys < 1 || !Number.isInteger(maxKeys)) {
    throw new Error(`maxKeys must be a positive integer, got ${maxKeys}`);
  }
  return src.replaceAll("$MAX_KEYS", String(maxKeys));
}

/**
 * f16 -> f32 in f32 arithmetic, for backends without unpack2x16float. Kept as a
 * string rather than duplicated into each kernel so there is one copy to be
 * right about. inf clamps to the largest finite f32, matching lib.ts halfToF32.
 */
const HALF_TO_F32_WGSL = `
fn half_to_f32(h: u32) -> f32 {
  let sign = select(1.0, -1.0, (h & 0x8000u) != 0u);
  let exp  = (h >> 10u) & 0x1Fu;
  let man  = h & 0x3FFu;
  if (exp == 0u) { return sign * f32(man) * 0.000000059604645; }   // 2^-24
  if (exp == 31u) { return sign * 3.4028235e38; }
  return sign * (1.0 + f32(man) / 1024.0) * exp2(f32(exp) - 15.0);
}`;

export function storageBuffer(dev: GPUDevice, data: Uint8Array | Float32Array): GPUBuffer {
  const bytes = data instanceof Float32Array
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : data;
  // WebGPU requires a 4-byte-aligned size; the 34-byte Q8_0 layout is not.
  const size = (bytes.byteLength + 3) & ~3;
  // COPY_SRC as well as COPY_DST: some kernels (RoPE) work in place, so their
  // input buffer is also the buffer a test reads back.
  const buf = dev.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const padded = size === bytes.byteLength ? bytes : (() => {
    const p = new Uint8Array(size); p.set(bytes); return p;
  })();
  dev.queue.writeBuffer(buf, 0, padded);
  return buf;
}

export function uniformBuffer(dev: GPUDevice, vals: number[]): GPUBuffer {
  const size = Math.max(16, (vals.length * 4 + 15) & ~15);
  const buf = dev.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(buf, 0, new Uint32Array(vals));
  return buf;
}

export async function readBack(dev: GPUDevice, src: GPUBuffer, bytes: number): Promise<Float32Array> {
  const rd = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rd, 0, bytes);
  dev.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rd.getMappedRange().slice(0));
  rd.unmap(); rd.destroy();
  return out;
}

export function randVec(n: number, scale = 1): Float32Array {
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) v[i] = (Math.random() - 0.5) * 2 * scale;
  return v;
}

// ----------------------------------------------------------------- reporting

let pass = 0, fail = 0;
export function ok(name: string, cond: boolean, extra = "") {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? "  " + extra : ""}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? "  " + extra : ""}`); }
}
export function summary(): number {
  console.log(`\n${pass} passed, ${fail} failed`);
  return fail;
}
