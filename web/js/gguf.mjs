// flock/gguf — read a GGUF model's directory over HTTP range requests.
//
// A GGUF file is a directory followed by a blob: the first few MB list every
// tensor with its exact byte offset, and everything after is raw weights. That
// layout means a device can fetch ONLY the layers it owns:
//
//   1. range-fetch the directory        (~6 MB for Qwen3-0.6B)
//   2. look up its own blk.N.* tensors
//   3. range-fetch just those byte spans
//
// For a bird holding 4 of 28 layers that is ~60 MB instead of the 252 MB ONNX
// export -- and the host needs no build step and no disk at all, because peers
// pull from the model host directly. It also makes the split DYNAMIC: layer
// ranges are just byte ranges, so the coordinator can decide them when devices
// join rather than baking them into files.
//
// Shared by Node and browsers, like wire.mjs.

const MAGIC = 0x46554747;            // "GGUF" little-endian

// GGUF metadata value types. 8 = string, 9 = array, rest are fixed-width.
const VSIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8};

// Tensor element types we can describe. block_size/type_size follow ggml:
// a quantized "block" is N weights sharing one or two scale factors.
export const GGML = {
  0:  {name: 'F32',  block: 1,  bytes: 4},
  1:  {name: 'F16',  block: 1,  bytes: 2},
  2:  {name: 'Q4_0', block: 32, bytes: 18},
  3:  {name: 'Q4_1', block: 32, bytes: 20},
  6:  {name: 'Q5_0', block: 32, bytes: 22},
  7:  {name: 'Q5_1', block: 32, bytes: 24},
  8:  {name: 'Q8_0', block: 32, bytes: 34},
  9:  {name: 'Q8_1', block: 32, bytes: 40},
  10: {name: 'Q2_K', block: 256, bytes: 84},
  11: {name: 'Q3_K', block: 256, bytes: 110},
  12: {name: 'Q4_K', block: 256, bytes: 144},
  13: {name: 'Q5_K', block: 256, bytes: 176},
  14: {name: 'Q6_K', block: 256, bytes: 210},
  15: {name: 'Q8_K', block: 256, bytes: 292},
};

/** Bytes a tensor of `dims` occupies in `type`. */
export function tensorBytes(dims, type) {
  const spec = GGML[type];
  if (!spec) throw new Error(`unknown ggml type ${type}`);
  const n = dims.reduce((a, b) => a * b, 1);
  if (n % spec.block !== 0) {
    throw new Error(`${n} elements not divisible by block ${spec.block}`);
  }
  return (n / spec.block) * spec.bytes;
}

class Reader {
  constructor(buf) {
    this.dv = new DataView(buf);
    this.u8 = new Uint8Array(buf);
    this.o = 0;
  }
  need(n) {
    if (this.o + n > this.u8.length) {
      const e = new Error('gguf: directory extends past the fetched range');
      e.short = true; e.need = this.o + n;
      throw e;
    }
  }
  u32() { this.need(4); const v = this.dv.getUint32(this.o, true); this.o += 4; return v; }
  u64() {
    this.need(8);
    const v = this.dv.getBigUint64(this.o, true); this.o += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('gguf: value too large');
    return Number(v);
  }
  str() {
    const n = this.u64(); this.need(n);
    const s = new TextDecoder().decode(this.u8.subarray(this.o, this.o + n));
    this.o += n; return s;
  }
  skipValue(t) {
    if (t === 8) { const n = this.u64(); this.need(n); this.o += n; return; }
    if (t === 9) {
      const et = this.u32(), n = this.u64();
      if (et === 8) {                       // array of strings: variable stride
        for (let i = 0; i < n; i++) { const len = this.u64(); this.need(len); this.o += len; }
      } else {
        const w = VSIZE[et];
        if (w === undefined) throw new Error(`gguf: bad array type ${et}`);
        this.need(w * n); this.o += w * n;  // fixed stride: one jump
      }
      return;
    }
    const w = VSIZE[t];
    if (w === undefined) throw new Error(`gguf: bad value type ${t}`);
    this.need(w); this.o += w;
  }
  /** Read a value we actually care about (scalars and strings only). */
  readValue(t) {
    if (t === 8) return this.str();
    if (t === 9) { this.skipValue(9); return null; }   // arrays: vocab etc, skip
    const at = this.o;
    switch (t) {
      case 0: this.o += 1; return this.dv.getUint8(at);
      case 1: this.o += 1; return this.dv.getInt8(at);
      case 2: this.o += 2; return this.dv.getUint16(at, true);
      case 3: this.o += 2; return this.dv.getInt16(at, true);
      case 4: this.o += 4; return this.dv.getUint32(at, true);
      case 5: this.o += 4; return this.dv.getInt32(at, true);
      case 6: this.o += 4; return this.dv.getFloat32(at, true);
      case 7: this.o += 1; return !!this.dv.getUint8(at);
      case 10: return this.u64();
      case 11: this.o += 8; return Number(this.dv.getBigInt64(at, true));
      case 12: this.o += 8; return this.dv.getFloat64(at, true);
      default: throw new Error(`gguf: bad value type ${t}`);
    }
  }
}

/**
 * Parse a GGUF directory from the head of the file.
 * Throws an error with `.short = true` if `buf` doesn't reach far enough --
 * the caller should refetch a larger prefix and retry.
 */
export function parseDirectory(buf) {
  const r = new Reader(buf);
  if (r.u32() !== MAGIC) throw new Error('not a GGUF file');
  const version = r.u32();
  if (version < 2 || version > 3) throw new Error(`unsupported GGUF version ${version}`);
  const nTensors = r.u64(), nKV = r.u64();

  const meta = {};
  for (let i = 0; i < nKV; i++) {
    const key = r.str();
    const t = r.u32();
    const v = r.readValue(t);
    if (v !== null) meta[key] = v;          // arrays (vocab) are skipped
  }

  const tensors = new Map();
  for (let i = 0; i < nTensors; i++) {
    const name = r.str();
    const nd = r.u32();
    const dims = [];
    for (let j = 0; j < nd; j++) dims.push(r.u64());
    const type = r.u32();
    const offset = r.u64();                 // relative to dataStart
    tensors.set(name, {name, dims, type, offset, bytes: tensorBytes(dims, type)});
  }

  // Tensor data begins after the directory, aligned up.
  const align = meta['general.alignment'] ?? 32;
  const dataStart = Math.ceil(r.o / align) * align;
  return {version, meta, tensors, dataStart, directoryBytes: r.o};
}

/** Every tensor a given layer owns, e.g. layerTensors(dir, 24) -> blk.24.*  */
export function layerTensors(dir, layer) {
  const prefix = `blk.${layer}.`;
  return [...dir.tensors.values()]
    .filter(t => t.name.startsWith(prefix))
    .sort((a, b) => a.offset - b.offset);
}

/**
 * Byte ranges (absolute, inclusive-exclusive) covering `layers`, merged where
 * they are contiguous so a device issues as few range requests as possible.
 */
export function byteRanges(dir, layers, {gap = 1 << 20} = {}) {
  const ts = layers.flatMap(l => layerTensors(dir, l)).sort((a, b) => a.offset - b.offset);
  const out = [];
  for (const t of ts) {
    const start = dir.dataStart + t.offset, end = start + t.bytes;
    const last = out[out.length - 1];
    // Merge when the hole is small enough that one request beats two.
    if (last && start - last.end <= gap) last.end = Math.max(last.end, end);
    else out.push({start, end});
  }
  return out;
}

export function totalBytes(ranges) {
  return ranges.reduce((a, r) => a + (r.end - r.start), 0);
}

/**
 * Fetch the directory, growing the prefix until it is complete.
 * `fetchRange(start, end)` must resolve to an ArrayBuffer.
 */
export async function loadDirectory(fetchRange, {start = 1 << 20, max = 64 << 20} = {}) {
  let size = start;
  for (;;) {
    const buf = await fetchRange(0, size - 1);
    try {
      return parseDirectory(buf);
    } catch (e) {
      if (!e.short || size >= max) throw e;
      // Ask for what the parser said it needed, with headroom.
      size = Math.min(max, Math.max(size * 2, (e.need || size) * 1.25 | 0));
    }
  }
}

/** fetchRange backed by HTTP Range requests. */
export function httpRange(url) {
  return async (start, end) => {
    const r = await fetch(url, {headers: {Range: `bytes=${start}-${end}`}});
    if (!r.ok && r.status !== 206) throw new Error(`range fetch failed: ${r.status}`);
    return r.arrayBuffer();
  };
}
