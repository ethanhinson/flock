// flock/gguf-dir — read a GGUF file's directory in a browser, over range requests.
//
// WHY this exists at all, when server/gguf.mjs already reads directories:
// that module imports @huggingface/gguf, a bare npm specifier, so it cannot be
// served to a phone. The alternative was to have the coordinator parse the
// header and hand birds the directory as JSON — which is less code here but
// worse in the way that matters: it makes the coordinator a required middleman
// for a fetch that otherwise goes bird -> HuggingFace directly. A bird could
// then only load models the coordinator had already read, and the whole point
// of the GGUF path is that a device needs nothing but a URL and a layer range.
// So the header parse lives here, and the coordinator stays out of the loop.
//
// The format, from the ggml spec:
//   magic u32 "GGUF" | version u32 | tensor_count u64 | kv_count u64
//   kv_count entries of (string key, u32 type, value)
//   tensor_count entries of (string name, u32 n_dims, u64 dims[n_dims],
//                            u32 dtype, u64 offset)
//   padding to general.alignment, then the raw tensor data
//
// Tensor `offset` is relative to that data start, NOT to the file. Absolute =
// dataStart + offset. Treating it as absolute fetches the wrong bytes, and
// those bytes decode as valid-looking Q8_0 and produce plausible garbage rather
// than an error, so there is nothing to notice until the model talks nonsense.

const MAGIC = 0x46554747;              // "GGUF" little-endian

// GGUF metadata value types, in spec order.
const T = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11,
  FLOAT64: 12,
};

// block_size / type_size per ggml: a quantized "block" is N weights sharing one
// or two scale factors, so bytes-per-tensor is not elements * width. Kept in
// sync with the same table in server/gguf.mjs -- two copies because one is
// Node-only and one is browser-only, and neither can import the other.
export const BLOCK = {
  0:  {block: 1,   bytes: 4},    // F32
  1:  {block: 1,   bytes: 2},    // F16
  2:  {block: 32,  bytes: 18},   // Q4_0
  3:  {block: 32,  bytes: 20},   // Q4_1
  6:  {block: 32,  bytes: 22},   // Q5_0
  7:  {block: 32,  bytes: 24},   // Q5_1
  8:  {block: 32,  bytes: 34},   // Q8_0
  9:  {block: 32,  bytes: 40},   // Q8_1
  10: {block: 256, bytes: 84},   // Q2_K
  11: {block: 256, bytes: 110},  // Q3_K
  12: {block: 256, bytes: 144},  // Q4_K
  13: {block: 256, bytes: 176},  // Q5_K
  14: {block: 256, bytes: 210},  // Q6_K
  15: {block: 256, bytes: 292},  // Q8_K
};

export const F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8;

export function tensorBytes(shape, dtype) {
  const spec = BLOCK[dtype];
  if (!spec) throw new Error(`unsupported ggml type ${dtype}`);
  const n = shape.reduce((a, b) => a * Number(b), 1);
  if (n % spec.block !== 0) {
    throw new Error(`${n} elements not divisible by block ${spec.block}`);
  }
  return (n / spec.block) * spec.bytes;
}

/**
 * A cursor over a growable byte window, which is the whole trick to parsing a
 * header you have not finished downloading.
 *
 * The header's length is not stated anywhere in it: you only know where it ends
 * once you have walked every metadata entry and every tensor record. Qwen3-0.6B
 * spends ~6MB of that on the tokenizer's 151936 token strings. So the reader
 * starts with a small window and calls `grow` whenever a read would run past the
 * end, which turns "parse a 6MB header" into a handful of range requests instead
 * of either one 6MB fetch-before-you-know or a request per field.
 */
class Cursor {
  constructor(bytes, grow) {
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.at = 0;
    this.grow = grow;                  // async (needBytes) => Uint8Array
  }

  async need(n) {
    if (this.at + n <= this.bytes.byteLength) return;
    this.bytes = await this.grow(this.at + n);
    this.view = new DataView(this.bytes.buffer, this.bytes.byteOffset, this.bytes.byteLength);
    if (this.at + n > this.bytes.byteLength) {
      throw new Error(`GGUF header truncated at ${this.at}+${n}`);
    }
  }

  async u8()  { await this.need(1); return this.view.getUint8(this.at++); }
  async i8()  { await this.need(1); return this.view.getInt8(this.at++); }
  async u16() { await this.need(2); const v = this.view.getUint16(this.at, true); this.at += 2; return v; }
  async i16() { await this.need(2); const v = this.view.getInt16(this.at, true); this.at += 2; return v; }
  async u32() { await this.need(4); const v = this.view.getUint32(this.at, true); this.at += 4; return v; }
  async i32() { await this.need(4); const v = this.view.getInt32(this.at, true); this.at += 4; return v; }
  async f32() { await this.need(4); const v = this.view.getFloat32(this.at, true); this.at += 4; return v; }
  async f64() { await this.need(8); const v = this.view.getFloat64(this.at, true); this.at += 8; return v; }
  async bool(){ return (await this.u8()) !== 0; }

  // Counts and offsets are u64. Number() is safe here and BigInt is not worth
  // the friction: a GGUF field would have to exceed 2^53 bytes to lose
  // precision, which is four orders of magnitude past any real model.
  async u64() {
    await this.need(8);
    const v = this.view.getBigUint64(this.at, true); this.at += 8;
    return Number(v);
  }
  async i64() {
    await this.need(8);
    const v = this.view.getBigInt64(this.at, true); this.at += 8;
    return Number(v);
  }

  async str() {
    const n = await this.u64();
    await this.need(n);
    const s = new TextDecoder().decode(this.bytes.subarray(this.at, this.at + n));
    this.at += n;
    return s;
  }
}

async function readValue(c, type) {
  switch (type) {
    case T.UINT8:   return c.u8();
    case T.INT8:    return c.i8();
    case T.UINT16:  return c.u16();
    case T.INT16:   return c.i16();
    case T.UINT32:  return c.u32();
    case T.INT32:   return c.i32();
    case T.FLOAT32: return c.f32();
    case T.BOOL:    return c.bool();
    case T.STRING:  return c.str();
    case T.UINT64:  return c.u64();
    case T.INT64:   return c.i64();
    case T.FLOAT64: return c.f64();
    case T.ARRAY: {
      const et = await c.u32(), n = await c.u64();
      // The tokenizer arrays are 151936 entries and no bird needs them, but they
      // still have to be WALKED to find where the tensor records begin. Strings
      // are the expensive case, so they are skipped without being decoded.
      if (et === T.STRING) {
        for (let i = 0; i < n; i++) {
          const len = await c.u64();
          await c.need(len);
          c.at += len;
        }
        return {skipped: n, type: 'string[]'};
      }
      const out = [];
      for (let i = 0; i < n; i++) out.push(await readValue(c, et));
      return out;
    }
    default: throw new Error(`unknown GGUF metadata type ${type}`);
  }
}

/** Fetch bytes [start, end) of a URL. One HTTP range request. */
export async function fetchRange(url, {start, end}, init = {}) {
  const r = await fetch(url, {...init, headers: {...init.headers, Range: `bytes=${start}-${end - 1}`}});
  if (!r.ok && r.status !== 206) throw new Error(`range fetch failed: ${r.status}`);
  // A server that ignores Range answers 200 with the WHOLE file, which on a
  // 639MB model is an instant OOM rather than a wrong answer. Refuse it.
  if (r.status === 200 && end - start < 1 << 24) {
    const len = +r.headers.get('content-length') || 0;
    if (len > (end - start) * 2) {
      throw new Error(`server ignored Range (sent ${len} bytes for ${end - start})`);
    }
  }
  return new Uint8Array(await r.arrayBuffer());
}

/**
 * Read a GGUF model's directory: metadata, every tensor's dtype/shape/offset,
 * and where the tensor data begins. Only the header is fetched.
 *
 * `chunk` is how much is pulled per growth step. 1MB is not an arbitrary round
 * number: it is a compromise between request count (Qwen3-0.6B's header is
 * ~6MB, so ~7 requests) and the wasted tail on a model with a small header.
 */
export async function readDirectory(url, {chunk = 1 << 20, fetchInit = {}} = {}) {
  let have = new Uint8Array(0), fetched = 0, requests = 0;
  const grow = async (need) => {
    const want = Math.max(need, fetched + chunk);
    const next = await fetchRange(url, {start: fetched, end: want}, fetchInit);
    requests++;
    if (next.byteLength === 0) throw new Error('GGUF header: server returned no bytes');
    const merged = new Uint8Array(fetched + next.byteLength);
    merged.set(have); merged.set(next, fetched);
    have = merged; fetched = have.byteLength;
    return have;
  };

  const c = new Cursor(have, grow);
  if (await c.u32() !== MAGIC) throw new Error(`not a GGUF file: ${url}`);
  const version = await c.u32();
  if (version !== 2 && version !== 3) throw new Error(`unsupported GGUF version ${version}`);
  const tensorCount = await c.u64();
  const kvCount = await c.u64();

  const metadata = {};
  for (let i = 0; i < kvCount; i++) {
    const key = await c.str();
    const type = await c.u32();
    metadata[key] = await readValue(c, type);
  }

  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = await c.str();
    const nDims = await c.u32();
    const shape = [];
    for (let d = 0; d < nDims; d++) shape.push(await c.u64());
    const dtype = await c.u32();
    const offset = await c.u64();
    tensors.push({name, shape, dtype, offset, bytes: tensorBytes(shape, dtype)});
  }

  // Tensor data starts at the next multiple of general.alignment after the
  // header. The default of 32 is the spec's, not a guess.
  const align = Number(metadata['general.alignment'] ?? 32);
  const dataStart = Math.ceil(c.at / align) * align;

  const arch = metadata['general.architecture'];
  return {
    url, arch, metadata, tensors, dataStart, version,
    nLayers: Number(metadata[`${arch}.block_count`]),
    // What the parse itself cost, so a caller can report it honestly instead of
    // claiming "only the header" without a number behind it.
    headerBytes: c.at, headerFetched: fetched, headerRequests: requests,
  };
}

/** Every tensor a given layer owns, e.g. layerTensors(m, 24) -> blk.24.*  */
export function layerTensors(model, layer) {
  const prefix = `blk.${layer}.`;
  return model.tensors
    .filter(t => t.name.startsWith(prefix))
    .sort((a, b) => a.offset - b.offset);
}

/**
 * Absolute byte ranges covering `layers`, merged where one request beats two.
 * Layers are laid out sequentially in GGUF, so a contiguous slice usually
 * collapses to a SINGLE range request.
 *
 * `gap` is how much dead space is worth swallowing to avoid a second request.
 * At LAN-to-WAN latencies a request costs more than 1MB of transfer, and in
 * practice layer tensors are back-to-back so nothing is swallowed at all.
 */
export function byteRanges(model, tensors, {gap = 1 << 20} = {}) {
  const ts = [...tensors].sort((a, b) => a.offset - b.offset);
  const out = [];
  for (const t of ts) {
    const start = model.dataStart + t.offset, end = start + t.bytes;
    const last = out[out.length - 1];
    if (last && start - last.end <= gap) last.end = Math.max(last.end, end);
    else out.push({start, end});
  }
  return out;
}

export function totalBytes(ranges) {
  return ranges.reduce((a, r) => a + (r.end - r.start), 0);
}
