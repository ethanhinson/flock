// flock/gguf-stream — stream GGUF tensors from HuggingFace straight into GPU buffers.
//
// THE PROBLEM THIS SOLVES. A bird's slice of Qwen3-0.6B is ~67MB, and the
// obvious way to load it -- `await res.arrayBuffer()` then `writeBuffer` --
// holds the whole thing in JS heap while the GPU driver makes its own copy.
// That is what killed an iPad on the ONNX path at 126MB: ~250MB peak for a
// 126MB shard. The fix is not a smaller model, it is never holding the tensor
// at all: read the HTTP response as a STREAM and copy each ~4MB chunk into the
// GPU buffer as it arrives. Peak JS memory becomes the staging chunk, which is
// independent of how big the tensor is.
//
// THE SPLIT LAYOUT. The on-disk Q8_0 layout interleaves a 2-byte f16 scale with
// 32 int8 quants every 34 bytes, so nothing is u32-aligned and a kernel has to
// extract bytes with shift/mask. The kernels in kernels/ want two separate
// buffers instead: `qs` as contiguous int8 (8 per u32 word) and `scales` as raw
// f16 packed two per u32 word. See splitQ8/splitQ4 in kernels/lib.ts -- this
// module produces byte-identical output, but does it INCREMENTALLY as bytes
// arrive rather than over a whole materialised tensor. That is the only reason
// the code here is more than a call to splitQ8.
//
// The incremental part has one subtlety worth naming: a 34-byte block does not
// divide a network chunk, so most chunks end mid-block. The splitter therefore
// carries a partial-block remainder across chunk boundaries, and every write to
// the GPU is aligned to a whole number of blocks. Getting this wrong shifts the
// stream by a few bytes, which decodes as valid-looking Q8_0 and produces
// plausible garbage rather than an error.
//
// Browser-safe on purpose: no npm specifiers, no Node builtins. It is served to
// phones.

import {
  readDirectory, layerTensors, byteRanges, totalBytes, fetchRange,
  F32, F16, Q4_0, Q8_0,
} from './gguf-dir.mjs';

export {readDirectory, layerTensors, byteRanges, totalBytes, fetchRange};

export const Q8_BLOCK = 32, Q8_BYTES = 34;
export const Q4_BLOCK = 32, Q4_BYTES = 18;

// 4MB, the same figure swarmllm stages at. Small enough that peak JS heap is
// noise next to a 67MB slice, large enough that the per-write overhead and the
// yield below are amortised over a useful amount of data.
export const STAGING = 4 << 20;

/** Per-dtype block geometry, for the two quantizations the kernels decode. */
function blockSpec(dtype) {
  if (dtype === Q8_0) return {block: Q8_BLOCK, bytes: Q8_BYTES, qsPerBlock: 32};
  if (dtype === Q4_0) return {block: Q4_BLOCK, bytes: Q4_BYTES, qsPerBlock: 16};
  return null;
}

/**
 * Allocate a storage buffer, turning an OOM into a sentence a person can act on.
 *
 * WebGPU reports allocation failure asynchronously through an error scope, not
 * as a throw from createBuffer, so without the scope an over-committed device
 * fails later and somewhere else -- usually as a mystery validation error on the
 * first dispatch. Catching it here is what lets the bird say "this device
 * pledged more layers than its GPU can hold" instead.
 */
export async function allocStorage(device, size, label) {
  // WebGPU requires a 4-byte-aligned size and the 34-byte Q8_0 layout is not,
  // so a tensor's scales array can land on an odd size.
  const aligned = (size + 3) & ~3;
  device.pushErrorScope('out-of-memory');
  const buf = device.createBuffer({
    label,
    size: aligned,
    // COPY_SRC as well as COPY_DST so a caller can read a buffer back and check
    // it against a reference -- which is how this module is tested.
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const err = await device.popErrorScope();
  if (err) {
    throw new Error(
      `this device pledged more than its GPU can hold: ${label} needs ` +
      `${(aligned / 1e6).toFixed(1)}MB and the allocation failed (${err.message})`);
  }
  return buf;
}

/**
 * One set of staging buffers, shared by every tensor of a load.
 *
 * A bird streams 44 tensors and each needs a coalescing buffer, a qs buffer and a
 * scales buffer. Allocating them per tensor is correct but leaves 44 sets of
 * multi-MB arrays as garbage, and the GC does not keep up while the loop is
 * saturating the network -- measured peak JS memory was ~2x the working set for
 * that reason alone. Buffers here are allocated at the largest size any tensor
 * asked for and handed out as subarrays, so the pool never shrinks and never
 * grows past one tensor's need.
 */
class Scratch {
  constructor() { this._qs = null; this._sc = null; this._co = null; }
  #take(field, n) {
    if (!this[field] || this[field].byteLength < n) this[field] = new Uint8Array(n);
    return this[field].subarray(0, n);
  }
  qs(n) { return this.#take('_qs', n); }
  sc(n) { return this.#take('_sc', n); }
  coalesce(n) { return this.#take('_co', n); }
}

/**
 * Incrementally de-interleave a Q8_0/Q4_0 byte stream into qs and scales.
 *
 * Fed arbitrary-length chunks, it emits writes only at whole-block boundaries
 * and holds at most one partial block (34 bytes) plus one staging buffer's worth
 * of output. `flush` is called with (qsBytes, qsOffset, scaleBytes, scaleOffset)
 * and must not retain the arrays -- they are reused.
 */
class SplitStreamer {
  constructor(dtype, staging, flush, scratch = null) {
    const spec = blockSpec(dtype);
    if (!spec) throw new Error(`SplitStreamer: dtype ${dtype} is not a block quant`);
    this.spec = spec;
    this.flush = flush;
    this.scratch = scratch;
    // How many blocks fit in one staging flush -- rounded DOWN to an even count,
    // which is not cosmetic. writeBuffer requires both its offset and its size to
    // be 4-byte aligned, and a flush of N blocks writes exactly 2N bytes of
    // scales; at an odd N that is a 2-mod-4 size, and the flush after it starts
    // at a 2-mod-4 offset. Measured symptom with staging=100003 (2941 blocks):
    // the scales buffer diverged from the reference at byte 5882, which is
    // exactly 2*2941 -- the second flush's offset. An even block count makes
    // every scales write and every scales offset a multiple of 4 for free.
    const fit = Math.floor(staging / spec.bytes);
    this.blocksPerFlush = Math.max(2, fit - (fit % 2));
    // Staging buffers come from a caller-supplied pool when there is one. A bird
    // streams 44 tensors, and allocating a fresh pair of multi-MB arrays for each
    // leaves 44 pairs for the GC to catch up with -- which it does not do
    // promptly, so measured peak JS memory came out ~2x what the code actually
    // needs at any instant. Reusing one pool made it flat.
    const qsSize = this.blocksPerFlush * spec.qsPerBlock, scSize = this.blocksPerFlush * 2;
    this.qsOut = scratch ? scratch.qs(qsSize) : new Uint8Array(qsSize);
    this.scOut = scratch ? scratch.sc(scSize) : new Uint8Array(scSize);
    this.pending = 0;                              // blocks sitting in qsOut/scOut
    this.partial = new Uint8Array(spec.bytes);     // a block split across chunks
    this.partialLen = 0;
    this.qsAt = 0;                                 // byte offset into the qs buffer
    this.scAt = 0;                                 // byte offset into the scales buffer
  }

  async push(chunk) {
    const {bytes} = this.spec;
    let at = 0;

    // Finish the block left over from the previous chunk first, so the main loop
    // below only ever sees whole blocks.
    if (this.partialLen) {
      const want = Math.min(bytes - this.partialLen, chunk.byteLength);
      this.partial.set(chunk.subarray(0, want), this.partialLen);
      this.partialLen += want;
      at = want;
      if (this.partialLen < bytes) return;         // still incomplete
      await this.#block(this.partial, 0);
      this.partialLen = 0;
    }

    while (at + bytes <= chunk.byteLength) {
      await this.#block(chunk, at);
      at += bytes;
    }

    const left = chunk.byteLength - at;
    if (left > 0) {
      this.partial.set(chunk.subarray(at), 0);
      this.partialLen = left;
    }
  }

  async #block(src, at) {
    const {bytes, qsPerBlock} = this.spec;
    const i = this.pending;
    this.scOut[i * 2] = src[at];
    this.scOut[i * 2 + 1] = src[at + 1];
    this.qsOut.set(src.subarray(at + 2, at + bytes), i * qsPerBlock);
    this.pending++;
    if (this.pending === this.blocksPerFlush) await this.#emit();
  }

  async #emit() {
    if (!this.pending) return;
    const {qsPerBlock} = this.spec;
    const qsLen = this.pending * qsPerBlock, scLen = this.pending * 2;
    await this.flush(
      this.qsOut.subarray(0, qsLen), this.qsAt,
      this.scOut.subarray(0, scLen), this.scAt);
    this.qsAt += qsLen;
    this.scAt += scLen;
    this.pending = 0;
  }

  /** No more bytes are coming: emit what is buffered and check nothing is left. */
  async end() {
    if (this.partialLen) {
      throw new Error(
        `tensor ended mid-block: ${this.partialLen} of ${this.spec.bytes} bytes ` +
        `(a short read, or the wrong byte range)`);
    }
    await this.#emit();
  }
}

/**
 * Copy `src` into `buf` at `offset`, then let the GPU process actually take it.
 *
 * The yield is not politeness. writeBuffer copies into a driver staging area and
 * that copy happens off the JS thread; without giving WebKit a turn, a tight
 * device queues faster than it drains and dies with the staging allocations
 * outstanding -- which looks like an OOM on a 4MB buffer. swarmllm carries the
 * same setTimeout(0) with the same reasoning.
 */
async function writeAndYield(device, buf, offset, src) {
  // writeBuffer requires a 4-byte-aligned size and offset. A quantized tensor's
  // scales run 2 bytes per block, so only the final flush of an odd block count
  // can be misaligned -- pad it rather than refusing the write. The buffer was
  // allocated 4-byte-aligned, so the padding lands inside it.
  const size = src.byteLength;
  // A misaligned OFFSET cannot be fixed by padding -- it would write the bytes to
  // the wrong place. The block-count rounding in SplitStreamer is what keeps this
  // from happening, so an assert here is how that invariant stays true rather
  // than quietly producing a buffer that is a few bytes out of phase.
  if (offset & 3) {
    throw new Error(`writeBuffer offset ${offset} is not 4-byte aligned (${buf.label})`);
  }
  if (size & 3) {
    // Only the FINAL flush of a tensor can be misaligned, when its total block
    // count is odd. The buffer was allocated 4-byte-aligned, so the pad lands
    // inside it and past the tensor's own bytes.
    const pad = new Uint8Array((size + 3) & ~3);
    pad.set(src);
    device.queue.writeBuffer(buf, offset, pad);
  } else {
    // A Uint8Array view's own byteOffset is already baked in, so the data offset
    // is 0 -- passing src.byteOffset here would double-count it.
    device.queue.writeBuffer(buf, offset, src, 0, size);
  }
  await new Promise(r => setTimeout(r, 0));
}

/**
 * Stream one tensor's byte range into GPU buffers.
 *
 * `openRange(start, end)` returns a Response; injected rather than fetched here
 * so a caller can serve it from an IndexedDB cache, from a coordinator proxy, or
 * from a test fixture without this function knowing the difference.
 *
 * Returns {qs, scales, rows, cols, dtype} for a quantized tensor, or
 * {data, rows, cols, dtype} for an f32/f16 one (norm gains: a few KB, no split
 * to do, so they go into a single buffer).
 *
 * GGUF shapes are [in, out] -- ggml is column-major and lists the fastest
 * varying dimension first -- so shape[0] is the row width (`cols`) and shape[1]
 * is the number of rows.
 */
export async function streamTensorToGPU(device, tensor, openRange, opts = {}) {
  const {staging = STAGING, onProgress, absolute, scratch = null} = opts;
  const start = absolute ?? tensor.absolute;
  if (start == null) throw new Error(`tensor ${tensor.name} has no absolute offset`);
  const end = start + tensor.bytes;
  const [cols, rows] = tensor.shape.length > 1
    ? [Number(tensor.shape[0]), Number(tensor.shape[1])]
    : [Number(tensor.shape[0]), 1];

  const spec = blockSpec(tensor.dtype);
  let out, streamer, sink;

  if (spec) {
    if (cols % spec.block !== 0) {
      throw new Error(`${tensor.name}: ${cols} cols not divisible by block ${spec.block}`);
    }
    const nb = (rows * cols) / spec.block;
    const qs = await allocStorage(device, nb * spec.qsPerBlock, `${tensor.name}.qs`);
    const scales = await allocStorage(device, nb * 2, `${tensor.name}.scales`);
    out = {name: tensor.name, dtype: tensor.dtype, rows, cols, qs, scales,
           qsBytes: nb * spec.qsPerBlock, scaleBytes: nb * 2};
    streamer = new SplitStreamer(tensor.dtype, staging, async (q, qAt, s, sAt) => {
      await writeAndYield(device, qs, qAt, q);
      await writeAndYield(device, scales, sAt, s);
    }, scratch);
    sink = chunk => streamer.push(chunk);
  } else if (tensor.dtype === F32 || tensor.dtype === F16) {
    // Unquantized: nothing to de-interleave, so it goes straight in. These are
    // the norm gains, a few KB each.
    const buf = await allocStorage(device, tensor.bytes, tensor.name);
    out = {name: tensor.name, dtype: tensor.dtype, rows, cols, data: buf,
           dataBytes: tensor.bytes};
    let at = 0;
    sink = async chunk => {
      await writeAndYield(device, buf, at, chunk);
      at += chunk.byteLength;
    };
  } else {
    throw new Error(`${tensor.name}: dtype ${tensor.dtype} not supported for GPU upload`);
  }

  // Read the response body as a stream and hand it on in staging-sized pieces.
  // Network chunks are whatever size the transport gives us -- often 16-64KB --
  // so they are coalesced up to `staging` before each GPU write, or a 67MB slice
  // becomes thousands of tiny writeBuffer calls and thousands of yields.
  const res = await openRange(start, end);
  if (!res.ok && res.status !== 206) {
    throw new Error(`${tensor.name}: range fetch failed ${res.status}`);
  }
  let got = 0;
  if (res.body) {
    const reader = res.body.getReader();
    // A tensor smaller than the staging size does not need a full one, which
    // matters for the 512-byte norm gains sitting between the projections.
    const want = Math.min(staging, Math.max(4096, tensor.bytes));
    const coalesce = scratch ? scratch.coalesce(want) : new Uint8Array(want);
    const chunkSize = coalesce.byteLength;
    let held = 0;
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      let at = 0;
      while (at < value.byteLength) {
        const take = Math.min(chunkSize - held, value.byteLength - at);
        coalesce.set(value.subarray(at, at + take), held);
        held += take; at += take;
        if (held === chunkSize) { await sink(coalesce); held = 0; }
      }
      got += value.byteLength;
      if (got > tensor.bytes) {
        throw new Error(`${tensor.name}: server sent more than the requested ${tensor.bytes} bytes`);
      }
      onProgress?.(got, tensor.bytes);
    }
    if (held) await sink(coalesce.subarray(0, held));
  } else {
    // No streaming body (a cached ArrayBuffer, or a transport without one). Still
    // fed through the same sink in staging pieces so the split logic has exactly
    // one implementation.
    const all = new Uint8Array(await res.arrayBuffer());
    for (let at = 0; at < all.byteLength; at += staging) {
      await sink(all.subarray(at, Math.min(at + staging, all.byteLength)));
      got = Math.min(at + staging, all.byteLength);
      onProgress?.(got, tensor.bytes);
    }
  }
  if (streamer) await streamer.end();
  if (got !== tensor.bytes) {
    throw new Error(`${tensor.name}: short read ${got}/${tensor.bytes} bytes`);
  }
  return out;
}

// ------------------------------------------------------------------ the cache
//
// IndexedDB, NOT the Cache API. `caches` only exists in a secure context and a
// phone reaches the coordinator over plain http:// on a LAN, so it is undefined
// there -- the same reason bird.html caches its ONNX shards this way. IndexedDB
// has no such restriction and holds blobs of this size fine.
//
// What is cached is the RAW range bytes, not the split buffers. Caching the
// split form would save the de-interleave on a refresh, but a refresh is not
// where the time goes: the download is. And raw ranges are what the reference
// path reads, so one cached artifact serves both.

const DB_NAME = 'flock-gguf', DB_VERSION = 1, STORE = 'ranges';

export function openCache() {
  return new Promise(res => {
    let req;
    try { req = indexedDB.open(DB_NAME, DB_VERSION); } catch { return res(null); }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => res(req.result);
    req.onerror = () => res(null);          // no cache is survivable
  });
}

function cacheGet(db, key) {
  return new Promise(res => {
    if (!db) return res(null);
    try {
      const r = db.transaction(STORE).objectStore(STORE).get(key);
      r.onsuccess = () => res(r.result || null);
      r.onerror = () => res(null);
    } catch { res(null); }
  });
}

function cachePut(db, key, val) {
  return new Promise((res, rej) => {
    if (!db) return res();
    try {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(val, key);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error || new Error('idb write failed'));
    } catch (e) { rej(e); }
  });
}

/** Drop every cached range. Exposed so a device tight on storage can reclaim it. */
export function clearCache(db) {
  return new Promise(res => {
    if (!db) return res();
    try {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).clear();
      tx.oncomplete = () => res();
      tx.onerror = () => res();
    } catch { res(); }
  });
}

/**
 * A range reader over one contiguous slice of the model, backed by an optional
 * IndexedDB cache.
 *
 * ONE HTTP REQUEST FOR THE WHOLE SLICE. This is the part that took a measurement
 * to get right. The obvious implementation gives each tensor its own range
 * request, and that is what this did first: 44 requests for a 4-layer bird, which
 * is 44 round trips to a CDN for bytes that are contiguous on disk. Instead a
 * single request is opened over the merged range and its body reader is handed
 * out to each tensor in offset order -- so the 44 tensors of layers 24-27 are 44
 * sequential reads from ONE response stream. The tensors must be consumed in
 * ascending offset order for this to work, which is what layerTensors + the
 * offset sort in byteRanges already guarantee; anything out of order falls back
 * to its own request rather than silently reading the wrong bytes.
 *
 * The cache is keyed on the slice for the same reason: one entry per bird rather
 * than eleven per layer.
 *
 * `blob` is held only when the slice came from (or has just been written to) the
 * cache. On the cold path nothing is retained -- which is the point of this
 * module, so the cold path never assembles the blob at all unless caching is on.
 */
export class RangeSource {
  constructor(url, {db = null, cache = true, onNote} = {}) {
    this.url = url;
    this.db = db;
    this.cache = cache && !!db;
    this.onNote = onNote;
    this.blob = null;
    this.base = 0;
    this.requests = 0;
    this.fetchedBytes = 0;
    this.fromCache = false;
    // The shared sequential stream: a reader over the merged range, and how far
    // through it we have read.
    this.reader = null;
    this.readerAt = 0;
    this.readerEnd = 0;
    this.held = null;        // bytes read from `reader` but not yet consumed
  }

  key(range) { return `${this.url}#${range.start}-${range.end}`; }

  /**
   * Make [range.start, range.end) available. On a cache hit the bytes are loaded
   * from IndexedDB; on a miss nothing is fetched here -- individual tensors
   * stream straight from the network, one request for the merged range.
   *
   * Returns true when the bytes are resident (so every later read is local).
   */
  async prime(range) {
    this.base = range.start;
    if (!this.cache) return false;
    const hit = await cacheGet(this.db, this.key(range));
    if (!hit) return false;
    const bytes = hit instanceof ArrayBuffer ? new Uint8Array(hit) : new Uint8Array(hit.buffer || hit);
    if (bytes.byteLength !== range.end - range.start) {
      this.onNote?.(`cached slice is ${bytes.byteLength} bytes, expected ` +
                   `${range.end - range.start} — refetching`);
      return false;
    }
    this.blob = bytes;
    this.fromCache = true;
    this.onNote?.(`${(bytes.byteLength / 1e6).toFixed(0)}MB from cache`);
    return true;
  }

  /**
   * Open ONE request over the merged range and keep its reader, so every tensor
   * inside it is a sequential read from the same response body.
   */
  async openSlice(range) {
    if (this.blob) return;                 // resident; nothing to stream
    this.requests++;
    this.fetchedBytes += range.end - range.start;
    const r = await fetch(this.url, {
      headers: {Range: `bytes=${range.start}-${range.end - 1}`},
    });
    if (!r.ok && r.status !== 206) throw new Error(`range fetch failed: ${r.status}`);
    if (!r.body) {
      // No streaming body at all (some polyfills, and file:// in a few engines).
      // Fall back to holding the slice; the per-tensor path still works, it just
      // costs the memory this module exists to avoid, so say so.
      this.onNote?.('no streaming response body — falling back to a buffered slice');
      this.blob = new Uint8Array(await r.arrayBuffer());
      this.base = range.start;
      return;
    }
    this.reader = r.body.getReader();
    this.readerAt = range.start;
    this.readerEnd = range.end;
    this.held = null;
  }

  /**
   * A Response for [start, end).
   *
   * Three paths, in order of preference: out of the resident slice (cache hit),
   * out of the shared sequential stream (the normal cold path -- zero extra
   * requests), or its own range request (a read that is not where the shared
   * stream is sitting, so there is nothing to reuse).
   */
  async open(start, end) {
    if (this.blob) {
      const from = start - this.base, to = end - this.base;
      if (from < 0 || to > this.blob.byteLength) {
        throw new Error(`range ${start}-${end} outside the primed slice`);
      }
      // A Response over a subarray, so the caller's streaming path is identical
      // whether the bytes came from the network or from the cache.
      return new Response(this.blob.subarray(from, to), {status: 206});
    }

    if (this.reader && start >= this.readerAt && end <= this.readerEnd) {
      // Skip any gap between tensors rather than seeking, which a stream cannot
      // do. Gaps inside a merged range are at most `gap` bytes by construction.
      if (start > this.readerAt) await this.#skip(start - this.readerAt);
      return new Response(this.#substream(end - start), {status: 206});
    }

    this.requests++;
    this.fetchedBytes += end - start;
    const r = await fetch(this.url, {headers: {Range: `bytes=${start}-${end - 1}`}});
    if (!r.ok && r.status !== 206) {
      throw new Error(`range fetch failed: ${r.status}`);
    }
    return r;
  }

  /** Pull and discard n bytes of the shared stream. */
  async #skip(n) {
    let left = n;
    while (left > 0) {
      const chunk = await this.#pull();
      if (!chunk) throw new Error(`slice stream ended ${left} bytes early`);
      const take = Math.min(left, chunk.byteLength);
      this.held = take < chunk.byteLength ? chunk.subarray(take) : null;
      this.readerAt += take;
      left -= take;
    }
  }

  /** The next chunk of the shared stream: whatever is held, else a read. */
  async #pull() {
    if (this.held) { const h = this.held; this.held = null; return h; }
    const {value, done} = await this.reader.read();
    return done ? null : value;
  }

  /**
   * A ReadableStream over the next `n` bytes of the shared stream, leaving the
   * remainder of the final chunk held for the next tensor.
   *
   * This is where the one-request claim is actually implemented: the consumer
   * sees a normal streaming body and never knows it is a window onto a larger
   * response.
   */
  #substream(n) {
    let left = n;
    const self = this;
    return new ReadableStream({
      async pull(ctrl) {
        if (left === 0) { ctrl.close(); return; }
        const chunk = await self.#pull();
        if (!chunk) { ctrl.error(new Error('slice stream ended early')); return; }
        const take = Math.min(left, chunk.byteLength);
        self.held = take < chunk.byteLength ? chunk.subarray(take) : null;
        self.readerAt += take;
        left -= take;
        ctrl.enqueue(chunk.subarray(0, take));
        if (left === 0) ctrl.close();
      },
    });
  }

  /** Release the shared stream. Safe to call twice. */
  async close() {
    if (!this.reader) return;
    try { await this.reader.cancel(); } catch {}
    this.reader = null; this.held = null;
  }

  /**
   * Download the whole slice once and cache it, so a refresh is free.
   *
   * This DOES hold the slice in JS memory -- ~67MB for a 4-layer bird -- which is
   * exactly what the streaming path exists to avoid, so it is opt-in and happens
   * AFTER the GPU buffers are already built and inference is known to work. A
   * device tight enough that this is the difference should run with cache:false
   * and pay the download on every refresh; the streaming upload itself never
   * needs it.
   */
  async fill(range) {
    if (!this.cache || this.blob) return;
    this.requests++;
    this.fetchedBytes += range.end - range.start;
    const bytes = await fetchRange(this.url, range);
    this.blob = bytes;
    await cachePut(this.db, this.key(range), bytes.buffer)
      .catch(e => this.onNote?.('cache write skipped: ' + e.message));
  }
}

/**
 * Load every tensor of `layers` into GPU buffers, in the split qs/scales layout.
 *
 * Returns {layers, stats}. `layers` is keyed by layer index, and each layer is
 * keyed by the tensor suffix after `blk.N.` -- so `layers[24].attn_q` -- which is
 * the shape kernels/layer.ts consumes.
 *
 * `onProgress({tensor, i, n, at, total, bytesDone, bytesTotal})` fires as each
 * tensor streams, so the UI can name what is loading rather than showing one
 * opaque bar for 67MB.
 */
export async function loadLayersToGPU(device, model, layers, opts = {}) {
  const {
    staging = STAGING, onProgress, onNote, cache = true, db = null,
    fillCacheAfter = true,
  } = opts;

  const tensors = layers.flatMap(l => layerTensors(model, l));
  if (!tensors.length) throw new Error(`no tensors for layers ${layers.join(',')}`);
  const ranges = byteRanges(model, tensors);
  const bytesTotal = totalBytes(ranges);

  // Layers are contiguous in GGUF, so a bird's slice is normally ONE range.
  // Anything else means the tensors were not adjacent, which is worth saying out
  // loud because it changes the request count.
  if (ranges.length > 1) {
    onNote?.(`slice needs ${ranges.length} range requests (layers not contiguous)`);
  }

  const src = new RangeSource(model.url, {db, cache, onNote});
  const resident = await src.prime(ranges[0]);
  // One pool for the whole load, not one per tensor. See Scratch.
  const scratch = new Scratch();

  const out = {};
  let bytesDone = 0;
  try {
    // Tensors are already in ascending offset order, which is what lets the whole
    // slice come down one range at a time rather than one request per tensor.
    for (const range of ranges) {
      if (!resident) await src.openSlice(range);
      const inRange = tensors.filter(t => {
        const a = model.dataStart + t.offset;
        return a >= range.start && a + t.bytes <= range.end;
      });
      for (const t of inRange) {
        const absolute = model.dataStart + t.offset;
        const base = bytesDone;
        const i = tensors.indexOf(t);
        const up = await streamTensorToGPU(device, {...t, absolute}, (a, b) => src.open(a, b), {
          staging, scratch,
          onProgress: (at, total) => onProgress?.({
            tensor: t.name, i, n: tensors.length, at, total,
            bytesDone: base + at, bytesTotal,
          }),
        });
        bytesDone += t.bytes;

        const m = /^blk\.(\d+)\.(.+?)(?:\.weight)?$/.exec(t.name);
        const layer = m ? Number(m[1]) : -1;
        const key = m ? m[2] : t.name;
        (out[layer] ??= {})[key] = up;
      }
      await src.close();
    }
  } finally {
    await src.close();
  }

  // Caching happens last and on purpose: only once every buffer is on the GPU is
  // it safe to spend 67MB of JS heap on a copy to store. A device that cannot
  // afford it has already succeeded at the part that matters.
  let cached = false;
  if (cache && db && fillCacheAfter && !src.fromCache) {
    try {
      await src.fill(ranges[0]);
      cached = true;
    } catch (e) {
      onNote?.('cache fill skipped: ' + e.message);
    }
  }

  return {
    layers: out,
    stats: {
      tensors: tensors.length, ranges: ranges.length, bytesTotal,
      requests: src.requests, fromCache: src.fromCache, cached,
      staging,
    },
  };
}

/**
 * The whole bird-side load, from a URL and a layer range to GPU buffers.
 *
 * One call so bird.html does not have to know about directories, ranges, or
 * caches: it has a device, a model URL, and the layers the coordinator gave it.
 */
export async function loadBirdWeights(device, url, first, last, opts = {}) {
  const {onNote, onProgress, staging = STAGING, cache = true} = opts;
  const t0 = (globalThis.performance ?? Date).now();
  const model = await readDirectory(url, {chunk: opts.headerChunk});
  onNote?.(`directory: ${model.tensors.length} tensors, ${model.nLayers} layers ` +
           `(${(model.headerFetched / 1e6).toFixed(1)}MB header, ` +
           `${model.headerRequests} requests)`);
  if (last >= model.nLayers) {
    throw new Error(`layer ${last} is past the model's ${model.nLayers} layers`);
  }
  const db = cache ? await openCache() : null;
  const layers = [];
  for (let l = first; l <= last; l++) layers.push(l);
  const {layers: built, stats} = await loadLayersToGPU(device, model, layers, {
    staging, onProgress, onNote, cache, db,
  });
  const secs = ((globalThis.performance ?? Date).now() - t0) / 1000;
  return {model, layers: built, stats: {...stats, seconds: secs}};
}
