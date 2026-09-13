// The GPU side of tensor parallelism: a matvec whose weight matrix is cut across
// N shards, in either direction, plus the sharded LM head that motivates it.
//
// WHY, in one measured line: Qwen3-14B Q4_K_M's `output.weight` is 638 MB as ONE
// tensor with tie_word_embeddings: False, and WebGPU's default
// maxStorageBufferBindingSize is 128 MiB. Cutting the model by layer range --
// what flock does today and what test_split.ts validates -- cannot help, because
// no number of devices makes a single tensor smaller. See shard_ref.ts's header
// for the full argument and memShardPlan for the arithmetic.
//
// WHAT IS HERE. `ShardedMatvec` holds N shards of one tensor on ONE device and
// runs them as N dispatches plus (column-wise only) one reduce dispatch. Running
// N shards on one device is not what tensor parallelism is FOR -- it buys nothing
// on a single device and bench_shard.ts measures exactly how much it costs -- but
// it is what makes the correctness claim testable: the unsharded kernel, N shards
// and an FMA-modelled CPU reference all run on the same inputs and the same
// hardware, so a difference is the split and nothing else. A real deployment puts
// each shard on its own device and replaces the local dispatch with a message;
// the kernel, the slicing and the reduction order do not change.
//
// THE DIRECTION CHOSEN FOR THE LM HEAD IS ROW-WISE. Recorded here because the
// code makes it look like a free option and it is not:
//
//   * bit-identical to the unsharded kernel (proven, N = 1..4 and a non-divisible
//     split) versus column-wise's necessary reassociation
//   * the slice is a contiguous byte range -- a remote shard range-fetches its own
//     bytes and never sees the rest
//   * the wire carries `cols` floats of x broadcast in (4 KB) instead of `rows`
//     floats of partials all-reduced out (608 KB per shard per token at
//     vocab 151936)
//   * argmax over a concatenation is the max of per-shard (value, index) pairs,
//     which argmax.wgsl's stage 1 already produces -- no new kernel
//
// Column-wise is built and validated anyway, because "row-wise is better" is not
// a finding unless the alternative exists and was measured.

import {
  coopSource, probeUnpack, probeUnpackF16, uniformBuffer, type SplitQ8,
} from "./lib.ts";
import { wgslSource } from "./layer.ts";
import {
  memShardPlan, shardRanges, sliceCols, sliceRows, type ShardRange,
} from "./shard_ref.ts";

const ROWS_PER_WG = 4;     // must match q8_shard.wgsl
const REDUCE_WG = 256;     // must match shard_reduce.wgsl

export type Direction = "row" | "col";

/**
 * storageBuffer, but correct for a TypedArray that VIEWS part of a larger buffer.
 *
 * THIS IS CORRECTNESS TRAP 2 FROM kernels/README.md, and row-wise sharding walks
 * straight into it. `sliceRows` hands back subarray VIEWS of the whole repacked
 * tensor -- that is the entire reason row-wise is cheap to load -- and lib.ts's
 * `storageBuffer` ends in `queue.writeBuffer(buf, 0, bytes)`, which on Deno
 * 2.1.2's wgpu IGNORES a view's byteOffset and length and uploads the whole
 * underlying ArrayBuffer from 0.
 *
 * It was caught here only because a shard's destination buffer is SMALLER than
 * the tensor, so the overrun is a loud validation error ("Copy of 0..2048 would
 * end up overrunning the bounds of the Destination buffer of size 1024"). Had the
 * shards been uploaded into one buffer sized for the whole tensor, every shard
 * would have silently received SHARD 0's weights -- N identical shards, plausible
 * logits, and a row-wise "bit-identical" test that passes for N = 1 and fails
 * mysteriously above it. Which is the trap's exact signature: the bounds error
 * only appears when the destination happens to be too small.
 *
 * `writeBuffer(buf, 0, arrayBuffer, byteOffset, byteLength)` IS honoured, so that
 * is what this uses. lib.ts is left alone: every existing caller passes a whole
 * array, where the two spellings agree.
 */
function storageBufferView(dev: GPUDevice, data: Uint8Array): GPUBuffer {
  const size = (data.byteLength + 3) & ~3;
  const buf = dev.createBuffer({
    size: Math.max(4, size),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  if (data.byteLength % 4 === 0) {
    dev.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  } else {
    // writeBuffer's size must be a multiple of 4, and a shard of the 2-byte
    // scales array is odd-length whenever its block count is odd. Only this case
    // copies, and it copies `scales` (cols/32 * 2 bytes per row), never `qs`, so
    // the zero-copy property that makes row-wise cheap is preserved for the
    // buffer that actually carries the weight bytes.
    const p = new Uint8Array(size);
    p.set(data);
    dev.queue.writeBuffer(buf, 0, p);
  }
  return buf;
}

/** Which decode spellings this device's WGSL supports, probed once per device. */
const probeCache = new WeakMap<GPUDevice, Promise<{ unpack8: boolean; unpackF16: boolean }>>();
function probeOnce(dev: GPUDevice) {
  let p = probeCache.get(dev);
  if (!p) {
    p = (async () => ({
      unpack8: await probeUnpack(dev), unpackF16: await probeUnpackF16(dev),
    }))();
    probeCache.set(dev, p);
  }
  return p;
}

export interface ShardedMatvecOpts {
  /**
   * Bytes a single storage binding may be. Checked BEFORE upload, because
   * exceeding it is a validation error that makes a kernel write zeros rather
   * than throwing at the call site -- correctness trap 3. Defaults to the
   * device's real limit; pass a smaller number to simulate a constrained device,
   * which is what test_shard_mem.ts does.
   */
  bindingLimit?: number;
}

/** One shard's resident buffers and the uniform that tells the kernel its shape. */
interface ShardGpu {
  range: ShardRange;
  qs: GPUBuffer;
  scales: GPUBuffer;
  dims: GPUBuffer;
  /** Dispatch count: rows this shard produces, over ROWS_PER_WG. */
  groups: number;
  qsBytes: number;
  scalesBytes: number;
}

/**
 * A matvec y = W x with W held as N shards.
 *
 * Row-wise: each shard writes its own slice of `out` directly, so `out` IS the
 * answer after N dispatches and there is no reduction.
 *
 * Column-wise: shard s writes a full-length partial at offset s * rows of a
 * `n * rows` partials buffer, and one reduce dispatch sums them into `out`. The
 * partials buffer is the thing that makes column-wise expensive in a distributed
 * setting -- it is what has to cross the wire.
 */
export class ShardedMatvec {
  readonly rows: number;
  readonly cols: number;
  readonly direction: Direction;
  readonly ranges: ShardRange[];
  private dev: GPUDevice;
  private shards: ShardGpu[] = [];
  private matvec: GPUComputePipeline;
  private reduce: GPUComputePipeline | null = null;
  private xBuf: GPUBuffer;
  private outBuf: GPUBuffer;
  private partsBuf: GPUBuffer | null = null;
  private reduceDims: GPUBuffer | null = null;
  private bgs: GPUBindGroup[] = [];
  private reduceBg: GPUBindGroup | null = null;
  /** Bytes of weights actually uploaded, for the memory report. */
  readonly uploadedBytes: number;

  private constructor(
    dev: GPUDevice, rows: number, cols: number, direction: Direction,
    ranges: ShardRange[], matvec: GPUComputePipeline, reduce: GPUComputePipeline | null,
    shards: ShardGpu[], xBuf: GPUBuffer, outBuf: GPUBuffer,
    partsBuf: GPUBuffer | null, reduceDims: GPUBuffer | null,
  ) {
    this.dev = dev;
    this.rows = rows; this.cols = cols; this.direction = direction;
    this.ranges = ranges; this.matvec = matvec; this.reduce = reduce;
    this.shards = shards; this.xBuf = xBuf; this.outBuf = outBuf;
    this.partsBuf = partsBuf; this.reduceDims = reduceDims;
    this.uploadedBytes = shards.reduce((a, s) => a + s.qsBytes + s.scalesBytes, 0);

    // Bind groups built once. Rebuilding them per dispatch cost ~4 ms/layer of
    // 5.4 in the layer benchmark, and a sharded matvec issues N of them.
    const target = direction === "col" ? this.partsBuf! : this.outBuf;
    this.bgs = shards.map((s) => dev.createBindGroup({
      layout: matvec.getBindGroupLayout(0),
      entries: [s.qs, s.scales, this.xBuf, target, s.dims].map(
        (buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    if (direction === "col") {
      this.reduceBg = dev.createBindGroup({
        layout: reduce!.getBindGroupLayout(0),
        entries: [this.partsBuf!, this.outBuf, this.reduceDims!].map(
          (buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
  }

  /**
   * Build a sharded matvec over a tensor already repacked into the split layout.
   *
   * `split` is the WHOLE tensor; the slicing happens here so the caller does not
   * have to know which direction implies a gather. Row-wise slices are subarray
   * views (no copy); column-wise slices are gathered (a full pass per shard).
   * The views go through `storageBufferView` above, NOT lib.ts's storageBuffer --
   * see the note there; the latter would upload the whole tensor to every shard.
   */
  static async create(
    dev: GPUDevice, split: SplitQ8, rows: number, cols: number,
    n: number, direction: Direction, opts: ShardedMatvecOpts = {},
  ): Promise<ShardedMatvec> {
    if (cols % 32 !== 0) throw new Error(`cols ${cols} must be a multiple of 32`);
    const limit = opts.bindingLimit ?? dev.limits.maxStorageBufferBindingSize;

    // The capacity check, before any upload. memShardPlan is the same arithmetic
    // test_shard_mem.ts asserts against, so a configuration that passes that test
    // is a configuration this constructor accepts.
    const plan = memShardPlan(
      { name: "W", rows, cols }, n, direction, limit);
    if (!plan.fits) {
      throw new Error(
        `${direction}-wise ${n} shards of ${rows}x${cols} need a ${plan.maxBindingBytes} ` +
        `byte binding, over the ${limit} byte limit; ` +
        (plan.minShardsForLimit === null
          ? "no shard count fits"
          : `${plan.minShardsForLimit} shards would fit`));
    }

    // Probed once per device and cached: the probe deliberately compiles a shader
    // that fails on this backend (no unpack4xI8), and wgpu logs that to stderr
    // regardless of the error scope, so probing per construction buries a test's
    // output in the same expected message N times.
    const { unpack8, unpackF16 } = await probeOnce(dev);
    const mk = (code: string) => dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code }), entryPoint: "main" },
    });
    const matvec = mk(coopSource(await wgslSource("q8_shard.wgsl"), { unpack8, unpackF16 }));
    const reduce = direction === "col" ? mk(await wgslSource("shard_reduce.wgsl")) : null;

    const ranges = direction === "row"
      ? shardRanges(rows, n, 1)
      : shardRanges(cols, n, 32);

    const rw = (bytes: number) => dev.createBuffer({
      size: Math.max(16, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // x is the WHOLE input in both directions. Column-wise shards read their own
    // window through `col_off` rather than being handed a pre-sliced copy, which
    // is what lets test_shard.ts assert that slicing x on the host and offsetting
    // into it on the device give the identical answer -- i.e. that the split lives
    // in the indexing and not in the data movement. A remote shard receives only
    // its slice and passes col_off = 0; same kernel either way.
    const xBuf = rw(cols * 4);
    const outBuf = rw(rows * 4);
    const partsBuf = direction === "col" ? rw(n * rows * 4) : null;
    const reduceDims = direction === "col"
      ? uniformBuffer(dev, [rows, n, 0, 0])
      : null;

    const shards: ShardGpu[] = ranges.map((rg) => {
      const sl = direction === "row"
        ? sliceRows(split, rows, cols, rg.start, rg.end)
        : sliceCols(split, rows, cols, rg.start, rg.end);
      const sRows = direction === "row" ? rg.count : rows;
      const sCols = direction === "row" ? cols : rg.count;
      return {
        range: rg,
        qs: storageBufferView(dev, sl.qs),
        scales: storageBufferView(dev, sl.scales),
        // out_off: row-wise writes into its own slice of the answer; column-wise
        // writes a full-length partial into slot s of the partials buffer.
        // col_off is in VEC4 units because x is bound as array<vec4<f32>>, and a
        // 32-column block is 8 vec4s -- which is why every column boundary is a
        // multiple of 32 (shardRanges enforces it) and col_off/4 is always exact.
        dims: uniformBuffer(dev, [
          sRows, sCols,
          direction === "row" ? rg.start : rg.index * rows,
          direction === "row" ? 0 : rg.start / 4,
        ]),
        groups: Math.ceil(sRows / ROWS_PER_WG),
        qsBytes: sl.qs.byteLength,
        scalesBytes: sl.scales.byteLength,
      };
    });

    return new ShardedMatvec(
      dev, rows, cols, direction, ranges, matvec, reduce, shards,
      xBuf, outBuf, partsBuf, reduceDims);
  }

  /** Upload the input vector. `cols` floats, the whole x, in both directions. */
  setInput(x: Float32Array) {
    if (x.length !== this.cols) {
      throw new Error(`x is ${x.length} floats, want ${this.cols}`);
    }
    // Copied via a fresh array rather than passed as a view: writeBuffer ignores
    // a view's byteOffset on this backend (correctness trap 2), and a caller
    // handing in a subarray of a larger activation buffer is the normal case.
    this.dev.queue.writeBuffer(this.xBuf, 0, x.slice());
  }

  /**
   * Encode every shard's dispatch, plus the reduce for column-wise, into `enc`.
   *
   * One compute pass covering all N shards. They write disjoint regions (row-wise
   * disjoint slices of `out`, column-wise disjoint slots of `parts`), so there is
   * no ordering requirement between them -- which is exactly the property that
   * makes them independent devices in a real deployment. The reduce needs a
   * SEPARATE pass: it reads what the matvecs wrote through a different bind
   * group, and a pass boundary is what makes those writes visible.
   */
  encode(enc: GPUCommandEncoder) {
    const p = enc.beginComputePass();
    this.dispatchShards(p);
    p.end();
    if (this.direction === "col") {
      const p2 = enc.beginComputePass();
      this.dispatchReduce(p2);
      p2.end();
    }
  }

  /** Encode ONLY the shard matvecs, no reduction. For isolating the reduce cost. */
  encodeShardsOnly(enc: GPUCommandEncoder) {
    const p = enc.beginComputePass();
    this.dispatchShards(p);
    p.end();
  }

  /** Encode ONLY the reduction. For isolating its cost. Column-wise only. */
  encodeReduceOnly(enc: GPUCommandEncoder) {
    if (this.direction !== "col") throw new Error("row-wise sharding has no reduction");
    const p = enc.beginComputePass();
    this.dispatchReduce(p);
    p.end();
  }

  /**
   * The N shard dispatches into an EXISTING pass, for a benchmark that wants many
   * repetitions behind one fence.
   *
   * bench.ts's method requires this: a map readback is ~24 ms, so a 20 us matvec
   * has to be measured as a batch of thousands of dispatches with the fence's cost
   * subtracted. Putting each repetition in its own pass -- which is what calling
   * `encode()` in a loop does -- builds thousands of compute passes into one
   * command buffer, and on this backend that WEDGES: the submit never completes
   * and mapAsync never resolves, at 0% CPU, which looks exactly like a slow
   * benchmark rather than a hang. Found by watching a 2000-rep batch sit at
   * 0.19 s of CPU time indefinitely.
   *
   * A benchmark using these is measuring the dispatches and not the pass
   * boundaries, which is the right unit anyway: a real forward pass encodes one
   * pass covering many operations, exactly like layer.ts does.
   */
  dispatchShards(p: GPUComputePassEncoder) {
    p.setPipeline(this.matvec);
    for (let i = 0; i < this.shards.length; i++) {
      p.setBindGroup(0, this.bgs[i]);
      p.dispatchWorkgroups(this.shards[i].groups);
    }
  }

  /** The reduce dispatch into an existing pass. Column-wise only. */
  dispatchReduce(p: GPUComputePassEncoder) {
    if (this.direction !== "col") throw new Error("row-wise sharding has no reduction");
    p.setPipeline(this.reduce!);
    p.setBindGroup(0, this.reduceBg!);
    p.dispatchWorkgroups(Math.ceil(this.rows / REDUCE_WG));
  }

  /** The buffer holding the final y. */
  output(): GPUBuffer { return this.outBuf; }

  /** The n * rows partials buffer. Column-wise only; this is the all-reduce payload. */
  partials(): GPUBuffer {
    if (!this.partsBuf) throw new Error("row-wise sharding has no partials buffer");
    return this.partsBuf;
  }

  /** Bytes one matvec would have to move between devices, per call. */
  wireBytesPerCall(): number {
    return this.direction === "row"
      // x broadcast to every shard, and each shard's output slice comes back.
      // The broadcast is one message to N devices; counted once per shard here so
      // the two directions are counted the same way.
      ? this.shards.length * this.cols * 4 + this.rows * 4
      // x's slice out to each shard, and a full-length partial back from each.
      : this.cols * 4 + this.shards.length * this.rows * 4;
  }

  destroy() {
    for (const s of this.shards) { s.qs.destroy(); s.scales.destroy(); s.dims.destroy(); }
    this.xBuf.destroy(); this.outBuf.destroy();
    this.partsBuf?.destroy(); this.reduceDims?.destroy();
  }
}

// ------------------------------------------------------------ the sharded head
//
// The concrete motivating case: an LM head of [vocab, hidden] where the whole
// tensor does not fit one binding. Row-wise, so the shards' logits CONCATENATE
// and the argmax is the max of per-shard winners.

export interface ShardedHeadOpts extends ShardedMatvecOpts {
  /** How many shards. */
  shards: number;
}

/**
 * A row-wise sharded LM head: hidden state in, token id out.
 *
 * Argmax runs per shard and the winners are reduced on the HOST, over N pairs.
 * That is not a shortcut -- it is what a distributed head has to do, because the
 * shards are on different devices and nothing can reduce across them on-GPU. The
 * per-shard reduction is still argmax.wgsl unchanged, so the tie rule (lowest
 * index wins) is the tested one; reduceShardArgmax in shard_ref.ts extends it
 * across shards, and the extension is sound only because shard s owns a
 * CONTIGUOUS row range with s increasing. An interleaved layout would break it.
 *
 * What crosses the host boundary per token is N * 8 bytes, against 608 KB if the
 * shards sent logits back. The whole vocab never leaves device memory.
 */
export class ShardedHead {
  readonly vocab: number;
  readonly hidden: number;
  readonly nShards: number;
  private dev: GPUDevice;
  private mv: ShardedMatvec;
  private am1: GPUComputePipeline;
  private am2: GPUComputePipeline;
  private perShard: {
    range: ShardRange;
    groups: number;
    pVal: GPUBuffer; pIdx: GPUBuffer;
    oVal: GPUBuffer; oIdx: GPUBuffer;
    d1: GPUBuffer; d2: GPUBuffer;
    dummy: GPUBuffer;
    /** Argmax stage-1 and stage-2 bind groups, built once. */
    bg1: GPUBindGroup; bg2: GPUBindGroup;
    /** A view of the sharded matvec's output covering just this shard's rows. */
    logitsOffset: number;
  }[] = [];
  private logitsSlice: GPUBuffer[] = [];

  private constructor(dev: GPUDevice, vocab: number, hidden: number, mv: ShardedMatvec,
                      am1: GPUComputePipeline, am2: GPUComputePipeline) {
    this.dev = dev; this.vocab = vocab; this.hidden = hidden; this.mv = mv;
    this.am1 = am1; this.am2 = am2;
    this.nShards = mv.ranges.length;
  }

  static async create(
    dev: GPUDevice, split: SplitQ8, vocab: number, hidden: number,
    opts: ShardedHeadOpts,
  ): Promise<ShardedHead> {
    // Row-wise is not a parameter. See the header: the direction is the finding,
    // not a knob, and a column-wise head would need a 608 KB all-reduce per token
    // plus a full logits buffer on every shard -- which defeats the purpose,
    // since the logits are `vocab` long on EVERY column shard.
    const mv = await ShardedMatvec.create(
      dev, split, vocab, hidden, opts.shards, "row", opts);
    const src = await wgslSource("argmax.wgsl");
    const mk = (entryPoint: string) => dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: src }), entryPoint },
    });
    const H = new ShardedHead(dev, vocab, hidden, mv, mk("pass1"), mk("pass2"));

    const rw = (n: number) => dev.createBuffer({
      size: Math.max(16, n * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    for (const rg of mv.ranges) {
      // Stage-1 groups sized so each thread handles ~1 element, the same sizing
      // model.ts uses (594 groups for 151936). Capped so stage 2's single
      // workgroup can still reduce the partials with a short strided loop.
      const groups = Math.max(1, Math.min(1024, Math.ceil(rg.count / 256)));
      const pVal = rw(groups), pIdx = rw(groups);
      const oVal = rw(1), oIdx = rw(1);
      const d1 = uniformBuffer(dev, [rg.count, groups, 0, 0]);
      const d2 = uniformBuffer(dev, [groups, groups, 0, 0]);
      const dummy = rw(1);
      // Each shard's argmax reads its own rows of the logits. The shards' logits
      // are contiguous in one buffer (row-wise concatenates), but a bind group
      // cannot express "this range of that buffer" without dynamic offsets, and a
      // dynamic offset has a 256-byte alignment requirement the row boundaries do
      // not respect. So each shard's logit slice is its OWN buffer, which is also
      // what a real distributed shard has: it never holds the other shards' logits.
      const slice = rw(rg.count);
      H.logitsSlice.push(slice);
      const bg = (pipe: GPUComputePipeline, bufs: GPUBuffer[]) => dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      H.perShard.push({
        range: rg, groups, pVal, pIdx, oVal, oIdx, d1, d2, dummy,
        // Built once here rather than per encode(). `dummy` is the buffer
        // argmax.wgsl's pass1 binds for `xidx`, which it declares (so both passes
        // share one auto layout) but reads nothing live from -- it cannot be the
        // buffer pass1 also writes, because read + read_write on one buffer is a
        // validation error.
        bg1: bg(H.am1, [slice, dummy, pVal, pIdx, d1]),
        bg2: bg(H.am2, [pVal, pIdx, oVal, oIdx, d2]),
        logitsOffset: rg.start,
      });
    }
    return H;
  }

  /** Upload the (already normed) hidden state. */
  setInput(h: Float32Array) { this.mv.setInput(h); }

  /**
   * Encode the whole head: N shard matvecs, then per-shard argmax stages.
   *
   * The matvec writes into one buffer that the shards' row ranges partition, and
   * each shard's slice is then copied into its own buffer for the argmax. That
   * copy is an artifact of running N shards on ONE device -- a real shard's matvec
   * writes straight into its own local buffer -- so bench_shard.ts reports it
   * separately rather than folding it into the head's cost.
   */
  encode(enc: GPUCommandEncoder) {
    this.mv.encode(enc);
    const logits = this.mv.output();
    for (let i = 0; i < this.perShard.length; i++) {
      const s = this.perShard[i];
      enc.copyBufferToBuffer(
        logits, s.logitsOffset * 4, this.logitsSlice[i], 0, s.range.count * 4);
    }
    // Bind groups are built once (see `bgs` below), not per encode. Rebuilding
    // them per step cost ~4 ms of a 5.4 ms layer in the layer benchmark, and a
    // sharded head builds 2N of them.
    for (let i = 0; i < this.perShard.length; i++) {
      const s = this.perShard[i];
      const p = enc.beginComputePass();
      p.setPipeline(this.am1);
      p.setBindGroup(0, s.bg1);
      p.dispatchWorkgroups(s.groups);
      p.end();
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.am2);
      p2.setBindGroup(0, s.bg2);
      p2.dispatchWorkgroups(1);
      p2.end();
    }
  }

  /** The concatenated logits buffer, `vocab` long. For tests. */
  logits(): GPUBuffer { return this.mv.output(); }

  /** Per-shard (value, shard-local index, row offset), ready for reduceShardArgmax. */
  argmaxBuffers(): { oVal: GPUBuffer; oIdx: GPUBuffer; rowOffset: number }[] {
    return this.perShard.map((s) => ({
      oVal: s.oVal, oIdx: s.oIdx, rowOffset: s.range.start,
    }));
  }

  /** Bytes each shard has resident. The whole point of sharding. */
  shardBytes(): number[] {
    return this.mv.ranges.map((rg) => rg.count * this.hidden + rg.count * (this.hidden / 32) * 2);
  }

  destroy() {
    this.mv.destroy();
    for (const s of this.perShard) {
      s.pVal.destroy(); s.pIdx.destroy(); s.oVal.destroy(); s.oIdx.destroy();
      s.d1.destroy(); s.d2.destroy(); s.dummy.destroy();
    }
    for (const b of this.logitsSlice) b.destroy();
  }
}
