// Tensor parallelism: how a single weight matrix is cut across N devices, the
// host-side slicing, and the strict-f32 CPU references for both directions.
//
// THE PROBLEM THIS EXISTS FOR, measured rather than hypothetical. Qwen3-14B
// Q4_K_M's `output.weight` is 638 MB as ONE tensor, and that model sets
// tie_word_embeddings: False, so it cannot be shared with the embedding table.
// WebGPU's DEFAULT maxStorageBufferBindingSize is 128 MiB. Splitting the model by
// LAYER RANGE -- which is what flock does today, and what test_split.ts validates
// -- cannot help: no number of devices makes one tensor smaller. Either that
// tensor is cut, or a device whose adapter caps bindings at the default cannot
// hold the LM head at all. memShardPlan() below turns that into an arithmetic
// check, and test_shard_mem.ts asserts it.
//
// THE TWO DIRECTIONS, and which one to use. y = W x with W = [rows, cols]:
//
//   ROW-WISE (output-split). Shard s owns rows [r0, r1). Needs the whole x;
//     produces y[r0:r1]; the outputs CONCATENATE. No reduction.
//     * BIT-IDENTICAL to the unsharded kernel. Row r is computed by the same 64
//       lanes over the same nb = cols/32 blocks in the same order whether the
//       shard has 4 rows or 151936; only `row` changes, and `row` does not enter
//       the arithmetic. sliceRows + the kernel proves this at N = 1..4 and at a
//       non-divisible split.
//     * The slice is a CONTIGUOUS byte range in both qs and scales, because both
//       are row-major. So a shard can be range-fetched straight out of a GGUF
//       with no gather, no repack and no host-side copy of the other shards.
//     * For an LM head it also makes the argmax trivially shardable: argmax over
//       a concatenation is the max of the per-shard (value, index) pairs, and the
//       existing two-stage argmax.wgsl already produces exactly that pair.
//
//   COLUMN-WISE (input-split). Shard s owns columns [c0, c1) of every row. Needs
//     only x[c0:c1]; produces a PARTIAL y over ALL rows; the partials must be
//     SUMMED.
//     * NOT bit-identical, and not fixable. N column shards partition each row's
//       blocks N ways before the lanes see them, so the f32 addition order
//       changes. What IS achievable -- and is what the tests assert -- is that
//       the GPU matches an FMA-modelled strict-f32 reference OF THE SHARDED ORDER
//       bit-for-bit, and that the residual against the unsharded result is the
//       size f32 reassociation predicts. See the note on shardedColErr below.
//     * The slice is STRIDED: every row contributes (c1-c0) of its cols bytes.
//       sliceCols has to gather, which is a full pass over the tensor per shard.
//     * It needs a reduction kernel and, across devices, an all-reduce of `rows`
//       floats per matvec. For an LM head that is 151936 floats = 608 KB per
//       token per shard. Row-wise moves `cols` = 1024 floats of x instead.
//
// So: ROW-WISE for the LM head. The reasoning is not aesthetic and the numbers
// are in bench_shard.ts and the report -- bit-identical instead of approximate,
// a contiguous slice instead of a gather, 4 KB of x broadcast instead of 608 KB
// of partials all-reduced, and an argmax that needs no extra kernel. Column-wise
// is implemented and validated anyway because it is the right split for a
// down-projection (ffn_down reads a 3072-wide input and writes 1024 rows, so
// row-wise leaves each shard reading all 3072 of x while column-wise cuts the
// input too), and because a claim that one direction is better is worth nothing
// without the other one built and measured.

import { fma32, halfToF32, Q8_BLOCK, Q8_BYTES, type SplitQ8, splitQ8 } from "./lib.ts";

const fr = Math.fround;

// ------------------------------------------------------------------ planning

export interface ShardRange {
  /** Which shard this is, 0..n-1. */
  index: number;
  /** First row (row-wise) or column (column-wise) this shard owns. */
  start: number;
  /** One past the last. */
  end: number;
  /** end - start. */
  count: number;
}

/**
 * Cut `total` items into `n` shards, as evenly as possible, on a multiple of
 * `align`.
 *
 * ALIGNMENT IS NOT OPTIONAL FOR COLUMNS. A Q8_0 block is 32 weights sharing one
 * f16 scale, so a column boundary inside a block would split a scale across two
 * shards and there is no correct way to divide it. Every column boundary is
 * therefore a multiple of 32, and a matrix whose `cols` is not a multiple of
 * 32 * n simply gets uneven shards -- which is the case worth testing, because
 * remainder handling is where an off-by-one lives.
 *
 * Rows need no alignment for correctness (each row is independent), but a
 * boundary that is a multiple of ROWS_PER_WG = 4 means no workgroup straddles
 * two shards and each shard's dispatch count is exact. It is NOT enforced: the
 * kernel's `row < n_rows` guard already handles a ragged last workgroup, and
 * test_shard.ts deliberately runs a row split with align = 1 and a prime shard
 * count so that path is exercised rather than assumed.
 *
 * The remainder is spread one item at a time over the FIRST shards rather than
 * dumped on the last, so the largest shard is `ceil` and not `total - (n-1)*floor`
 * -- the difference matters for the memory bound: with total = 248320 rows over
 * 7 shards, spreading gives a max of 35476 rows and dumping on the last gives
 * 35482. Small here, but the bound memShardPlan reports is the MAX shard, so the
 * balanced split is the one that makes the bound tight.
 */
export function shardRanges(total: number, n: number, align = 1): ShardRange[] {
  if (!Number.isInteger(n) || n < 1) throw new Error(`n shards must be >= 1, got ${n}`);
  if (!Number.isInteger(total) || total < 0) throw new Error(`total must be >= 0, got ${total}`);
  if (!Number.isInteger(align) || align < 1) throw new Error(`align must be >= 1, got ${align}`);
  if (total % align !== 0) {
    throw new Error(`total ${total} is not a multiple of align ${align}`);
  }
  const units = total / align;
  if (units < n) {
    throw new Error(
      `cannot cut ${total} into ${n} shards aligned to ${align}: only ${units} units`,
    );
  }
  const base = Math.floor(units / n), extra = units % n;
  const out: ShardRange[] = [];
  let at = 0;
  for (let i = 0; i < n; i++) {
    const count = (base + (i < extra ? 1 : 0)) * align;
    out.push({ index: i, start: at, end: at + count, count });
    at += count;
  }
  return out;
}

// ------------------------------------------------------------------- slicing

/**
 * Rows [start, end) of a Q8_0 tensor already in the split (qs, scales) layout.
 *
 * Both arrays are row-major, so this is two subarray views -- no copy, no
 * gather. That is the whole practical advantage of row-wise: a shard's weights
 * are a byte range, which means a remote shard can HTTP-range-fetch exactly its
 * own bytes and never see the rest of the tensor.
 *
 * Returned as subarrays deliberately. `storageBuffer` in lib.ts handles a view
 * correctly (it converts to a byte view and passes an explicit length), unlike
 * `queue.writeBuffer` on a raw TypedArray view -- see the writeView note in
 * layer.ts, correctness trap 2. A caller that hands one of these to
 * writeBuffer directly will silently upload the whole tensor from 0.
 */
export function sliceRows(
  t: SplitQ8,
  rows: number,
  cols: number,
  start: number,
  end: number,
): SplitQ8 {
  if (start < 0 || end > rows || start > end) {
    throw new Error(`row range ${start}..${end} is not inside 0..${rows}`);
  }
  const nb = cols / Q8_BLOCK;
  return {
    qs: t.qs.subarray(start * cols, end * cols),
    scales: t.scales.subarray(start * nb * 2, end * nb * 2),
  };
}

/**
 * Columns [start, end) of every row of a Q8_0 tensor in the split layout.
 *
 * Unavoidably a gather: row r's window sits at r * cols + start, so the output
 * is (end-start) bytes per row out of cols, for every row. That is a full read
 * of the tensor per shard, i.e. N passes over the whole thing to produce N
 * shards -- which is a one-time load cost, but it is a cost row-wise does not
 * have, and it means a column-wise shard cannot be range-fetched.
 *
 * `start` and `end` must be multiples of 32 so no Q8_0 block is split across
 * shards; shardRanges(cols, n, 32) is how to get them.
 */
export function sliceCols(
  t: SplitQ8,
  rows: number,
  cols: number,
  start: number,
  end: number,
): SplitQ8 {
  if (start % Q8_BLOCK !== 0 || end % Q8_BLOCK !== 0) {
    throw new Error(
      `column range ${start}..${end} must be multiples of ${Q8_BLOCK}: a Q8_0 block's scale cannot be split`,
    );
  }
  if (start < 0 || end > cols || start > end) {
    throw new Error(`column range ${start}..${end} is not inside 0..${cols}`);
  }
  const nb = cols / Q8_BLOCK, w = end - start, wb = w / Q8_BLOCK;
  const b0 = start / Q8_BLOCK;
  const qs = new Uint8Array(rows * w);
  const scales = new Uint8Array(rows * wb * 2);
  for (let r = 0; r < rows; r++) {
    qs.set(t.qs.subarray(r * cols + start, r * cols + end), r * w);
    scales.set(
      t.scales.subarray((r * nb + b0) * 2, (r * nb + b0 + wb) * 2),
      r * wb * 2,
    );
  }
  return { qs, scales };
}

/** The on-disk 34-byte Q8_0 layout, columns [start, end) of every row. */
export function slicePackedCols(
  packed: Uint8Array,
  rows: number,
  cols: number,
  start: number,
  end: number,
): Uint8Array {
  if (start % Q8_BLOCK !== 0 || end % Q8_BLOCK !== 0) {
    throw new Error(`column range ${start}..${end} must be multiples of ${Q8_BLOCK}`);
  }
  const nb = cols / Q8_BLOCK, wb = (end - start) / Q8_BLOCK, b0 = start / Q8_BLOCK;
  const out = new Uint8Array(rows * wb * Q8_BYTES);
  for (let r = 0; r < rows; r++) {
    out.set(
      packed.subarray((r * nb + b0) * Q8_BYTES, (r * nb + b0 + wb) * Q8_BYTES),
      r * wb * Q8_BYTES,
    );
  }
  return out;
}

/** The on-disk layout, rows [start, end). Contiguous, like sliceRows. */
export function slicePackedRows(
  packed: Uint8Array,
  rows: number,
  cols: number,
  start: number,
  end: number,
): Uint8Array {
  if (start < 0 || end > rows || start > end) {
    throw new Error(`row range ${start}..${end} is not inside 0..${rows}`);
  }
  const rowBytes = (cols / Q8_BLOCK) * Q8_BYTES;
  return packed.subarray(start * rowBytes, end * rowBytes);
}

// ---------------------------------------------------------- CPU references
//
// Everything below is strict f32 with the GPU's ROUNDING COUNT modelled, not
// merely its arithmetic -- see the header of lib.ts. The kernel contracts every
// multiply-add into one FMA and its dot(vec4, vec4) is one exact product
// followed by three FMAs, so a reference that rounds each product separately
// does NOT match, and a reference in f64 is worse than either. These share the
// shape of lib.ts's coop reference on purpose; they are not a second model of
// the arithmetic, only a second model of the summation ORDER.

/** dot(vec4, vec4) as Metal emits it: one exact product, then three FMAs. */
function dot4(a: number[], b: number[]): number {
  let s = fr(a[0] * b[0]);
  s = fma32(a[1], b[1], s);
  s = fma32(a[2], b[2], s);
  s = fma32(a[3], b[3], s);
  return s;
}

/** Pairwise f32 tree sum: the order a workgroup reduction produces. */
function treeSum(a: Float32Array): number {
  const buf = Float32Array.from(a);
  for (let stride = buf.length >> 1; stride > 0; stride >>= 1) {
    for (let i = 0; i < stride; i++) buf[i] = fr(buf[i] + buf[i + stride]);
  }
  return buf[0];
}

/**
 * One shard's matvec, exactly as q8_shard.wgsl computes it: `lanes` lanes over
 * the shard's own nb blocks, each lane folding `scale * (dot + dot)` into its
 * accumulator four times per block, then a pairwise tree reduction.
 *
 * `packed` is the SHARD's weights in the 34-byte on-disk layout, shaped
 * [rows][cols] with `cols` being the shard's width. `x` is the shard's slice of
 * the input, `cols` long. So this function does not know or care which direction
 * the shard came from -- row-wise and column-wise differ only in what the caller
 * slices, which is the point and is why one reference covers both.
 *
 * Decoding from the 34-byte layout rather than from (qs, scales) is deliberate:
 * it means this reference re-derives the weights independently of the split the
 * kernel reads, so a slicing bug in sliceRows/sliceCols cannot hide by being
 * shared between kernel and reference. That is correctness trap "a consistency
 * test between a kernel and its reference cannot catch a shared assumption",
 * applied to the slicer.
 */
export function shardMatvecRef(
  packed: Uint8Array,
  x: Float32Array,
  rows: number,
  cols: number,
  lanes = 64,
): Float32Array {
  const nb = cols / Q8_BLOCK;
  const dv = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  const out = new Float32Array(rows);
  const part = new Float32Array(lanes);
  // The kernel's four word-pairs per block: (0..3, 4..7), (8..11, 12..15), ...
  const pairs = [0, 1, 2, 3].map((g) =>
    [[0, 1, 2, 3].map((k) => g * 8 + k), [0, 1, 2, 3].map((k) => g * 8 + 4 + k)] as [
      number[],
      number[],
    ]
  );
  const q = new Array<number>(32);
  for (let r = 0; r < rows; r++) {
    part.fill(0);
    for (let b = 0; b < nb; b++) {
      const base = (r * nb + b) * Q8_BYTES;
      const scale = fr(halfToF32(dv.getUint16(base, true)));
      for (let i = 0; i < 32; i++) {
        let v = packed[base + 2 + i];
        if (v > 127) v -= 256;
        q[i] = v;
      }
      const lane = b % lanes;
      for (const [ia, ib] of pairs) {
        const d0 = dot4(ia.map((i) => q[i]), ia.map((i) => x[b * 32 + i]));
        const d1 = dot4(ib.map((i) => q[i]), ib.map((i) => x[b * 32 + i]));
        part[lane] = fma32(scale, fr(d0 + d1), part[lane]);
      }
    }
    out[r] = treeSum(part);
  }
  return out;
}

/**
 * shard_reduce.wgsl's summation, in strict f32: shard 0, then 1, then 2, in
 * index order, one f32 add each.
 *
 * `parts[s]` is shard s's partial vector, all the same length. Serial and not a
 * tree, matching the kernel -- see the note in shard_reduce.wgsl for why the
 * order is a contract rather than an implementation detail.
 */
export function shardReduceRef(parts: Float32Array[]): Float32Array {
  if (!parts.length) throw new Error("nothing to reduce");
  const n = parts[0].length;
  for (const p of parts) {
    if (p.length !== n) throw new Error(`partials disagree on length: ${p.length} vs ${n}`);
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = parts[0][i];
    for (let s = 1; s < parts.length; s++) acc = fr(acc + parts[s][i]);
    out[i] = acc;
  }
  return out;
}

/**
 * The whole COLUMN-WISE sharded matvec on the CPU, in the exact order the GPU
 * path runs it: slice the columns, run each shard's coop reduction over its own
 * blocks, then sum the partials serially in shard order.
 *
 * This is what the sharded GPU result is asserted BIT-IDENTICAL to. The weaker
 * statement -- "close to the unsharded result" -- is reported alongside it as a
 * measured number, but it is not the assertion, because a tolerance against the
 * unsharded result cannot distinguish f32 reassociation from a slicing bug that
 * dropped a block.
 */
export function shardedColMatvecRef(
  packed: Uint8Array,
  x: Float32Array,
  rows: number,
  cols: number,
  ranges: ShardRange[],
  lanes = 64,
): Float32Array {
  const parts = ranges.map((rg) =>
    shardMatvecRef(
      slicePackedCols(packed, rows, cols, rg.start, rg.end),
      x.subarray(rg.start, rg.end) as Float32Array,
      rows,
      rg.count,
      lanes,
    )
  );
  return shardReduceRef(parts);
}

/**
 * The whole ROW-WISE sharded matvec on the CPU: concatenate the per-shard
 * outputs. No reduction, and no reassociation -- which is why this is expected
 * to equal the UNSHARDED reference exactly, not merely closely.
 */
export function shardedRowMatvecRef(
  packed: Uint8Array,
  x: Float32Array,
  rows: number,
  cols: number,
  ranges: ShardRange[],
  lanes = 64,
): Float32Array {
  const out = new Float32Array(rows);
  for (const rg of ranges) {
    const y = shardMatvecRef(
      slicePackedRows(packed, rows, cols, rg.start, rg.end),
      x,
      rg.count,
      cols,
      lanes,
    );
    out.set(y, rg.start);
  }
  return out;
}

// ------------------------------------------------------ argmax over a concat
//
// The LM head's output is 151936 (or 248320) logits and greedy decode needs one
// number out of them. Row-wise sharding makes that free, and this is the whole
// argument for the direction:
//
//   argmax(concat(y_0, ..., y_{n-1})) = the winning (value, index) pair among the
//   per-shard pairs, with the shard's row offset added back.
//
// argmax.wgsl's stage 1 ALREADY produces a (value, index) pair per workgroup and
// stage 2 reduces pairs, so a sharded argmax is stage 1 on each shard plus one
// stage-2 over the collected pairs -- no new kernel, and 8 bytes per shard on the
// wire instead of 608 KB of logits.
//
// TIE-BREAKING SURVIVES THE SPLIT, which is the part that needs care. The rule is
// "lowest index wins", and because shard s owns a contiguous row range starting
// at rowOffset_s with s increasing, comparing shards in index order with the same
// strictly-greater-or-equal-with-lower-index test gives the same answer as a
// single scan. A shard layout that interleaved rows would NOT have this property.

export interface ShardArgmax {
  /** Best value in this shard. */
  value: number;
  /** Its index WITHIN the shard. */
  index: number;
  /** The shard's first row in the full output. */
  rowOffset: number;
}

/**
 * Reduce per-shard argmax results to a global index, lowest index winning ties.
 *
 * Deliberately takes the shard-local index plus an offset rather than a global
 * index, because that is what a remote shard can actually report: it holds rows
 * [r0, r1) and knows nothing about the others. Adding r0 here rather than there
 * also means a shard cannot corrupt the tie rule by reporting a wrong offset.
 */
export function reduceShardArgmax(parts: ShardArgmax[]): number {
  if (!parts.length) throw new Error("nothing to reduce");
  let bi = -1, bv = -Infinity;
  for (const p of parts) {
    const gi = p.rowOffset + p.index;
    if (p.value > bv || (p.value === bv && gi < bi)) {
      bv = p.value;
      bi = gi;
    }
  }
  return bi;
}

// ------------------------------------------------------- memory accounting
//
// The reason tensor parallelism exists here. A device cannot bind a storage
// buffer larger than maxStorageBufferBindingSize, and exceeding it is a
// VALIDATION error -- a kernel that silently writes zeros, not a throw at the
// call site (correctness trap 3). So the check has to be arithmetic, done before
// any upload.

export interface TensorSpec {
  name: string;
  rows: number;
  cols: number;
  /** Bytes per weight in the quantization actually used. Q8_0 is 1. */
  bytesPerWeight?: number;
  /** Bytes of scale per 32-weight block. Q8_0 is 2 (one f16). */
  scaleBytesPerBlock?: number;
}

export interface ShardMemory {
  index: number;
  rows: number;
  cols: number;
  qsBytes: number;
  scalesBytes: number;
  /** The largest SINGLE binding, which is what the limit applies to. */
  maxBindingBytes: number;
  /** qs + scales, what the shard has to hold. */
  totalBytes: number;
}

export interface MemPlan {
  tensor: string;
  direction: "row" | "col";
  shards: ShardMemory[];
  /** Largest single binding over all shards. */
  maxBindingBytes: number;
  /** Same figure for n = 1, i.e. what the unsharded tensor needs. */
  unshardedMaxBindingBytes: number;
  /** Smallest n that fits under `limitBytes`, or null if no n does. */
  minShardsForLimit: number | null;
  limitBytes: number;
  fits: boolean;
}

/**
 * What each shard of `t` has to bind, split `n` ways in `direction`.
 *
 * The limit applies PER BINDING, not per shard total: qs and scales are separate
 * bindings, so a shard holding 130 MB of qs and 8 MB of scales fails a 128 MiB
 * limit on the qs binding alone. `maxBindingBytes` is therefore the figure to
 * compare, and `totalBytes` is reported separately because it is what bounds a
 * device's total memory rather than a single binding.
 *
 * Both directions shrink qs by the same factor -- the tensor has rows * cols
 * weights and n shards each hold rows*cols/n of them either way -- so the
 * capacity argument does not prefer one direction. What differs is everything
 * else: bit-exactness, whether the slice is contiguous, and what crosses the
 * wire. See the header.
 */
export function memShardPlan(
  t: TensorSpec,
  n: number,
  direction: "row" | "col",
  limitBytes = 128 * 1024 * 1024,
): MemPlan {
  const bpw = t.bytesPerWeight ?? 1;
  const sbb = t.scaleBytesPerBlock ?? 2;
  const per = (rows: number, cols: number): Omit<ShardMemory, "index"> => {
    const qsBytes = rows * cols * bpw;
    // Scales are per (row, 32-column block). A column-wise shard has fewer
    // blocks per row; a row-wise shard has fewer rows. Same product.
    const scalesBytes = rows * Math.ceil(cols / Q8_BLOCK) * sbb;
    return {
      rows,
      cols,
      qsBytes,
      scalesBytes,
      maxBindingBytes: Math.max(qsBytes, scalesBytes),
      totalBytes: qsBytes + scalesBytes,
    };
  };
  const ranges = direction === "row" ? shardRanges(t.rows, n, 1) : shardRanges(t.cols, n, Q8_BLOCK);
  const shards: ShardMemory[] = ranges.map((rg) => ({
    index: rg.index,
    ...(direction === "row" ? per(rg.count, t.cols) : per(t.rows, rg.count)),
  }));
  const maxBindingBytes = Math.max(...shards.map((s) => s.maxBindingBytes));
  const unsharded = per(t.rows, t.cols).maxBindingBytes;

  // The smallest n that fits. Searched rather than derived because the alignment
  // constraint makes the shard sizes non-monotone in the obvious closed form:
  // column shards are quantized to 32 columns, so ceil() can plateau.
  let minShardsForLimit: number | null = null;
  const maxN = direction === "row" ? t.rows : t.cols / Q8_BLOCK;
  for (let k = 1; k <= maxN; k++) {
    const p = direction === "row" ? shardRanges(t.rows, k, 1) : shardRanges(t.cols, k, Q8_BLOCK);
    const m = Math.max(
      ...p.map((rg) =>
        (direction === "row" ? per(rg.count, t.cols) : per(t.rows, rg.count)).maxBindingBytes
      ),
    );
    if (m <= limitBytes) {
      minShardsForLimit = k;
      break;
    }
  }

  return {
    tensor: t.name,
    direction,
    shards,
    maxBindingBytes,
    unshardedMaxBindingBytes: unsharded,
    minShardsForLimit,
    limitBytes,
    fits: maxBindingBytes <= limitBytes,
  };
}

/**
 * Repack a whole Q8_0 tensor once and hand back per-shard views.
 *
 * One splitQ8 of the full tensor, then N zero-copy subarrays for row-wise. For
 * column-wise it is one splitQ8 plus N gathers, and the asymmetry in the cost is
 * real and is what bench_shard.ts reports: 155.6 MB row-wise is N views, the
 * same tensor column-wise is N * 155.6 MB of copying.
 */
export function shardTensor(
  packed: Uint8Array,
  rows: number,
  cols: number,
  n: number,
  direction: "row" | "col",
): { ranges: ShardRange[]; shards: SplitQ8[]; split: SplitQ8 } {
  const split = splitQ8(packed, rows, cols);
  const ranges = direction === "row" ? shardRanges(rows, n, 1) : shardRanges(cols, n, Q8_BLOCK);
  const shards = ranges.map((rg) =>
    direction === "row"
      ? sliceRows(split, rows, cols, rg.start, rg.end)
      : sliceCols(split, rows, cols, rg.start, rg.end)
  );
  return { ranges, shards, split };
}
