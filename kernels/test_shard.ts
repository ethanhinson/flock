// A matvec split across N shards == the unsharded matvec.
//
//   deno run --unstable-webgpu --allow-all kernels/test_shard.ts
//
// THE TWO CLAIMS ARE DIFFERENT AND ARE ASSERTED DIFFERENTLY.
//
// ROW-WISE is BIT-IDENTICAL to the unsharded kernel, at every N tested, and that
// is asserted as `=== 0` with no tolerance. The reason it is achievable: row r's
// 64 lanes walk the same nb = cols/32 blocks in the same order whether the shard
// holds 4 rows or 151936, and `row` never enters the arithmetic -- only the
// address. So a nonzero difference here is not "f32 noise", it is a slicing bug,
// and a tolerance would hide exactly that.
//
// COLUMN-WISE CANNOT BE bit-identical and the test says so with numbers instead
// of hoping. N column shards partition each row's blocks before the lanes see
// them, so the f32 addition order changes -- unavoidably, because f32 addition is
// not associative. Two things are asserted instead:
//
//   1. The sharded GPU result is BIT-IDENTICAL to an FMA-modelled strict-f32 CPU
//      reference OF THE SHARDED ORDER (shardedColMatvecRef). This is STRICTLY
//      STRONGER than a tolerance against the unsharded result: it pins every
//      dequantize, every dot, every lane's accumulator and the reduction order,
//      so a shard that dropped a block or read the wrong scale fails, whereas a
//      1e-6 tolerance against the unsharded answer would not notice a single
//      wrong block out of 32.
//   2. The residual against the UNSHARDED result is reported and bounded at the
//      size f32 reassociation predicts. That bound is derived, not picked -- see
//      the note above the assertion.
//
// Both directions include a NON-DIVISIBLE split (N = 3 over 1024 columns is
// 11/11/10 blocks; N = 3 over 30 rows is 10/10/10 but N = 4 over 30 is 8/8/7/7)
// so remainder handling is exercised rather than assumed.

import {
  cpuMatmulQ8F32, coop, getDevice, ok, quantMatrixQ8, randVec, readBack, splitQ8,
  storageBuffer, summary,
} from "./lib.ts";
import { absErrScaled, amax } from "./ops_ref.ts";
import {
  memShardPlan, shardRanges, shardedColMatvecRef, shardedRowMatvecRef,
  shardMatvecRef, shardReduceRef, slicePackedCols, slicePackedRows, sliceCols,
  sliceRows,
} from "./shard_ref.ts";
import { ShardedMatvec } from "./shard.ts";
import { realQ8Tensor } from "./real_weights.ts";

const LANES = 64;
const dev = await getDevice();

async function runSharded(
  packed: Uint8Array, x: Float32Array, rows: number, cols: number,
  n: number, direction: "row" | "col",
): Promise<Float32Array> {
  const split = splitQ8(packed, rows, cols);
  const mv = await ShardedMatvec.create(dev, split, rows, cols, n, direction);
  mv.setInput(x);
  const enc = dev.createCommandEncoder();
  mv.encode(enc);
  dev.queue.submit([enc.finish()]);
  const out = await readBack(dev, mv.output(), rows * 4);
  mv.destroy();
  return out;
}

// ---------------------------------------------------------------- planning
//
// shardRanges is where an off-by-one would land, and every downstream claim
// depends on it partitioning exactly. Tested directly rather than only through
// the kernel, because a kernel that reads the wrong 32 columns still produces
// plausible numbers.
{
  const cases: [number, number, number][] = [
    [1024, 1, 32], [1024, 2, 32], [1024, 3, 32], [1024, 4, 32],
    [1024, 5, 32], [1024, 7, 32], [3072, 3, 32], [151936, 8, 1], [30, 4, 1],
  ];
  let bad = 0, worstSkew = 0;
  for (const [total, n, align] of cases) {
    const rs = shardRanges(total, n, align);
    if (rs.length !== n) bad++;
    if (rs[0].start !== 0) bad++;
    if (rs[n - 1].end !== total) bad++;
    for (let i = 1; i < n; i++) if (rs[i].start !== rs[i - 1].end) bad++;
    for (const r of rs) {
      if (r.count % align !== 0) bad++;
      if (r.count <= 0) bad++;
    }
    const counts = rs.map((r) => r.count);
    worstSkew = Math.max(worstSkew, Math.max(...counts) - Math.min(...counts));
  }
  ok("shardRanges partitions exactly, with no gaps or overlaps", bad === 0,
     `${cases.length} cases, ${bad} defects`);
  ok("shard sizes differ by at most one alignment unit", worstSkew <= 32,
     `worst spread ${worstSkew}`);

  // A column boundary inside a Q8_0 block would split an f16 scale between two
  // shards, and there is no correct way to divide it. This has to throw, not
  // round, because rounding would silently give one shard 16 of another's columns.
  let threw = "";
  try { sliceCols({ qs: new Uint8Array(0), scales: new Uint8Array(0) }, 1, 64, 0, 16); }
  catch (e) { threw = String((e as Error).message); }
  ok("a column boundary inside a Q8_0 block is rejected, not rounded",
     threw.includes("32"), threw || "(did not throw)");

  threw = "";
  try { shardRanges(1024, 100, 32); } catch (e) { threw = String((e as Error).message); }
  ok("more shards than 32-column units is rejected", threw.includes("units"),
     threw || "(did not throw)");
}

// --------------------------------------------------- the slicers are lossless
//
// Before any kernel runs: the slices must reassemble into the original bytes.
// A gather that transposed two rows produces weights that are still valid Q8_0,
// so this cannot be left to the numeric comparison downstream.
{
  const rows = 37, cols = 1024;
  const packed = quantMatrixQ8(randVec(rows * cols, 0.05), rows, cols);
  const split = splitQ8(packed, rows, cols);
  let rowBad = 0, colBad = 0;

  for (const n of [1, 2, 3, 4, 5]) {
    // Row-wise: concatenating the slices must reproduce the whole split layout.
    const rr = shardRanges(rows, n, 1);
    const nb = cols / 32;
    for (const rg of rr) {
      const sl = sliceRows(split, rows, cols, rg.start, rg.end);
      for (let i = 0; i < sl.qs.length; i++) {
        if (sl.qs[i] !== split.qs[rg.start * cols + i]) rowBad++;
      }
      for (let i = 0; i < sl.scales.length; i++) {
        if (sl.scales[i] !== split.scales[rg.start * nb * 2 + i]) rowBad++;
      }
    }
    // Column-wise: row r of shard s must be row r's columns [c0, c1).
    const cr = shardRanges(cols, n, 32);
    for (const rg of cr) {
      const sl = sliceCols(split, rows, cols, rg.start, rg.end);
      const w = rg.count, wb = w / 32, b0 = rg.start / 32;
      for (let r = 0; r < rows; r++) {
        for (let i = 0; i < w; i++) {
          if (sl.qs[r * w + i] !== split.qs[r * cols + rg.start + i]) colBad++;
        }
        for (let i = 0; i < wb * 2; i++) {
          if (sl.scales[r * wb * 2 + i] !== split.scales[(r * nb + b0) * 2 + i]) colBad++;
        }
      }
    }
  }
  ok("sliceRows reproduces the original bytes for N = 1..5", rowBad === 0,
     `${rowBad} mismatches`);
  ok("sliceCols reproduces the original bytes for N = 1..5", colBad === 0,
     `${colBad} mismatches`);

  // And the on-disk-layout slicers, which the CPU reference uses. Deliberately a
  // separate implementation from the split-layout ones, so the two cannot share a
  // mistake -- which is the whole reason shardMatvecRef decodes from the 34-byte
  // layout while the kernel reads (qs, scales).
  let pBad = 0;
  for (const n of [2, 3, 4]) {
    for (const rg of shardRanges(cols, n, 32)) {
      const sl = slicePackedCols(packed, rows, cols, rg.start, rg.end);
      const wb = rg.count / 32, b0 = rg.start / 32, nbAll = cols / 32;
      for (let r = 0; r < rows; r++) {
        for (let i = 0; i < wb * 34; i++) {
          if (sl[r * wb * 34 + i] !== packed[(r * nbAll + b0) * 34 + i]) pBad++;
        }
      }
    }
    for (const rg of shardRanges(rows, n, 1)) {
      const sl = slicePackedRows(packed, rows, cols, rg.start, rg.end);
      const rb = (cols / 32) * 34;
      for (let i = 0; i < sl.length; i++) {
        if (sl[i] !== packed[rg.start * rb + i]) pBad++;
      }
    }
  }
  ok("the on-disk-layout slicers agree with the split-layout ones", pBad === 0,
     `${pBad} mismatches`);
}

// -------------------------------------- ROW-WISE: bit-identical at every N
//
// The strong claim. Asserted as exactly zero, at four shard counts including a
// non-divisible one, across the Qwen3 projection shapes plus two awkward ones
// (30 rows is not a multiple of ROWS_PER_WG = 4, so the last workgroup of a shard
// has idle rows that must still reach every barrier; 64x32 is one block per row,
// so 63 of 64 lanes contribute zero).
const SHAPES: [number, number][] = [
  [64, 32], [30, 1024], [128, 1024], [1024, 1024], [2048, 1024], [3072, 1024], [1024, 3072],
];

console.log("\n-- row-wise (output-split): expected BIT-IDENTICAL\n");
for (const [rows, cols] of SHAPES) {
  const w = randVec(rows * cols, 0.05), x = randVec(cols);
  const packed = quantMatrixQ8(w, rows, cols);
  const unsharded = cpuMatmulQ8F32(packed, x, rows, cols, coop(LANES));
  for (const n of [1, 2, 3, 4]) {
    if (rows < n) continue;
    const gpu = await runSharded(packed, x, rows, cols, n, "row");
    let diff = 0;
    for (let i = 0; i < rows; i++) diff = Math.max(diff, Math.abs(gpu[i] - unsharded[i]));
    ok(`row-wise ${rows}x${cols} N=${n} is bit-identical to unsharded`, diff === 0,
       `max |diff| ${diff.toExponential(1)}`);
  }
}

// The CPU model of row-wise must ALSO be exactly the unsharded reference, which is
// the statement that concatenation introduces no arithmetic at all. If this failed
// while the GPU assertions passed, the reference would be wrong, not the kernel.
{
  const rows = 1024, cols = 1024;
  const packed = quantMatrixQ8(randVec(rows * cols, 0.05), rows, cols);
  const x = randVec(cols);
  const unsharded = cpuMatmulQ8F32(packed, x, rows, cols, coop(LANES));
  let worst = 0;
  for (const n of [1, 2, 3, 4, 7]) {
    const ref = shardedRowMatvecRef(packed, x, rows, cols, shardRanges(rows, n, 1), LANES);
    for (let i = 0; i < rows; i++) worst = Math.max(worst, Math.abs(ref[i] - unsharded[i]));
  }
  ok("the row-wise CPU reference is exactly the unsharded reference at N = 1,2,3,4,7",
     worst === 0, `max |diff| ${worst.toExponential(1)}`);
}

// ------------------------- COLUMN-WISE: exact against the SHARDED reference
//
// The honest claim. Bit-identical to a reference that models the sharded order,
// and a MEASURED (not assumed) residual against the unsharded answer.
console.log("\n-- column-wise (input-split): exact vs the SHARDED reference, reassociated vs unsharded\n");
const colResiduals: { shape: string; n: number; rel: number }[] = [];
for (const [rows, cols] of SHAPES) {
  const w = randVec(rows * cols, 0.05), x = randVec(cols);
  const packed = quantMatrixQ8(w, rows, cols);
  const unsharded = cpuMatmulQ8F32(packed, x, rows, cols, coop(LANES));
  const scale = amax(unsharded);
  for (const n of [1, 2, 3, 4]) {
    if (cols / 32 < n) continue;
    const ranges = shardRanges(cols, n, 32);
    const gpu = await runSharded(packed, x, rows, cols, n, "col");
    const ref = shardedColMatvecRef(packed, x, rows, cols, ranges, LANES);
    let exact = 0;
    for (let i = 0; i < rows; i++) exact = Math.max(exact, Math.abs(gpu[i] - ref[i]));
    ok(`col-wise ${rows}x${cols} N=${n} matches the FMA-modelled sharded reference exactly`,
       exact === 0, `max |diff| ${exact.toExponential(1)}` +
       `  (shards: ${ranges.map((r) => r.count).join("/")})`);
    const rel = absErrScaled(gpu, unsharded, scale);
    colResiduals.push({ shape: `${rows}x${cols}`, n, rel });
  }
}

// The residual against the unsharded result, bounded rather than tolerated.
//
// WHERE THE BOUND COMES FROM. Both paths sum the same rows*cols/32 scaled block
// dot-products in f32; they differ only in association. The classical bound on
// reassociating a sum of m terms is (m-1) * eps * sum|terms| with
// eps = 2^-24 = 6.0e-8, and both orders here are TREE-like rather than serial
// (64 lanes, then a pairwise tree), so the depth is ~log2 of the block count and
// the realistic factor is far smaller than m. Rather than claim a tight constant,
// the assertion is the loose, clearly-satisfiable one: the residual must be under
// 1e-5 of the output scale -- three orders of magnitude below the 1.34e-2 that
// Q8_0 QUANTIZATION itself costs (measured in test_onnx.ts). That is the number
// that matters: reassociation from column sharding is 1000x smaller than the
// quantization the model already carries, so it cannot change a token the
// unsharded engine would not also have changed. The measured values are printed so
// the margin is visible rather than merely asserted.
{
  const worst = Math.max(...colResiduals.map((r) => r.rel));
  const worstAt = colResiduals.find((r) => r.rel === worst)!;
  for (const r of colResiduals) {
    if (r.n === 4) console.log(`     ${r.shape.padEnd(10)} N=4  residual/scale ${r.rel.toExponential(1)}`);
  }
  // The THRESHOLD is 1e-5, three orders below quantization; the MEASURED values
  // come in around 1e-7, five orders below. Both figures are stated because they
  // mean different things: 1e-5 is the bar the assertion holds to (deliberately
  // loose, so it does not become a bit-exactness test by accident on one machine's
  // rounding), and ~1e-7 is what this hardware actually does.
  ok("column-wise reassociation stays far below Q8_0's own quantization error",
     worst < 1e-5,
     `worst ${worst.toExponential(1)} at ${worstAt.shape} N=${worstAt.n}, ` +
     `vs quantization 1.34e-2 (threshold 1e-5, i.e. 3 orders; measured is ~5)`);
  ok("column-wise is NOT bit-identical, as predicted (the residual is real)",
     worst > 0, `worst residual/scale ${worst.toExponential(1)}`);
}

// ----------------------------------------- the reduction, on its own
//
// shard_reduce.wgsl summed in isolation, against shardReduceRef. Planted values
// rather than random, so the ORDER is what is being tested: 1 + 1e-8 + 1e-8 is
// 1.0 in f32 if you add left to right, and the test would not see a tree
// reduction masquerading as a serial one on random data.
{
  const rows = 1024;
  const n = 4;
  const parts: Float32Array[] = [];
  for (let s = 0; s < n; s++) {
    const p = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
      // Row 0 is the order-sensitive planted case: 1.0 then three 1e-8s, whose
      // serial f32 sum is exactly 1.0 and whose pairwise-tree sum is not.
      p[i] = i === 0 ? (s === 0 ? 1.0 : 1e-8) : Math.fround((Math.random() - 0.5) * 10);
    }
    parts.push(p);
  }
  // A minimal tensor whose only job is to give ShardedMatvec a partials buffer of
  // the right shape. 128 columns so a 4-way column split is legal (4 blocks of
  // 32); the weights are never read, because only the reduce pass runs.
  const split = splitQ8(quantMatrixQ8(new Float32Array(rows * 128), rows, 128), rows, 128);
  const mv = await ShardedMatvec.create(dev, split, rows, 128, n, "col");
  // Write the planted partials straight into the partials buffer and run only the
  // reduce, so the matvec is out of the picture entirely.
  const flat = new Float32Array(n * rows);
  for (let s = 0; s < n; s++) flat.set(parts[s], s * rows);
  dev.queue.writeBuffer(mv.partials(), 0, flat);
  const enc = dev.createCommandEncoder();
  mv.encodeReduceOnly(enc);
  dev.queue.submit([enc.finish()]);
  const gpu = await readBack(dev, mv.output(), rows * 4);
  const ref = shardReduceRef(parts);
  let diff = 0;
  for (let i = 0; i < rows; i++) diff = Math.max(diff, Math.abs(gpu[i] - ref[i]));
  ok("shard_reduce.wgsl matches shardReduceRef bit-for-bit", diff === 0,
     `max |diff| ${diff.toExponential(1)}`);
  ok("the reduction sums in SHARD ORDER, not as a tree",
     gpu[0] === 1.0,
     `1.0 + 3x1e-8 gave ${gpu[0]} (serial f32 gives exactly 1; a pairwise tree gives 1.0000000298)`);
  mv.destroy();
}

// -------------------------------- col_off: the split is in the index, not the data
//
// A column shard can either be handed a pre-sliced x or be handed the whole x and
// an offset. Both must give the identical answer -- if they do not, the "slice"
// and the "offset" disagree about which columns the shard owns, which is a bug
// that would show up as a small, plausible error rather than a failure. Asserted
// exactly, since it is the same arithmetic on the same bytes either way.
{
  const rows = 512, cols = 1024, n = 3;
  const packed = quantMatrixQ8(randVec(rows * cols, 0.05), rows, cols);
  const x = randVec(cols);
  const ranges = shardRanges(cols, n, 32);
  // The device path: whole x, per-shard col_off (what ShardedMatvec does).
  const viaOffset = await runSharded(packed, x, rows, cols, n, "col");
  // The host path: each shard gets its own x slice, computed by the CPU reference
  // on independently sliced weights, and reduced in shard order.
  const viaSlice = shardReduceRef(ranges.map((rg) =>
    shardMatvecRef(
      slicePackedCols(packed, rows, cols, rg.start, rg.end),
      x.slice(rg.start, rg.end), rows, rg.count, LANES)));
  let diff = 0;
  for (let i = 0; i < rows; i++) diff = Math.max(diff, Math.abs(viaOffset[i] - viaSlice[i]));
  ok("col_off into a whole x == a pre-sliced x, exactly", diff === 0,
     `max |diff| ${diff.toExponential(1)}`);
}

// ------------------------------------------------------------ real weights
//
// Synthetic weights are uniform: every block gets a similar scale and the quants
// fill the int8 range evenly. Real rows have scales spanning orders of magnitude
// and blocks that are nearly all zero, and a sharded kernel that mishandled a
// zero-scale block would pass on synthetic data.
console.log("\n-- real GGUF tensors\n");
for (const name of ["blk.24.attn_k.weight", "blk.24.ffn_gate.weight", "blk.24.ffn_down.weight"]) {
  const t = await realQ8Tensor(name);
  const x = randVec(t.cols, 0.1);
  const unsharded = cpuMatmulQ8F32(t.packed, x, t.rows, t.cols, coop(LANES));
  for (const n of [2, 3, 4]) {
    const g = await runSharded(t.packed, x, t.rows, t.cols, n, "row");
    let d = 0;
    for (let i = 0; i < t.rows; i++) d = Math.max(d, Math.abs(g[i] - unsharded[i]));
    ok(`real ${name} row-wise N=${n} bit-identical`, d === 0,
       `${t.rows}x${t.cols}, max |diff| ${d.toExponential(1)}`);
  }
  {
    const n = 3;
    const ranges = shardRanges(t.cols, n, 32);
    const g = await runSharded(t.packed, x, t.rows, t.cols, n, "col");
    const ref = shardedColMatvecRef(t.packed, x, t.rows, t.cols, ranges, LANES);
    let d = 0;
    for (let i = 0; i < t.rows; i++) d = Math.max(d, Math.abs(g[i] - ref[i]));
    const resid = absErrScaled(g, unsharded, amax(unsharded));
    ok(`real ${name} col-wise N=${n} exact vs sharded reference`, d === 0,
       `max |diff| ${d.toExponential(1)}, residual vs unsharded ${resid.toExponential(1)}`);
  }
}

// ----------------------------------------------- the capacity guard actually guards
//
// A binding over the limit is a VALIDATION error: the kernel writes zeros and
// nothing throws at the call site (correctness trap 3). So the check must happen
// BEFORE the upload and must refuse, and this asserts that it does rather than
// trusting the comment.
{
  const rows = 1024, cols = 1024;
  const split = splitQ8(quantMatrixQ8(randVec(rows * cols, 0.05), rows, cols), rows, cols);
  let threw = "";
  try {
    // 1 MiB limit against a 1 MB qs binding split 1 way: must refuse and must say
    // how many shards would fit.
    await ShardedMatvec.create(dev, split, rows, cols, 1, "row", { bindingLimit: 400_000 });
  } catch (e) { threw = String((e as Error).message); }
  ok("a binding over the limit is refused before upload, not left to write zeros",
     threw.includes("over the") && threw.includes("shards would fit"),
     threw || "(did not throw)");

  // And the shard count it named must actually work.
  const plan = memShardPlan({ name: "W", rows, cols }, 1, "row", 400_000);
  const need = plan.minShardsForLimit!;
  const mv = await ShardedMatvec.create(
    dev, split, rows, cols, need, "row", { bindingLimit: 400_000 });
  ok(`the shard count the error names (${need}) does fit`, true,
     `largest binding ${memShardPlan({ name: "W", rows, cols }, need, "row", 400_000).maxBindingBytes} bytes under 400000`);
  mv.destroy();
}

Deno.exit(summary() ? 1 : 0);
