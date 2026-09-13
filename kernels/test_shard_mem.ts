// Memory accounting: each shard's buffers stay under a configurable limit.
//
//   deno run --unstable-webgpu --allow-all kernels/test_shard_mem.ts
//
// THIS IS THE DELIVERABLE TENSOR PARALLELISM EXISTS FOR, so it is asserted as
// arithmetic and then as an actual upload, not one or the other.
//
// The numbers that motivate it, all measured rather than assumed:
//
//   Qwen3-14B Q4_K_M `output.weight`   638 MB, ONE tensor, tie_word_embeddings
//                                      false so it cannot be shared with the
//                                      embedding table
//   Qwen3-0.6B `token_embd.weight`     155.6 MB of qs once repacked into the split
//                                      layout the kernels read (measured: the test
//                                      below reads the real file)
//   WebGPU default binding limit       128 MiB = 134217728 bytes
//   this Mac's adapter                 4295 MB, reported at the end -- which is why
//                                      the 0.6B model works here at all and why the
//                                      limit has to be a PARAMETER rather than read
//                                      off the device
//
// WHY ARITHMETIC AND NOT JUST "TRY IT". Exceeding maxStorageBufferBindingSize is a
// VALIDATION error, not an exception at the call site (correctness trap 3): the
// bind group is rejected and the kernel silently writes zeros. So "did it fit" is
// not something a device reliably tells you by failing. The check has to be made
// before the upload, from the shape -- which is what memShardPlan does and what
// ShardedMatvec.create calls.
//
// WHY THE LIMIT APPLIES PER BINDING, NOT PER SHARD. qs and scales are separate
// storage bindings. A shard holding 130 MB of qs and 9 MB of scales fails a 128 MiB
// limit on the qs binding alone, even though neither number alone is the shard's
// footprint. maxBindingBytes is therefore the figure compared against the limit and
// totalBytes is reported separately, because the two bound different things: one is
// a binding limit, the other is device memory.

import { getDevice, limitOf, ok, randVec, readBack, splitQ8, summary } from "./lib.ts";
import { memShardPlan, type TensorSpec } from "./shard_ref.ts";
import { ShardedMatvec } from "./shard.ts";
import { realModel } from "./real_weights.ts";

const MIB = 1024 * 1024;
const DEFAULT_LIMIT = 128 * MIB;
const dev = await getDevice();

const mb = (b: number) => (b / 1e6).toFixed(1) + " MB";

// -------------------------------------------------- the 638 MB tensor, split 8 ways
//
// The headline case from the brief: Qwen3-14B Q4_K_M's output.weight. Its shape is
// [vocab 151936, hidden 5120]; Q4_K_M stores that tensor at ~4.5 bits per weight,
// which is where 638 MB comes from. Modelled here as bytesPerWeight rather than by
// implementing Q4_K, because the memory arithmetic does not depend on the
// quantization's internals -- only on bytes per weight and bytes of scale per
// block, both of which are parameters.
console.log("-- Qwen3-14B Q4_K_M output.weight: 638 MB as ONE tensor\n");
{
  // 638 MB over 151936 x 5120 = 777,912,320 weights is 0.820 bytes/weight, i.e.
  // 6.56 bits -- which is Q4_K_M's blended figure for this tensor (Q4_K_M promotes
  // output.weight to Q6_K, and Q6_K is 6.5625 bits/weight). Using the measured
  // total rather than a nominal 4.5 bits is the point: the tensor is 638 MB
  // because that is what the file says, and the split has to work on that number.
  const Q6K_BYTES_PER_WEIGHT = 210 / 256; // Q6_K: 210 bytes per 256 weights
  const t: TensorSpec = {
    name: "output.weight (Qwen3-14B Q4_K_M)",
    rows: 151936,
    cols: 5120,
    bytesPerWeight: Q6K_BYTES_PER_WEIGHT,
    // Q6_K carries its scales inside the 210 bytes, so there is no separate scales
    // binding to account for. Set to 0 rather than left at the Q8_0 default, which
    // would overstate the footprint.
    scaleBytesPerBlock: 0,
  };
  const whole = memShardPlan(t, 1, "row", DEFAULT_LIMIT);
  console.log(`  unsharded largest binding: ${mb(whole.unshardedMaxBindingBytes)}`);
  ok(
    "the tensor really is ~638 MB",
    Math.abs(whole.unshardedMaxBindingBytes / 1e6 - 638) < 5,
    `${mb(whole.unshardedMaxBindingBytes)}`,
  );
  ok(
    "it does NOT fit a 128 MiB binding limit unsharded",
    !whole.fits,
    `${mb(whole.maxBindingBytes)} > ${mb(DEFAULT_LIMIT)}`,
  );

  // Split 8 ways: the brief's specific ask.
  const p8 = memShardPlan(t, 8, "row", DEFAULT_LIMIT);
  for (const s of p8.shards) {
    console.log(`    shard ${s.index}: ${s.rows} rows, largest binding ${mb(s.maxBindingBytes)}`);
  }
  ok(
    "split 8 ways, every shard's largest binding is under 128 MiB",
    p8.fits,
    `max ${mb(p8.maxBindingBytes)} < ${mb(DEFAULT_LIMIT)}`,
  );
  ok(
    "8 shards is not gratuitous: fewer would also fit, and the plan says how few",
    p8.minShardsForLimit !== null && p8.minShardsForLimit <= 8,
    `minimum ${p8.minShardsForLimit} shards`,
  );

  // Column-wise gives the same capacity relief, which is worth asserting because
  // it is the reason the direction choice is NOT a capacity argument -- both
  // directions divide rows*cols by N. The choice rests on bit-exactness, on
  // contiguity, and on what crosses the wire. See shard_ref.ts's header.
  const c8 = memShardPlan(t, 8, "col", DEFAULT_LIMIT);
  ok(
    "column-wise 8 shards fits the same limit (capacity does not prefer a direction)",
    c8.fits && Math.abs(c8.maxBindingBytes - p8.maxBindingBytes) / p8.maxBindingBytes < 0.01,
    `row ${mb(p8.maxBindingBytes)} vs col ${mb(c8.maxBindingBytes)}`,
  );
}

// ------------------------------------ a sweep: how many shards does a limit need
//
// Across the tensors that actually matter and a range of limits a real device might
// impose -- 128 MiB is the WebGPU default, 256 MiB and 1 GiB are what some adapters
// grant, and 64 MiB is a conservative phone.
console.log("\n-- how many shards each limit needs\n");
{
  const tensors: TensorSpec[] = [
    { name: "Qwen3-0.6B token_embd (Q8_0, tied)", rows: 151936, cols: 1024 },
    {
      name: "Qwen3-14B output.weight (Q6_K)",
      rows: 151936,
      cols: 5120,
      bytesPerWeight: 210 / 256,
      scaleBytesPerBlock: 0,
    },
    { name: "a 248320-row head (Q8_0)", rows: 248320, cols: 5120 },
  ];
  const limits = [64 * MIB, 128 * MIB, 256 * MIB, 1024 * MIB];
  let bad = 0;
  console.log(
    `  ${"tensor".padEnd(38)} ${"unsharded".padStart(10)}   ` +
      limits.map((l) => `${(l / MIB)}MiB`.padStart(8)).join(" "),
  );
  for (const t of tensors) {
    const cells: string[] = [];
    for (const lim of limits) {
      const p = memShardPlan(t, 1, "row", lim);
      const n = p.minShardsForLimit;
      cells.push(n === null ? "never" : `${n}`);
      if (n !== null) {
        // The claimed minimum must actually fit, and one fewer must not. This is
        // what makes it a MINIMUM rather than a number that happens to work.
        const at = memShardPlan(t, n, "row", lim);
        if (!at.fits) bad++;
        if (n > 1 && memShardPlan(t, n - 1, "row", lim).fits) bad++;
      }
    }
    const p1 = memShardPlan(t, 1, "row", DEFAULT_LIMIT);
    console.log(
      `  ${t.name.padEnd(38)} ${mb(p1.unshardedMaxBindingBytes).padStart(10)}   ` +
        cells.map((c) => c.padStart(8)).join(" "),
    );
  }
  ok(
    "every reported minimum shard count fits, and one fewer does not",
    bad === 0,
    `${bad} defects over ${tensors.length} tensors x ${limits.length} limits`,
  );
}

// ------------------------------------ the real tensor, actually uploaded and run
//
// The arithmetic above is a model. This is the same claim made by uploading the
// real 155.6 MB tensor as 8 shards under a simulated 32 MiB limit and checking the
// answer is still bit-identical -- because a plan that fits and computes garbage
// would be worse than one that refuses.
console.log("\n-- the real 155.6 MB tied head, uploaded as shards under a simulated limit\n");
{
  const m = await realModel();
  const split = splitQ8(m.embd.packed, m.embd.rows, m.embd.cols);
  const qsBytes = split.qs.byteLength, scBytes = split.scales.byteLength;
  console.log(`  token_embd.weight repacked: qs ${mb(qsBytes)}, scales ${mb(scBytes)}`);
  ok(
    "the repacked qs really is over the 128 MiB default limit",
    qsBytes > DEFAULT_LIMIT,
    `${mb(qsBytes)} > ${mb(DEFAULT_LIMIT)}`,
  );

  // The model's own arithmetic must agree with the bytes actually produced. If
  // memShardPlan and splitQ8 disagree, every capacity claim above is unfounded.
  const planned = memShardPlan(
    { name: "token_embd", rows: m.embd.rows, cols: m.embd.cols },
    1,
    "row",
    DEFAULT_LIMIT,
  );
  ok(
    "memShardPlan's unsharded figure matches the bytes splitQ8 actually produced",
    planned.unshardedMaxBindingBytes === qsBytes,
    `planned ${planned.unshardedMaxBindingBytes}, actual ${qsBytes}`,
  );

  const x = randVec(m.embd.cols, 0.1);
  // 32 MiB: a limit this tensor needs 5 shards to satisfy, so it exercises the
  // "more shards than the obvious power of two" path rather than a tidy split.
  const LIMIT = 32 * MIB;
  const need = memShardPlan(
    { name: "token_embd", rows: m.embd.rows, cols: m.embd.cols },
    1,
    "row",
    LIMIT,
  ).minShardsForLimit!;
  console.log(`  a ${mb(LIMIT)} limit needs ${need} shards`);

  // The reference: the same tensor unsharded, through the same kernel, with the
  // device's real limit. Sharding must not change the answer.
  const ref = await ShardedMatvec.create(dev, split, m.embd.rows, m.embd.cols, 1, "row");
  ref.setInput(x);
  let enc = dev.createCommandEncoder();
  ref.encode(enc);
  dev.queue.submit([enc.finish()]);
  const refOut = await readBack(dev, ref.output(), m.embd.rows * 4);
  ref.destroy();

  for (const n of [need, 8]) {
    const mv = await ShardedMatvec.create(
      dev,
      split,
      m.embd.rows,
      m.embd.cols,
      n,
      "row",
      { bindingLimit: LIMIT },
    );
    const plan = memShardPlan(
      { name: "token_embd", rows: m.embd.rows, cols: m.embd.cols },
      n,
      "row",
      LIMIT,
    );
    mv.setInput(x);
    enc = dev.createCommandEncoder();
    mv.encode(enc);
    dev.queue.submit([enc.finish()]);
    const out = await readBack(dev, mv.output(), m.embd.rows * 4);
    let diff = 0;
    for (let i = 0; i < m.embd.rows; i++) diff = Math.max(diff, Math.abs(out[i] - refOut[i]));
    ok(
      `N=${n} under a ${mb(LIMIT)} limit: largest binding ${mb(plan.maxBindingBytes)}, ` +
        `answer bit-identical`,
      diff === 0 && plan.fits,
      `max |diff| ${diff.toExponential(1)}, largest binding ${plan.maxBindingBytes} bytes`,
    );
    // Every shard's uploaded bytes must match the plan, not just the plan's own
    // arithmetic. This is the check that catches a slicer handing a shard more
    // bytes than its range -- which would fit the plan on paper and not in fact.
    ok(
      `N=${n}: uploaded bytes equal the whole tensor, no shard double-counted`,
      mv.uploadedBytes === qsBytes + scBytes,
      `uploaded ${mb(mv.uploadedBytes)}, tensor ${mb(qsBytes + scBytes)}`,
    );
    mv.destroy();
  }
}

// ---------------------------------------------- what the wire costs, per direction
//
// Not a memory figure, but it belongs next to one: the reason row-wise was chosen
// for the head is that column-wise moves the OUTPUT between devices and the output
// of an LM head is the whole vocabulary.
console.log("\n-- per-call wire traffic, the two directions\n");
{
  const cases: [string, number, number][] = [
    ["LM head 151936x1024", 151936, 1024],
    ["LM head 248320x5120", 248320, 5120],
    ["ffn_down 1024x3072", 1024, 3072],
  ];
  // Computed from the shapes rather than by constructing a ShardedMatvec, so this
  // does not need 638 MB resident to state what 638 MB would cost to move. The
  // formula is the same one wireBytesPerCall implements, and test_shard.ts's
  // constructions are where the class itself is exercised.
  let rowWinsHead = 0;
  for (const [name, rows, cols] of cases) {
    const n = 4;
    // Row-wise: x broadcast to each of N shards, each shard's output slice back.
    const rowBytes = n * cols * 4 + rows * 4;
    // Column-wise: x's slices out (cols total), a FULL-LENGTH partial back from
    // every shard. The output of an LM head is the whole vocabulary, so this is
    // where column-wise loses by orders of magnitude.
    const colBytes = cols * 4 + n * rows * 4;
    console.log(
      `  ${name.padEnd(22)} N=4  row-wise ${(rowBytes / 1024).toFixed(0).padStart(6)} KB   ` +
        `col-wise ${(colBytes / 1024).toFixed(0).padStart(6)} KB   ` +
        `ratio ${(colBytes / rowBytes).toFixed(1)}x`,
    );
    if (rows > cols && rowBytes < colBytes) rowWinsHead++;
  }
  ok(
    "row-wise moves strictly less per call whenever rows > cols (every head shape)",
    rowWinsHead === 2,
    `${rowWinsHead} of 2 head shapes`,
  );
}

console.log(
  `\nthis adapter's maxStorageBufferBindingSize: ` +
    `${mb(limitOf(dev.limits, "maxStorageBufferBindingSize"))} ` +
    `(the WebGPU default is ${mb(DEFAULT_LIMIT)})`,
);

Deno.exit(summary() ? 1 : 0);
