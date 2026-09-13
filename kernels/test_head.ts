// Validate the three kernels that surround the layer stack, EACH ON ITS OWN,
// against a CPU reference -- before any of them is composed into a forward pass.
//
//   deno run --unstable-webgpu --allow-all kernels/test_head.ts
//
//   embed.wgsl              Q8_0 row gather + dequantize
//   argmax.wgsl             two-stage max-index reduction
//   attention_prefill.wgsl  N-query causal GQA with a streaming softmax
//
// Every case runs on REAL weights as well as synthetic ones where a real tensor
// exists, because token_embd is the tensor where a loading mistake is hardest to
// see: every row of it is a plausible embedding, so reading the wrong row does not
// look wrong -- it looks like a different token. The tests that pin that down are
// the identity ones (a row gathered twice must be identical; row r must differ
// from row r+1) plus the bit-exact diff against a decoder that reads the ORIGINAL
// 34-byte layout rather than the repacked one.

import {
  coopSource,
  getDevice,
  ok,
  probeUnpack,
  probeUnpackF16,
  quantMatrixQ8,
  randVec,
  readBack,
  splitQ8,
  storageBuffer,
  summary,
  uniformBuffer,
} from "./lib.ts";
import { absErrScaled, amax, attnRef, relErr } from "./ops_ref.ts";
import { argmaxRef, attnPrefillRef, embedRef } from "./head_ref.ts";
import { realQ8Tensor } from "./real_weights.ts";

const dev = await getDevice();
const unpack8 = await probeUnpack(dev);
const unpackF16 = await probeUnpackF16(dev);
const src = (f: string) => Deno.readTextFile(new URL("./" + f, import.meta.url));
const mk = (code: string, entryPoint = "main") =>
  dev.createComputePipeline({
    layout: "auto",
    compute: { module: dev.createShaderModule({ code }), entryPoint },
  });

const bind = (pipe: GPUComputePipeline, bufs: GPUBuffer[]) =>
  dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

function run(pipe: GPUComputePipeline, bufs: GPUBuffer[], groups: number | [number, number]) {
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(pipe);
  p.setBindGroup(0, bind(pipe, bufs));
  if (Array.isArray(groups)) p.dispatchWorkgroups(groups[0], groups[1]);
  else p.dispatchWorkgroups(groups);
  p.end();
  dev.queue.submit([enc.finish()]);
}

// ============================================================ embedding lookup
console.log("embed.wgsl -- Q8_0 row gather\n");

const embedPipe = mk(coopSource(await src("embed.wgsl"), { unpack8, unpackF16 }));

async function gpuEmbed(
  packed: Uint8Array,
  rows: number,
  cols: number,
  ids: number[],
): Promise<Float32Array> {
  const { qs, scales } = splitQ8(packed, rows, cols);
  const bufs = [
    storageBuffer(dev, qs),
    storageBuffer(dev, scales),
    storageBuffer(dev, new Uint8Array(new Uint32Array(ids).buffer)),
    dev.createBuffer({
      size: ids.length * cols * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }),
    uniformBuffer(dev, [ids.length, cols, 0, 0]),
  ];
  run(embedPipe, bufs, ids.length);
  const out = await readBack(dev, bufs[3], ids.length * cols * 4);
  for (const b of bufs) b.destroy();
  return out;
}

// Synthetic first: a small vocab where every row can be checked.
{
  const rows = 64, cols = 1024;
  const w = randVec(rows * cols, 0.05);
  const packed = quantMatrixQ8(w, rows, cols);
  for (const ids of [[0], [63], [7], [0, 1, 2], [63, 0, 31, 31, 5]]) {
    const got = await gpuEmbed(packed, rows, cols, ids);
    const want = embedRef(packed, ids, cols);
    ok(
      `embed ${rows}x${cols} ids=[${ids}]`,
      relErr(got, want, 1e-9) === 0,
      `rel err ${relErr(got, want, 1e-9).toExponential(1)}`,
    );
  }
  // A repeated id must produce identical rows -- the check that catches a gather
  // reading a stride instead of an index.
  const ids = [31, 31];
  const got = await gpuEmbed(packed, rows, cols, ids);
  let same = true;
  for (let i = 0; i < cols; i++) if (got[i] !== got[cols + i]) same = false;
  ok("embed of the same id twice is bit-identical", same);
  let differs = 0;
  const two = await gpuEmbed(packed, rows, cols, [17, 18]);
  for (let i = 0; i < cols; i++) if (two[i] !== two[cols + i]) differs++;
  ok(
    "adjacent rows differ (not reading one row twice)",
    differs > cols * 0.9,
    `${differs}/${cols} entries differ`,
  );
}

// The real token_embd. 151936 rows, so only a handful of ids are checked -- but
// they are checked against a decoder reading the ORIGINAL 34-byte layout, which
// makes this a test of splitQ8 on a 165 MB tensor as much as of the kernel.
const embd = await realQ8Tensor("token_embd.weight");
ok(
  "token_embd.weight is [151936 rows, 1024 cols] Q8_0",
  embd.rows === 151936 && embd.cols === 1024,
  `${embd.rows}x${embd.cols}, ${(embd.packed.byteLength / 1e6).toFixed(0)} MB`,
);
{
  // Include 0, the last row, and the special tokens the chat template uses, since
  // a bug at the top of the vocab (where the special tokens live, above 151643)
  // would only ever show up on a templated prompt.
  const ids = [0, 1, 151643, 151644, 151645, 151935, 9707, 3838];
  const got = await gpuEmbed(embd.packed, embd.rows, embd.cols, ids);
  const want = embedRef(embd.packed, ids, embd.cols);
  const err = relErr(got, want, 1e-9);
  ok(
    `real token_embd gather, ids=[${ids.slice(0, 4)}...] (${ids.length} rows)`,
    err === 0,
    `rel err ${err.toExponential(1)} (bit-exact expected: dequant is one multiply)`,
  );
  ok(
    "real embedding rows are finite and not all zero",
    got.every(Number.isFinite) && amax(got) > 1e-4,
    `|max| ${amax(got).toExponential(2)}`,
  );
}

// ==================================================================== argmax
console.log("\nargmax.wgsl -- two-stage max-index reduction\n");

const argmaxSrc = await src("argmax.wgsl");
const amPass1 = mk(argmaxSrc, "pass1");
const amPass2 = mk(argmaxSrc, "pass2");

/** The host half of the two-stage reduction: returns the winning index. */
async function gpuArgmax(x: Float32Array, nGroups = 64): Promise<number> {
  const u32 = (n: number) =>
    dev.createBuffer({
      size: Math.max(16, n * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  const xb = storageBuffer(dev, x);
  const pv = u32(nGroups), pi = u32(nGroups);
  const ov = u32(1), oi = u32(1);
  // pass1 declares xidx so that it shares a bind group layout with pass2, but it
  // never reads a live index there. It still needs a DISTINCT buffer bound: aiming
  // xidx at `pi`, which pass1 also writes, is read + read_write on one buffer,
  // which WebGPU rejects outright rather than tolerating.
  const dummy = u32(1);
  const d1 = uniformBuffer(dev, [x.length, nGroups, 0, 0]);
  // Stage 2 reduces nGroups partials with one workgroup, so its `n_groups` is the
  // partial count and its `n` is unused.
  const d2 = uniformBuffer(dev, [nGroups, nGroups, 0, 0]);

  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(amPass1);
  p.setBindGroup(0, bind(amPass1, [xb, dummy, pv, pi, d1]));
  p.dispatchWorkgroups(nGroups);
  p.end();
  // A second pass, not a second dispatch in the same pass: stage 2 reads the
  // buffers stage 1 wrote, and a pass boundary is the cheap way to be certain the
  // writes are visible rather than relying on in-pass ordering for a
  // read-after-write on a different bind group.
  const p2 = enc.beginComputePass();
  p2.setPipeline(amPass2);
  p2.setBindGroup(0, bind(amPass2, [pv, pi, ov, oi, d2]));
  p2.dispatchWorkgroups(1);
  p2.end();
  dev.queue.submit([enc.finish()]);

  const raw = await readBack(dev, oi, 4);
  const idx = new Uint32Array(raw.buffer)[0];
  for (const b of [xb, pv, pi, ov, oi, dummy, d1, d2]) b.destroy();
  return idx;
}

{
  for (const n of [256, 1000, 4096, 151936]) {
    const x = randVec(n, 10);
    const got = await gpuArgmax(x);
    const want = argmaxRef(x);
    ok(`argmax n=${n}`, got === want, `got ${got}, want ${want}`);
  }
  // A planted maximum at each of the awkward positions: first, last, and just
  // past a workgroup boundary.
  for (const at of [0, 1, 255, 256, 257, 151935, 75000]) {
    const x = new Float32Array(151936);
    for (let i = 0; i < x.length; i++) x[i] = -1 - (i % 7) * 0.001;
    x[at] = 5;
    const got = await gpuArgmax(x);
    ok(`argmax finds a planted max at index ${at}`, got === at, `got ${got}`);
  }
  // Ties. The contract is "lowest index wins", matching numpy/torch, so a greedy
  // decode diff against ONNX cannot be derailed by two logits landing on the same
  // f32. This is constructed rather than hoped for.
  {
    const x = new Float32Array(151936).fill(-1);
    x[900] = 3;
    x[40000] = 3;
    x[151000] = 3;
    const got = await gpuArgmax(x);
    ok(
      "argmax breaks ties toward the LOWEST index (numpy/torch rule)",
      got === 900 && argmaxRef(x) === 900,
      `got ${got}, ref ${argmaxRef(x)}`,
    );
  }
  // Group count must not change the answer: 594 groups is what the LM head uses
  // (151936 / 256), and 1 group exercises the degenerate stage 2.
  {
    const x = randVec(151936, 10);
    const want = argmaxRef(x);
    const results = [];
    for (const g of [1, 8, 64, 594]) results.push(await gpuArgmax(x, g));
    ok(
      "argmax is independent of the stage-1 group count",
      results.every((r) => r === want),
      `${results.join(", ")} vs want ${want}`,
    );
  }
  // All-negative logits: the sentinel must not win. Softmax inputs are routinely
  // all-negative, and an initial max of 0 instead of -FLT_MAX would return 0 here.
  {
    const x = new Float32Array(151936);
    for (let i = 0; i < x.length; i++) x[i] = -100 - (x.length - i) * 1e-3;
    const got = await gpuArgmax(x);
    ok(
      "argmax handles all-negative input",
      got === argmaxRef(x),
      `got ${got}, want ${argmaxRef(x)} (last index, largest = least negative)`,
    );
  }
}

// ======================================================== prefill attention
console.log("\nattention_prefill.wgsl -- N-query causal GQA, streaming softmax\n");

const prePipe = mk(await src("attention_prefill.wgsl"));

async function gpuPrefillAttn(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  nHeads: number,
  nKvHeads: number,
  headDim: number,
  nQueries: number,
  pos0: number,
): Promise<Float32Array> {
  const bufs = [
    storageBuffer(dev, q),
    storageBuffer(dev, k),
    storageBuffer(dev, v),
    dev.createBuffer({
      size: nQueries * nHeads * headDim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }),
    uniformBuffer(dev, [nHeads, nKvHeads, headDim, nQueries, pos0, 0, 0, 0]),
  ];
  run(prePipe, bufs, [nHeads, nQueries]);
  const out = await readBack(dev, bufs[3], nQueries * nHeads * headDim * 4);
  for (const b of bufs) b.destroy();
  return out;
}

{
  const cases: [number, number, number, number, number, string][] = [
    // nHeads, nKvHeads, headDim, nQueries, pos0
    [16, 8, 128, 1, 0, "single token, empty cache (the prefill-of-1 case)"],
    [16, 8, 128, 7, 0, "7-token prompt from scratch"],
    [16, 8, 128, 13, 0, "13-token prompt (the chat-template length)"],
    [16, 8, 128, 64, 0, "64 queries: exactly one TILE"],
    [16, 8, 128, 65, 0, "65 queries: TILE boundary + 1, the off-by-one case"],
    [16, 8, 128, 129, 0, "129 queries: three tiles"],
    [16, 8, 128, 5, 40, "5 new tokens against a 40-key cache (second turn)"],
    [16, 8, 128, 20, 300, "20 new tokens against a 300-key cache"],
    [16, 16, 128, 9, 0, "no GQA (heads == kv heads)"],
    [4, 1, 128, 11, 0, "extreme GQA, 4 heads sharing 1 kv head"],
    [16, 8, 128, 3, 2100, "past 2048 keys -- the cap attention.wgsl cannot pass"],
  ];
  for (const [nH, nKv, hd, nQ, pos0, label] of cases) {
    const nKeys = pos0 + nQ;
    const q = randVec(nQ * nH * hd, 1);
    const k = randVec(nKeys * nKv * hd, 1);
    const v = randVec(nKeys * nKv * hd, 1);
    const got = await gpuPrefillAttn(q, k, v, nH, nKv, hd, nQ, pos0);
    const want = attnPrefillRef(q, k, v, nH, nKv, hd, nQ, pos0);
    const err = absErrScaled(got, want, amax(want));
    ok(
      `prefill attn ${nH}q/${nKv}kv nQ=${nQ} pos0=${pos0}`,
      err < 1e-5,
      `abs err / scale ${err.toExponential(1)}  (${label})`,
    );
  }
}

// The structural check: a prefill's LAST query must equal what the decode kernel's
// reference computes for that same position, because both attend to the same keys.
// This is what proves the causal bound is `pos0 + qi + 1` and not off by one --
// an off-by-one there still produces a valid-looking softmax.
{
  const nH = 16, nKv = 8, hd = 128, nQ = 9;
  const q = randVec(nQ * nH * hd, 1);
  const k = randVec(nQ * nKv * hd, 1);
  const v = randVec(nQ * nKv * hd, 1);
  const pre = await gpuPrefillAttn(q, k, v, nH, nKv, hd, nQ, 0);
  for (const at of [0, 1, 4, nQ - 1]) {
    // Decode reference for query `at`: it sees keys 0..at.
    const qOne = q.slice(at * nH * hd, (at + 1) * nH * hd);
    const want = attnRef(
      qOne,
      k.subarray(0, (at + 1) * nKv * hd),
      v.subarray(0, (at + 1) * nKv * hd),
      nH,
      nKv,
      hd,
      at + 1,
    );
    const got = pre.slice(at * nH * hd, (at + 1) * nH * hd);
    const err = absErrScaled(got, want, amax(want));
    ok(
      `prefill query ${at} matches the decode reference at position ${at}`,
      err < 2e-5,
      `abs err / scale ${err.toExponential(1)}`,
    );
  }
}

// Causality, asserted directly rather than inferred: perturbing key t must not
// change the output of any query before t. A kernel that attends to the future
// still produces plausible numbers, so this is the check that catches it.
{
  const nH = 16, nKv = 8, hd = 128, nQ = 8;
  const q = randVec(nQ * nH * hd, 1);
  const k = randVec(nQ * nKv * hd, 1);
  const v = randVec(nQ * nKv * hd, 1);
  const base = await gpuPrefillAttn(q, k, v, nH, nKv, hd, nQ, 0);
  // Change key 5 (and its value) completely.
  const k2 = Float32Array.from(k), v2 = Float32Array.from(v);
  for (let i = 0; i < nKv * hd; i++) {
    k2[5 * nKv * hd + i] = 9 + i * 0.01;
    v2[5 * nKv * hd + i] = -7 - i * 0.01;
  }
  const pert = await gpuPrefillAttn(q, k2, v2, nH, nKv, hd, nQ, 0);
  let beforeChanged = 0, afterChanged = 0;
  for (let qi = 0; qi < nQ; qi++) {
    for (let i = 0; i < nH * hd; i++) {
      const at = qi * nH * hd + i;
      if (base[at] !== pert[at]) {
        if (qi < 5) beforeChanged++;
        else afterChanged++;
      }
    }
  }
  ok(
    "queries before a changed key are bit-identical (causal mask is real)",
    beforeChanged === 0,
    `${beforeChanged} changed before, ${afterChanged} at/after`,
  );
  ok(
    "queries at and after a changed key DO change (the key is actually read)",
    afterChanged > nH * hd,
    `${afterChanged} entries changed`,
  );
}

Deno.exit(summary() ? 1 : 0);
