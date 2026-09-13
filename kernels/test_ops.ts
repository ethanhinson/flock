// Per-op validation for the non-matmul kernels: RMSNorm, RoPE, SwiGLU, add, and
// causal grouped-query attention. Each is checked on its own against a strict-f32
// CPU reference BEFORE anything composes them, because a wrong op inside a
// composed layer is nearly impossible to isolate afterwards.
//
//   deno run --unstable-webgpu --allow-all kernels/test_ops.ts
//
// Which ops can be bit-exact and which cannot:
//
//   add                 exact.
//   rmsnorm             exact at most widths, 1 ULP at some. Metal's rsqrt is an
//                       approximation plus a refinement rather than a correctly
//                       rounded 1/sqrt, so the last bit depends on the value.
//                       Verified deterministic, so it is rounding, not a race.
//   rope, swiglu, attn  NOT exact, and no amount of care makes them so: they use
//                       cos/sin/exp, which are *different functions* in Metal and
//                       in V8. Measured on this machine, Metal's exp() differs
//                       from V8's Math.exp() by up to 15 ULP (1.0e-6 relative)
//                       over [-20, 0], the range a max-subtracted softmax feeds
//                       it. That is the floor for these three ops.
//
// The tolerances below are derived from that measurement, not tuned until the
// tests went green. The check that they are not hiding anything: a reference
// rewritten to mirror the kernel's exact reduction order does NOT reduce
// attention's error (1.6e-5 either way), which is what says the residue is the
// transcendental rather than a summation-order mismatch. Separately, each op has
// a structural assertion that does not depend on any tolerance -- RoPE's pairing
// convention and attention's GQA head mapping are pinned by exact equality on
// constructed inputs, because those are the errors a loose tolerance could hide.

import {
  attnSource, getDevice, ok, randVec, readBack, ropeSource, storageBuffer, summary,
  uniformBuffer,
} from "./lib.ts";
import {
  absErrScaled, addRef, amax, attnRef, relErr, rmsnormRef, ropeInvFreq, ropeRef,
  swigluRef,
} from "./ops_ref.ts";

const dev = await getDevice();
const read = (f: string) => Deno.readTextFile(new URL("./" + f, import.meta.url));

function pipelineFor(code: string, entryPoint: string) {
  return dev.createComputePipeline({
    layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint },
  });
}

/** Run a pipeline over `bufs`, reading back buffer index `outIdx`. */
async function run(
  pipe: GPUComputePipeline, bufs: GPUBuffer[], outIdx: number, outFloats: number,
  groups: number,
) {
  const bg = dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(groups); p.end();
  dev.queue.submit([enc.finish()]);
  return await readBack(dev, bufs[outIdx], outFloats * 4);
}

const rw = (n: number) => dev.createBuffer({
  size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

// ---------------------------------------------------------------- RMSNorm
{
  const pipe = pipelineFor(await read("rmsnorm.wgsl"), "main");
  const EPS = 9.999999974752427e-7;   // qwen3.attention.layer_norm_rms_epsilon

  // (hidden 1024, one vector) is the attn_norm/ffn_norm case; (head_dim 128, 16
  // vectors) is Qwen3's per-head q_norm, which is the unusual one; 1024x4 is a
  // 4-token prompt; 300 is deliberately not a multiple of the 128-wide workgroup.
  for (const [n, nVecs, why] of [
    [1024, 1, "attn_norm / ffn_norm"],
    [128, 16, "q_norm, per head"],
    [128, 8, "k_norm, per head"],
    [1024, 4, "4-token prompt"],
    [300, 3, "width not a multiple of the workgroup"],
  ] as [number, number, string][]) {
    const x = randVec(n * nVecs, 1);
    const g = randVec(n, 0.5);
    const bufs = [storageBuffer(dev, x), storageBuffer(dev, g), rw(n * nVecs),
      uniformBuffer(dev, [n, nVecs, 0, 0])];
    // eps is an f32 in the uniform, so it has to be written as float bits.
    dev.queue.writeBuffer(bufs[3], 0, new Uint32Array([n, nVecs, 0, 0]));
    dev.queue.writeBuffer(bufs[3], 8, new Float32Array([EPS]));
    const gpu = await run(pipe, bufs, 2, n * nVecs, nVecs);
    const cpu = rmsnormRef(x, g, n, nVecs, EPS);
    const e = relErr(gpu, cpu);
    // Exact at most sizes once the reference models Metal's single-rounding
    // reciprocal-sqrt (see rmsnormRef). At some widths it is still 1 ULP off:
    // Metal's rsqrt is evidently a hardware approximation plus a refinement step
    // rather than a correctly-rounded 1/sqrt, and which way the last bit lands
    // depends on the value. Measured across these shapes: 0 or 1.8e-7, never
    // more, and deterministic run to run (checked -- 0 differing entries over 6
    // runs, so this is a rounding difference, not a race).
    //
    // 3e-7 is 2.5 ULP. The whole output shares one scale, so a genuinely wrong
    // reduction moves every element together and by far more than this.
    ok(`rmsnorm ${n}x${nVecs}`, e < 3e-7, `rel err ${e.toExponential(1)}  (${why})`);
    for (const b of bufs) b.destroy();
  }

  // Determinism, run to run on identical input. This is what separates "the
  // reference models rounding slightly differently" (fine, bounded) from "there
  // is a missing barrier" (not fine, and would show up here as drift). Without
  // this check a racy reduction and a 1-ULP rounding difference look identical.
  {
    const n = 128, nVecs = 16;
    const x = randVec(n * nVecs, 1), g = randVec(n, 0.5);
    const runs: Float32Array[] = [];
    for (let k = 0; k < 6; k++) {
      const bufs = [storageBuffer(dev, x), storageBuffer(dev, g), rw(n * nVecs),
        uniformBuffer(dev, [n, nVecs, 0, 0])];
      dev.queue.writeBuffer(bufs[3], 8, new Float32Array([EPS]));
      runs.push(await run(pipe, bufs, 2, n * nVecs, nVecs));
      for (const b of bufs) b.destroy();
    }
    let diff = 0;
    for (let k = 1; k < runs.length; k++) {
      for (let i = 0; i < runs[0].length; i++) if (runs[k][i] !== runs[0][i]) diff++;
    }
    ok("rmsnorm is deterministic across runs", diff === 0,
      `${diff} entries drifted over 6 runs (a missing barrier would show here)`);
  }

  // A zero vector is the edge case eps exists for: without it the normalizer is
  // 1/0. With eps the output must be exactly zero, not NaN.
  {
    const n = 128, x = new Float32Array(n), g = randVec(n, 1);
    const bufs = [storageBuffer(dev, x), storageBuffer(dev, g), rw(n),
      uniformBuffer(dev, [n, 1, 0, 0])];
    dev.queue.writeBuffer(bufs[3], 8, new Float32Array([EPS]));
    const gpu = await run(pipe, bufs, 2, n, 1);
    ok("rmsnorm all-zero input stays finite", gpu.every((v) => v === 0), "no NaN");
    for (const b of bufs) b.destroy();
  }
}

// ---------------------------------------------------------------- RoPE
// BOTH pairing conventions are tested, because both are shipped: which one a set
// of weights needs is a property of the weights and is selected by the host (see
// rope.wgsl). Testing only the default would leave the other silently rotting, and
// the two differ by nothing an output magnitude can reveal.
for (const pairing of ["norm", "neox"] as const) {
  const pipe = pipelineFor(ropeSource(await read("rope.wgsl"), pairing), "main");
  const HEAD_DIM = 128, BASE = 1e6;   // qwen3.rope.freq_base
  const invFreq = ropeInvFreq(HEAD_DIM, BASE);

  for (const [nTokens, nHeads, pos0, why] of [
    [1, 16, 0, "q at position 0"],
    [1, 16, 137, "q mid-sequence (KV cache offset)"],
    [1, 8, 137, "k, 8 kv heads"],
    [7, 16, 0, "7-token prompt"],
  ] as [number, number, number, string][]) {
    const n = nTokens * nHeads * HEAD_DIM;
    const x = randVec(n, 1);
    const bufs = [storageBuffer(dev, x), uniformBuffer(dev, [nTokens, nHeads, HEAD_DIM, pos0]),
      storageBuffer(dev, invFreq)];
    const threads = nTokens * nHeads * (HEAD_DIM / 2);
    const gpu = await run(pipe, bufs, 0, n, Math.ceil(threads / 64));
    const cpu = ropeRef(x, nTokens, nHeads, HEAD_DIM, pos0, invFreq, pairing);
    // Absolute error against the data's scale, not per-element relative error.
    // A rotation of two O(1) inputs can land near zero by cancellation, and that
    // one entry then reports a huge relative error while carrying the same ~1 ULP
    // absolute error as everything else -- measured 7.8e-5 relative versus
    // 1.79e-7 absolute on the same tensor. See absErrScaled in ops_ref.ts.
    //
    // 4e-7 is ~3 ULP of the input scale: above the cos/sin ULP difference between
    // Metal and V8, and far below a structural error, which would be O(1).
    const e = absErrScaled(gpu, cpu, amax(x));
    ok(`rope[${pairing}] ${nTokens}x${nHeads}x${HEAD_DIM} pos0=${pos0}`, e < 4e-7,
      `abs err / scale ${e.toExponential(1)}  (${why})`);
    for (const b of bufs) b.destroy();
  }

  // The pairing itself, asserted without depending on the reference agreeing: put
  // a 1 at element 0 and nothing anywhere else, rotate by position 1, and read
  // where the sine landed. Under NORM the partner of element 0 is element 1; under
  // NEOX it is element 64. Both spellings produce a valid rotation of a valid
  // vector, so this planted-input check is the only thing that distinguishes them.
  {
    const n = HEAD_DIM;
    const x = new Float32Array(n);
    x[0] = 1;
    const bufs = [storageBuffer(dev, x), uniformBuffer(dev, [1, 1, HEAD_DIM, 1]),
      storageBuffer(dev, invFreq)];
    const gpu = await run(pipe, bufs, 0, n, Math.ceil(HEAD_DIM / 2 / 64));
    const theta = invFreq[0];            // pos 1 * inv_freq[0]
    const partner = pairing === "norm" ? 1 : HEAD_DIM / 2;
    const okPair = Math.abs(gpu[0] - Math.cos(theta)) < 1e-6 &&
      Math.abs(gpu[partner] - Math.sin(theta)) < 1e-6;
    ok(`rope[${pairing}] pairs element 0 with element ${partner}`, okPair,
      `x[0]=${gpu[0].toFixed(6)}, x[${partner}]=${gpu[partner].toFixed(6)} ` +
      `vs cos/sin(${theta.toFixed(6)})`);
    // And the OTHER convention's partner must be untouched, which is what makes
    // this a discriminating test rather than a consistency one.
    const other = pairing === "norm" ? HEAD_DIM / 2 : 1;
    ok(`rope[${pairing}] leaves element ${other} alone (the other convention's partner)`,
      gpu[other] === 0, `x[${other}] = ${gpu[other]}`);
    for (const b of bufs) b.destroy();
  }
}

// ------------------------------------------------------- SwiGLU and add
{
  const code = await read("elementwise.wgsl");
  const swiglu = pipelineFor(code, "swiglu");
  const addPipe = pipelineFor(code, "add");

  for (const n of [3072, 1024, 1000]) {
    const a = randVec(n, 3), b = randVec(n, 1);
    const mk = () => [storageBuffer(dev, a), storageBuffer(dev, b), rw(n),
      uniformBuffer(dev, [n, 0, 0, 0])];

    const sb = mk();
    const gs = await run(swiglu, sb, 2, n, Math.ceil(n / 256));
    // exp() is a few ULP different between backends, and silu's derivative is
    // bounded, so a few ULP in gives a few ULP out. 1e-6 is ~8 ULP near 1.0.
    const es = relErr(gs, swigluRef(a, b), 1e-3);
    ok(`swiglu n=${n}`, es < 1e-6, `rel err ${es.toExponential(1)}`);
    for (const x of sb) x.destroy();

    const ab = mk();
    const ga = await run(addPipe, ab, 2, n, Math.ceil(n / 256));
    const ea = relErr(ga, addRef(a, b));
    ok(`add n=${n}`, ea === 0, `rel err ${ea.toExponential(1)}`);
    for (const x of ab) x.destroy();
  }

  // silu at large negative input underflows to zero rather than producing a NaN
  // from exp(-v) overflowing. Worth pinning: it is the one input range where a
  // naive sigmoid breaks.
  {
    const n = 256;
    const a = new Float32Array(n), b = new Float32Array(n).fill(1);
    for (let i = 0; i < n; i++) a[i] = -100 - i;
    const bufs = [storageBuffer(dev, a), storageBuffer(dev, b), rw(n),
      uniformBuffer(dev, [n, 0, 0, 0])];
    const g = await run(swiglu, bufs, 2, n, Math.ceil(n / 256));
    ok("swiglu stays finite for large negative input", g.every(Number.isFinite),
      `min ${Math.min(...g).toExponential(1)}`);
    for (const x of bufs) x.destroy();
  }
}

// ---------------------------------------------------- attention (GQA, causal)
{
  // 1024 keys is plenty for these shapes and keeps the workgroup scores array at
  // 4 KB. Sizing it to the 4096 the spec guarantees costs 15x; see attention.wgsl.
  const pipe = pipelineFor(attnSource(await read("attention.wgsl"), 1024), "main");
  const HEAD_DIM = 128;

  for (const [nHeads, nKvHeads, nKeys, why] of [
    [16, 8, 1, "first token, cache of 1"],
    [16, 8, 137, "mid-sequence decode"],
    [16, 8, 512, "longer cache"],
    [16, 16, 64, "no GQA (heads == kv heads)"],
    [4, 1, 33, "extreme GQA, 4 heads sharing 1 kv head"],
  ] as [number, number, number, string][]) {
    const q = randVec(nHeads * HEAD_DIM, 1);
    const k = randVec(nKeys * nKvHeads * HEAD_DIM, 1);
    const v = randVec(nKeys * nKvHeads * HEAD_DIM, 1);
    const bufs = [storageBuffer(dev, q), storageBuffer(dev, k), storageBuffer(dev, v),
      rw(nHeads * HEAD_DIM), uniformBuffer(dev, [nHeads, nKvHeads, HEAD_DIM, nKeys])];
    const gpu = await run(pipe, bufs, 3, nHeads * HEAD_DIM, nHeads);
    const cpu = attnRef(q, k, v, nHeads, nKvHeads, HEAD_DIM, nKeys);
    const e = relErr(gpu, cpu, 1e-3);
    // Budget: exp() contributes up to 1.0e-6 (measured), softmax divides by a sum
    // of nKeys of them, and the v-weighted sum accumulates nKeys more. 1e-4 is
    // roughly an order of magnitude above what that produces in practice (1.5e-5
    // to 2.4e-5 across these shapes) and orders of magnitude below any structural
    // error -- a wrong head map or a missing mask is O(1), not O(1e-5).
    ok(`attn ${nHeads}q/${nKvHeads}kv keys=${nKeys}`, e < 1e-4,
      `rel err ${e.toExponential(1)}  (${why})`);
    for (const b of bufs) b.destroy();
  }

  // The GQA mapping is the easiest thing to get subtly wrong, and a wrong map
  // still produces a well-formed softmax. So: make kv head 0 and kv head 1
  // carry completely different values and check that query heads 0-1 see the
  // first and heads 2-3 see the second. This fails loudly for an off-by-one or
  // an interleaved-instead-of-blocked grouping.
  {
    const nHeads = 4, nKvHeads = 2, nKeys = 1, hd = HEAD_DIM;
    const q = new Float32Array(nHeads * hd).fill(0);
    const k = new Float32Array(nKeys * nKvHeads * hd).fill(0);
    const v = new Float32Array(nKeys * nKvHeads * hd);
    // With one key the softmax is 1.0 regardless of scores, so the output of
    // head h is exactly v[kv head of h].
    for (let i = 0; i < hd; i++) { v[i] = 7; v[hd + i] = -3; }
    const bufs = [storageBuffer(dev, q), storageBuffer(dev, k), storageBuffer(dev, v),
      rw(nHeads * hd), uniformBuffer(dev, [nHeads, nKvHeads, hd, nKeys])];
    const gpu = await run(pipe, bufs, 3, nHeads * hd, nHeads);
    const want = [7, 7, -3, -3];
    let bad = 0;
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < hd; i++) if (gpu[h * hd + i] !== want[h]) bad++;
    }
    ok("GQA maps query head h to kv head h/(nHeads/nKvHeads)", bad === 0,
      `heads read ${[0, 1, 2, 3].map((h) => gpu[h * hd]).join(", ")}, want ${want.join(", ")}`);
    for (const b of bufs) b.destroy();
  }
}

Deno.exit(summary() ? 1 : 0);
