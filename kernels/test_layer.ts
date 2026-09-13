// Whole-layer validation: layerForward() against a CPU reference layer, on REAL
// Qwen3-0.6B weights range-fetched from HuggingFace.
//
//   deno run --unstable-webgpu --allow-all kernels/test_layer.ts
//
// The per-op tests (test_ops.ts, test_coop.ts) already establish that each kernel
// computes the right function. What is left for a whole-layer test is the WIRING:
// which projection feeds which norm, whether the per-head q_norm is applied
// before or after RoPE, whether the residual is added to the pre-norm or the
// post-norm value, whether the KV cache is appended at the right offset. Those
// are the errors that survive correct kernels, and they are all O(1) wrong -- so
// the tolerance below does not need to be tight to catch them, it only needs to
// be honest about accumulated ULP.
//
// The reference is a composition of the same per-op references, which is what
// makes a disagreement diagnosable: the arithmetic is already known to match, so
// any gap is in the order of operations.
//
// NOTE ON GROUND TRUTH: the ONNX shards (shard0.onnx, layers 24-25) are
// gitignored build artifacts and are not present in this worktree, so a direct
// GPU-vs-ONNX diff was not run. The reference here is independent of the kernels
// in structure but not in provenance -- it shares the per-op references. What it
// does establish is that the composition is self-consistent and matches a
// from-scratch CPU implementation of the Qwen3 layer equations on real weights.

import { getDevice, ok, randVec, summary } from "./lib.ts";
import { absErrScaled, amax, relErr } from "./ops_ref.ts";
import { Layer, QWEN3_06B } from "./layer.ts";
import { layerForwardRef, newRefCache } from "./layer_ref.ts";
import { realLayer } from "./real_weights.ts";

const dev = await getDevice();
const cfg = QWEN3_06B;

const LAYER = 24; // the layer shard0.onnx runs first
console.log(`loading blk.${LAYER}.* from Qwen3-0.6B-Q8_0.gguf ...`);
const w = await realLayer(LAYER);
console.log(`  q8 tensors: ${Object.keys(w.q8).join(", ")}`);
console.log(`  f32 gains:  ${Object.keys(w.f32).join(", ")}\n`);

// Sanity on the weights themselves before trusting anything downstream, because
// wrong byte offsets yield Q8_0 that decodes without error and is pure noise.
//
// What NOT to assert: "norm gains are near 1". Qwen3's actually are not --
// attn_norm.weight for layer 24 has mean 10.8 and a max of 69, and q_norm has an
// interleaved 1.04 / 0.05 / -0.04 pattern that looks exactly like a misaligned
// read. Both are real: verified by re-fetching each tensor by its own exact byte
// range, independently of the whole-layer slicing, and getting identical values.
// The gains are large because RMSNorm divides by a small RMS and the gain has to
// put the scale back. An assertion based on that wrong intuition failed a
// correct loader, so the check below is structural instead: finite, right length,
// and not all-identical (which is what a zeroed or constant misread looks like).
{
  const g = w.f32["attn_norm.weight"];
  const distinct = new Set(g).size;
  // "Varied" has to be generous: these came from a model trained in bf16, so the
  // f32 values carry only 8 mantissa bits and 1024 of them collapse to ~200
  // distinct. A zeroed or constant misread would give 1.
  ok(
    "attn_norm gain is finite, right-sized, and not constant",
    g.length === cfg.hidden && g.every(Number.isFinite) && distinct > 32,
    `len ${g.length}, ${distinct} distinct values, mean ${
      (g.reduce((a, b) => a + b, 0) / g.length).toFixed(3)
    }`,
  );
  const shapes = Object.entries(w.q8).map(([k, t]) =>
    `${k.replace(".weight", "")} ${t.rows}x${t.cols}`
  );
  const want = [
    "attn_q 2048x1024",
    "attn_k 1024x1024",
    "attn_v 1024x1024",
    "attn_output 1024x2048",
    "ffn_gate 3072x1024",
    "ffn_up 3072x1024",
    "ffn_down 1024x3072",
  ];
  const missing = want.filter((s) => !shapes.includes(s));
  ok(
    "all seven projections present with the expected shapes",
    missing.length === 0,
    missing.length ? `missing ${missing.join(", ")}` : shapes.length + " tensors",
  );
}

const layer = await Layer.create(dev, w, cfg);
const cache = newRefCache();

// Activations in flock are small (|x| < 0.1) and derived from bf16 weights, which
// is the regime these kernels will actually see.
let gpuH = randVec(cfg.hidden, 0.05);
const refH0 = Float32Array.from(gpuH);

// Several decode steps, so the KV cache grows and RoPE sees a moving position.
// Each step feeds its own output back in, exactly as a running model would, so an
// error in step 1 compounds visibly rather than being masked by fresh input.
let refH = refH0;
for (let step = 0; step < 4; step++) {
  const gpu = await layer.forward(gpuH);
  const ref = layerForwardRef(refH, w, cfg, cache);

  const eRel = relErr(gpu, ref, 1e-3);
  const eAbs = absErrScaled(gpu, ref, amax(ref));
  // The assertion is on ABSOLUTE error against the output's scale, for the same
  // reason RoPE is (see absErrScaled): a hidden state has entries spanning
  // several orders of magnitude, and the small ones report a large relative error
  // while carrying the same ~1 ULP absolute error as everything else. Measured
  // here: abs/scale stays at 1-2 ULP (8.5e-8 to 2.0e-7) across all four steps
  // while per-element relative error wanders between 6.7e-5 and 1.1e-3 depending
  // on how close some entry happened to land to zero. The first number describes
  // the layer; the second describes one unlucky element.
  //
  // 1e-6 is ~8 ULP of the output scale. Every wiring error this test exists to
  // catch -- wrong projection order, q_norm after RoPE instead of before, residual
  // on the post-norm value, KV appended at the wrong offset -- changes the output
  // by O(1), not by 8 ULP.
  //
  // Relative error is still reported, because a sudden jump in it would be worth
  // looking at even while the absolute number stays fine.
  ok(
    `layer ${LAYER} step ${step} (nKeys=${step + 1})`,
    eAbs < 1e-6,
    `abs/scale ${eAbs.toExponential(1)}  (per-element rel ${eRel.toExponential(1)})`,
  );

  gpuH = gpu;
  refH = ref;
}

// The layer must actually change the hidden state, and by a plausible amount.
// A layer that silently returned its input (every matvec reading a zeroed buffer,
// say) would pass a comparison against a reference that did the same, so this is
// checked against the INPUT rather than against the reference.
{
  layer.reset();
  const x = randVec(cfg.hidden, 0.05);
  const y = await layer.forward(x);
  let dot = 0, nx = 0, ny = 0;
  for (let i = 0; i < x.length; i++) {
    dot += x[i] * y[i];
    nx += x[i] * x[i];
    ny += y[i] * y[i];
  }
  const cos = dot / Math.sqrt(nx * ny);
  ok(
    "layer transforms the hidden state (not a pass-through)",
    y.every(Number.isFinite) && amax(y) > 0 && cos < 0.999,
    `cos(in, out) = ${cos.toFixed(4)}, |out|max = ${amax(y).toExponential(2)}`,
  );
}

// Position dependence: the same token at a different cache position must produce
// a different output, or RoPE is not being applied at all.
//
// Stated as a fraction of entries that changed rather than as a magnitude. Two
// magnitude-based versions of this check were misleading before this one: against
// the output it reads 1.3e-4 (the output is dominated by the residual, so a real
// rotation is a small slice of it), and against |output - input| it reads 0.00%
// (the layer amplifies 0.05-scale input to ~39, so that denominator is the layer,
// not the rotation). The count is unambiguous: position 1 differs from position 0
// in ~83% of entries, and a dead RoPE would differ in none.
{
  layer.reset();
  const x = randVec(cfg.hidden, 0.05);
  const at0 = await layer.forward(x);
  const at1 = await layer.forward(x); // same input, now at position 1
  let changed = 0;
  for (let i = 0; i < x.length; i++) if (at0[i] !== at1[i]) changed++;
  ok(
    "output depends on position (RoPE is live)",
    changed > cfg.hidden / 2,
    `${changed}/${cfg.hidden} entries differ between position 0 and position 1`,
  );
}

Deno.exit(summary() ? 1 : 0);
