// layerForward(): one Qwen3 transformer layer, on the GPU, from GGUF weights.
//
// This is the piece that replaces an ONNX shard. It takes a hidden state and a
// layer's tensors and returns the next hidden state, composing the six validated
// kernels:
//
//   h1 = h + attn_output @ attention(rope(q_norm(Wq h)), rope(k_norm(Wk h)), Wv h)
//   h2 = h1 + ffn_down @ (silu(ffn_gate @ n(h1)) * (ffn_up @ n(h1)))
//
// with RMSNorm before each block. Qwen3 details that are easy to get wrong and
// are therefore stated explicitly here:
//
//  - q_norm and k_norm are RMSNorm applied PER HEAD over head_dim=128, not once
//    over the whole projection. Qwen3 is unusual in having them at all.
//  - GQA: 16 query heads, 8 kv heads, so q is 2048 wide and k/v are 1024.
//  - RoPE is applied after the per-head norm, to q and k only, never to v.
//  - The residual is added to the input of the block, not to the normalized
//    value -- pre-norm, so `h` is kept around while `n(h)` is consumed.
//
// GGUF tensor shapes are [in, out]: ggml lists the fastest-varying dimension
// first, so shape[0] is the input width (our `cols`) and shape[1] is the number
// of output rows. attn_q.weight [1024, 2048] is 2048 rows of 1024 columns.
//
// A decode step is 16 dispatches whose arithmetic totals ~0.2 ms, and getting it
// to run in anything close to that is entirely about NOT paying host costs per
// step. Measured, per layer:
//
//   forward(), own command buffer, own readback     28.9 ms
//   encode() x28 behind one readback                 1.46 ms   19.7x
//   encode() x28 sharing one command buffer          1.39 ms   20.8x
//
// Three things bought that, in order of size:
//
//  - Not reading back per layer. A map readback is ~27 ms on this backend; see
//    bench.ts for why it is also the only honest fence. `encode()` exists so the
//    hidden state stays on the GPU and the host reads once per shard.
//  - Caching bind groups. Creating them per step cost ~4 ms per layer.
//  - Sharing one command buffer across layers (~0.07 ms per layer).
//
// So the API has two halves on purpose: `encode()` for chaining, `forward()` for
// tests and for the one place a host actually needs the numbers.
//
// Buffers are allocated once and reused across tokens. Attention is one fused
// dispatch rather than five because at a ~17.6 us per-dispatch floor the dispatch
// count matters more than the arithmetic.

import {
  attnSource,
  coopSource,
  probeUnpack,
  probeUnpackF16,
  type RopePairing,
  ropeSource,
  splitQ8,
  storageBuffer,
  uniformBuffer,
} from "./lib.ts";

export interface LayerWeights {
  /** Q8_0 packed, straight from the GGUF file, keyed by tensor suffix. */
  q8: Record<string, { rows: number; cols: number; packed: Uint8Array }>;
  /** f32 norm gains, keyed by tensor suffix. */
  f32: Record<string, Float32Array>;
}

/**
 * One tensor that is ALREADY on the GPU, in the split layout the kernels want.
 *
 * This is exactly what `streamTensorToGPU` in web/js/gguf-stream.mjs returns, and
 * that is the point: a quantized tensor arrives as `{qs, scales}` and an f32 norm
 * gain as `{data}`, keyed by the tensor suffix after `blk.N.`.
 */
export interface GpuTensor {
  rows: number;
  cols: number;
  /** Q8_0: contiguous int8 quants, 4 per u32 word. */
  qs?: GPUBuffer;
  /** Q8_0: raw f16 scales, 2 per u32 word. */
  scales?: GPUBuffer;
  /** F32/F16: the tensor as-is, for the norm gains. */
  data?: GPUBuffer;
}

/** One layer's tensors, already resident on the GPU, keyed by suffix. */
export type LayerBuffers = Record<string, GpuTensor>;

export interface LayerConfig {
  hidden: number; // 1024
  nHeads: number; // 16
  nKvHeads: number; // 8
  headDim: number; // 128
  ffn: number; // 3072
  eps: number; // 1e-6
  ropeBase: number; // 1e6
  maxKeys: number; // KV cache capacity
  /**
   * Largest prefill batch, which sizes every activation buffer (q, k, v, the ffn
   * intermediates). Prefill is what makes those buffers N-wide: a decode step
   * needs 3072 floats of `gate`, a 256-token prefill needs 786432. Default 512,
   * which is ~10 MB of activations for Qwen3-0.6B; a longer prompt is chunked
   * rather than requiring a reallocation.
   */
  maxPrefill: number;
  /**
   * Which two elements of a head RoPE rotates together. "neox" (HuggingFace
   * halves) is correct for Qwen3-0.6B-Q8_0 -- MEASURED against the ONNX graphs,
   * not inferred from the fact that the weights came out of a GGUF. See
   * rope.wgsl and test_rope_convention.ts; the wrong choice is a
   * position-dependent silent degradation, not a failure.
   */
  ropePairing: RopePairing;
}

// maxKeys is 2048, not the 4096 that 16 KB of workgroup memory allows, and that
// is deliberate. 16 KB is exactly the point where only one workgroup fits per
// core, so attention loses all latency hiding: measured 328 us per dispatch at
// 4096 versus 21 us at 2048, a 15x cliff for 2x the context. Raise it only with
// that number in hand.
export const QWEN3_06B: LayerConfig = {
  hidden: 1024,
  nHeads: 16,
  nKvHeads: 8,
  headDim: 128,
  ffn: 3072,
  eps: 9.999999974752427e-7,
  ropeBase: 1e6,
  maxKeys: 2048,
  maxPrefill: 512,
  ropePairing: "neox",
};

const ROWS_PER_WG = 4; // must match q8_coop.wgsl

/**
 * queue.writeBuffer, but correct for a TypedArray that VIEWS part of a larger
 * buffer.
 *
 * Deno 2.1.2's wgpu backend ignores a view's byteOffset and length and uploads the
 * whole underlying ArrayBuffer from 0. Measured directly: writing
 * `new Float32Array([1..8]).subarray(4, 8)` into a 4-float buffer reports "Copy of
 * 0..32 would end up overrunning the bounds of the Destination buffer of size 16".
 *
 * That error only appears when the destination happens to be too small. When it is
 * large enough -- which it is for every activation buffer here, since they are
 * sized for maxPrefill -- the write SILENTLY lands the wrong data, and the symptom
 * is a model that computes a plausible wrong answer. It cost real time: feeding
 * decode steps as `hidden.subarray(i * H, (i + 1) * H)` made every step consume
 * token 0, so the KV cache filled with one key repeated and the prefill-vs-decode
 * equivalence test failed with the GPU prefill (correct) blamed for it.
 *
 * Passing the ArrayBuffer with an explicit byte offset and size is honoured
 * correctly, so that is what this does. Slicing instead would also work and would
 * cost a copy of every hidden state on every step.
 */
function writeView(dev: GPUDevice, buf: GPUBuffer, offset: number, data: Float32Array) {
  dev.queue.writeBuffer(buf, offset, data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Read a WGSL file that sits next to this module.
 *
 * Two runtimes consume these kernels and they read files differently: the tests
 * and the coordinator run under Deno, where `Deno.readTextFile` on a file URL is
 * the direct route; a bird runs in a browser, where the same module is served
 * over HTTP and only `fetch` exists. Resolving against `import.meta.url` makes
 * one expression work for both -- the URL is a `file:` URL under Deno and an
 * `http:` one in the page -- so there is no build step and no second copy of the
 * loader to keep in sync.
 */
export async function wgslSource(file: string): Promise<string> {
  const url = new URL("./" + file, import.meta.url);
  const deno = (globalThis as { Deno?: { readTextFile(p: URL): Promise<string> } }).Deno;
  if (deno?.readTextFile) return await deno.readTextFile(url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not fetch ${file}: ${res.status}`);
  return await res.text();
}

/** inv_freq[j] = 1 / base^(2j/headDim), computed on the host so it is exact. */
function invFreqOf(headDim: number, base: number): Float32Array {
  const out = new Float32Array(headDim / 2);
  for (let j = 0; j < out.length; j++) {
    out[j] = Math.fround(1 / Math.pow(base, (2 * j) / headDim));
  }
  return out;
}

export class Layer {
  private dev: GPUDevice;
  private cfg: LayerConfig;
  private pipes!: Record<string, GPUComputePipeline>;
  private buf: Record<string, GPUBuffer> = {};
  private wq: Record<string, {
    qs: GPUBuffer;
    sc: GPUBuffer;
    /** matvec Dims with n_tokens = 1, for decode. */
    dims: GPUBuffer;
    /** matvec Dims with n_tokens = the current prefill batch, rewritten per chunk. */
    dimsPre: GPUBuffer;
    rows: number;
    cols: number;
  }> = {};
  /** How many keys are currently in the cache. */
  nKeys = 0;

  private constructor(dev: GPUDevice, cfg: LayerConfig) {
    this.dev = dev;
    this.cfg = cfg;
  }

  /**
   * Build a layer from weights that are ALREADY in GPU buffers.
   *
   * This is the bird's entry point. A bird streams its ~67 MB of layers straight
   * from HuggingFace into GPU buffers (web/js/gguf-stream.mjs) precisely so the
   * bytes never sit in the JS heap -- that streaming is what stops an iPad dying
   * at 126 MB. `create()` below cannot consume that: it takes packed CPU bytes and
   * calls splitQ8 on them, which is the materialize-the-whole-tensor step the
   * streaming exists to avoid. So the two paths differ in exactly one thing --
   * where the split happens -- and everything after it is shared.
   *
   * The keys are the tensor suffix after `blk.N.` WITH `.weight` left on
   * (`attn_q.weight`), which is the convention both real_weights.ts and the
   * streaming loader already use. Stripping it makes every lookup below return
   * undefined, which is a silent wrong answer rather than an error -- so the
   * required tensors are checked by name here.
   */
  static async fromBuffers(
    dev: GPUDevice,
    w: LayerBuffers,
    cfg: LayerConfig = QWEN3_06B,
  ): Promise<Layer> {
    const L = await Layer.build(dev, cfg);
    for (const [name, t] of Object.entries(w)) {
      if (t.qs && t.scales) {
        // Already split by the streamer, byte-identical to splitQ8's output. The
        // two dims uniforms are per-weight state, not weight bytes, so they are
        // built here exactly as create() builds them.
        L.wq[name] = {
          qs: t.qs,
          sc: t.scales,
          dims: uniformBuffer(dev, [t.rows, t.cols, 1, 0]),
          dimsPre: dev.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          }),
          rows: t.rows,
          cols: t.cols,
        };
      } else if (t.data) {
        L.buf["g_" + name] = t.data;
      } else {
        throw new Error(`${name}: neither a split Q8_0 tensor nor f32 data`);
      }
    }
    L.requireWeights();
    return L;
  }

  /**
   * Every tensor `encode()` will look up, checked once at construction.
   *
   * A missing weight is otherwise an `undefined` passed to createBindGroup deep
   * inside a dispatch, which reports a binding error naming a slot number rather
   * than a tensor -- or, for a norm gain, binds nothing and computes with garbage.
   * Naming the tensor at load time is the difference between a one-line fix and
   * bisecting a wrong model.
   */
  private requireWeights() {
    const q8 = [
      "attn_q.weight",
      "attn_k.weight",
      "attn_v.weight",
      "attn_output.weight",
      "ffn_gate.weight",
      "ffn_up.weight",
      "ffn_down.weight",
    ];
    const f32 = ["attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight"];
    const missing = [...q8.filter((n) => !this.wq[n]), ...f32.filter((n) => !this.buf["g_" + n])];
    if (missing.length) {
      throw new Error(
        `layer is missing ${missing.length} tensor(s): ` +
          missing.join(", "),
      );
    }
  }

  static async create(
    dev: GPUDevice,
    w: LayerWeights,
    cfg: LayerConfig = QWEN3_06B,
  ): Promise<Layer> {
    const L = await Layer.build(dev, cfg);
    // Weights: repack each Q8_0 tensor into split qs/scales once, here.
    //
    // Two dims uniforms per weight, not one: the matvec's n_tokens differs between
    // decode (1) and prefill (N), and a uniform that is REWRITTEN per call would
    // invalidate nothing but would serialize -- queue.writeBuffer before a dispatch
    // that reads it is ordered, but doing it 7 times per layer per prefill chunk is
    // 196 host calls. Two immutable buffers cost 32 bytes and keep both bind groups
    // permanently cacheable, which is worth ~4 ms/layer (see the bind-group note).
    for (const [name, t] of Object.entries(w.q8)) {
      const { qs, scales } = splitQ8(t.packed, t.rows, t.cols);
      L.wq[name] = {
        qs: storageBuffer(dev, qs),
        sc: storageBuffer(dev, scales),
        dims: uniformBuffer(dev, [t.rows, t.cols, 1, 0]),
        dimsPre: dev.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        rows: t.rows,
        cols: t.cols,
      };
    }
    for (const [name, v] of Object.entries(w.f32)) {
      L.buf["g_" + name] = storageBuffer(dev, v);
    }
    L.requireWeights();
    return L;
  }

  /**
   * Everything a layer needs that is not a weight: pipelines, activation buffers,
   * the KV cache and the uniforms.
   *
   * Split out so the two weight paths -- packed CPU bytes through splitQ8, or
   * buffers the streaming loader already filled -- share one definition of the
   * layer itself. A second copy of the buffer sizing is a second place for
   * maxPrefill to be wrong.
   */
  private static async build(dev: GPUDevice, cfg: LayerConfig): Promise<Layer> {
    const L = new Layer(dev, cfg);
    const unpack8 = await probeUnpack(dev);
    const unpackF16 = await probeUnpackF16(dev);
    const src = wgslSource;
    const mk = (code: string, entryPoint = "main") =>
      dev.createComputePipeline({
        layout: "auto",
        compute: { module: dev.createShaderModule({ code }), entryPoint },
      });
    const ew = await src("elementwise.wgsl");
    L.pipes = {
      matvec: mk(coopSource(await src("q8_coop.wgsl"), { unpack8, unpackF16 })),
      rmsnorm: mk(await src("rmsnorm.wgsl")),
      rope: mk(ropeSource(await src("rope.wgsl"), cfg.ropePairing)),
      // The scores array is sized to maxKeys, not to a fixed maximum: workgroup
      // memory caps occupancy, and declaring the full 16 KB made attention cost
      // 328 us instead of 21 us. Measured; see attention.wgsl.
      attn: mk(attnSource(await src("attention.wgsl"), cfg.maxKeys)),
      // Prefill attention is a SEPARATE kernel, not the decode one run N times.
      // Its softmax is streaming, so it holds O(TILE) scores instead of O(n_keys)
      // and has no maxKeys cap -- see attention_prefill.wgsl. Decode keeps the
      // one-pass kernel because at n_keys in the low thousands the single pass is
      // strictly less work and the cap is not yet binding.
      attnPre: mk(await src("attention_prefill.wgsl")),
      swiglu: mk(ew, "swiglu"),
      add: mk(ew, "add"),
    };

    const { hidden, nHeads, nKvHeads, headDim, ffn, maxKeys } = cfg;
    const P = Math.max(1, cfg.maxPrefill);
    const rw = (n: number) =>
      dev.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    // WebGPU forbids binding one buffer as both `read` and `read_write` in a
    // single dispatch, so no op can write its own input. Every stage therefore
    // has a distinct destination -- qn/kn for the per-head norms, h2 for the
    // residual -- rather than working in place. This is not a micro-optimization
    // being skipped: the in-place version is a validation error, not merely slow.
    //
    // Every activation buffer is maxPrefill-wide, and one buffer set serves both
    // paths: decode is the N=1 case and uses the first slice of each. That is why
    // there is one Layer class and not two -- the KV cache, the bind-group cache
    // and the weights are shared, and prefill differs only in the dispatch sizes
    // and in which attention kernel runs.
    L.buf.h = rw(P * hidden); // hidden state in / residual source
    L.buf.h2 = rw(P * hidden); // after the attention residual
    L.buf.out = rw(P * hidden); // after the ffn residual: the layer's output
    L.buf.nh = rw(P * hidden); // normalized hidden
    L.buf.q = rw(P * nHeads * headDim);
    L.buf.qn = rw(P * nHeads * headDim); // q after per-head norm
    L.buf.k = rw(P * nKvHeads * headDim);
    L.buf.kn = rw(P * nKvHeads * headDim); // k after per-head norm
    L.buf.v = rw(P * nKvHeads * headDim);
    L.buf.attn = rw(P * nHeads * headDim);
    L.buf.proj = rw(P * hidden);
    L.buf.gate = rw(P * ffn);
    L.buf.up = rw(P * ffn);
    L.buf.act = rw(P * ffn);
    // KV cache, laid out [key][kv_head][head_dim] so appending a token is one
    // contiguous write -- which is what a decode step does every time.
    L.buf.kcache = rw(maxKeys * nKvHeads * headDim);
    L.buf.vcache = rw(maxKeys * nKvHeads * headDim);
    L.buf.invFreq = storageBuffer(dev, invFreqOf(headDim, cfg.ropeBase));

    // Uniforms that never change. eps is an f32 in a u32 array, so it is written
    // as raw bits at its byte offset.
    const normDims = (n: number, nVecs: number) => {
      const b = dev.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(b, 0, new Uint32Array([n, nVecs, 0, 0]));
      dev.queue.writeBuffer(b, 8, new Float32Array([cfg.eps]));
      return b;
    };
    L.buf.d_norm_hidden = normDims(hidden, 1);
    L.buf.d_norm_q = normDims(headDim, nHeads);
    L.buf.d_norm_k = normDims(headDim, nKvHeads);
    L.buf.d_ew_hidden = uniformBuffer(dev, [hidden, 0, 0, 0]);
    L.buf.d_ew_ffn = uniformBuffer(dev, [ffn, 0, 0, 0]);
    // Rewritten every step, never reallocated -- see the bind-group cache note.
    const posU = () =>
      dev.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    L.buf.d_rope_q = posU();
    L.buf.d_rope_k = posU();
    L.buf.d_attn = posU();
    // The prefill counterparts of every shape-carrying uniform. They are rewritten
    // once per chunk (not per op and not per token), so the cost is 6 writeBuffers
    // per layer per chunk regardless of N, and their bind groups stay cached.
    //
    // RMSNorm's n_vecs is where prefill actually shows up in the norms: the hidden
    // norm becomes N vectors of `hidden`, and the per-head q/k norms become
    // N*nHeads and N*nKvHeads vectors of headDim. One dispatch each, still.
    L.buf.dp_norm_hidden = posU();
    L.buf.dp_norm_q = posU();
    L.buf.dp_norm_k = posU();
    L.buf.dp_ew_hidden = posU();
    L.buf.dp_ew_ffn = posU();
    L.buf.dp_attn = dev.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // eps never changes, so write it into the prefill norm uniforms once here; only
    // the n/n_vecs words are rewritten per chunk.
    for (const b of [L.buf.dp_norm_hidden, L.buf.dp_norm_q, L.buf.dp_norm_k]) {
      dev.queue.writeBuffer(b, 8, new Float32Array([cfg.eps]));
    }
    for (const [name, pipe] of Object.entries(L.pipes)) L.pipeKey.set(pipe, name);
    return L;
  }

  /** Reset the KV cache, e.g. for a new sequence. */
  reset() {
    this.nKeys = 0;
  }

  /**
   * Bind groups, memoized by pipeline and buffer identity.
   *
   * Not a micro-optimization: creating them per step cost 4 ms of the 5.4 ms a
   * chained layer step took. They are pure functions of (pipeline, buffers), and
   * every buffer this layer uses is allocated once in create(), so every bind
   * group can be built once too. The three position-carrying uniforms are
   * REWRITTEN each step rather than reallocated, which is what keeps their bind
   * groups cacheable as well.
   */
  private bgCache = new Map<string, GPUBindGroup>();

  private bind(pipe: GPUComputePipeline, bufs: GPUBuffer[]) {
    let key = this.pipeKey.get(pipe) ?? "?";
    for (const b of bufs) {
      let id = this.bufId.get(b);
      if (id === undefined) {
        id = this.bufId.size;
        this.bufId.set(b, id);
      }
      key += "/" + id;
    }
    let bg = this.bgCache.get(key);
    if (!bg) {
      bg = this.dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      this.bgCache.set(key, bg);
    }
    return bg;
  }

  private bufId = new Map<GPUBuffer, number>();
  private pipeKey = new Map<GPUComputePipeline, string>();

  /**
   * One projection. `nTokens > 1` dispatches the batched form: the token index
   * rides on workgroup_id.y, so N tokens are one dispatch rather than N.
   */
  private matvec(
    p: GPUComputePassEncoder,
    wName: string,
    xBuf: GPUBuffer,
    outBuf: GPUBuffer,
    nTokens = 1,
  ) {
    const w = this.wq[wName];
    const pipe = this.pipes.matvec;
    p.setPipeline(pipe);
    p.setBindGroup(
      0,
      this.bind(pipe, [w.qs, w.sc, xBuf, outBuf, nTokens === 1 ? w.dims : w.dimsPre]),
    );
    p.dispatchWorkgroups(Math.ceil(w.rows / ROWS_PER_WG), nTokens);
  }

  private norm(
    p: GPUComputePassEncoder,
    xBuf: GPUBuffer,
    gain: GPUBuffer,
    outBuf: GPUBuffer,
    dims: GPUBuffer,
    nVecs: number,
  ) {
    const pipe = this.pipes.rmsnorm;
    p.setPipeline(pipe);
    p.setBindGroup(0, this.bind(pipe, [xBuf, gain, outBuf, dims]));
    p.dispatchWorkgroups(nVecs);
  }

  private elementwise(
    p: GPUComputePassEncoder,
    kind: "swiglu" | "add",
    a: GPUBuffer,
    b: GPUBuffer,
    out: GPUBuffer,
    dims: GPUBuffer,
    n: number,
  ) {
    const pipe = this.pipes[kind];
    p.setPipeline(pipe);
    p.setBindGroup(0, this.bind(pipe, [a, b, out, dims]));
    p.dispatchWorkgroups(Math.ceil(n / 256));
  }

  /**
   * One decode step: consume `hidden` (length cfg.hidden) at position
   * `this.nKeys`, append this token's k/v to the cache, and return the next
   * hidden state.
   *
   * Everything is encoded into ONE command buffer and one compute pass. Within a
   * pass WebGPU gives no ordering guarantee between dispatches in general, but it
   * does guarantee that dispatches in the same pass execute in submission order
   * with respect to storage-buffer writes on the same queue -- which is what a
   * chain of dependent ops relies on. Splitting into per-op passes would be
   * clearly safe and cost ~17.6 us each, so it is the fallback if a backend ever
   * disagrees; test_layer.ts is what would catch that.
   */
  async forward(hidden: Float32Array): Promise<Float32Array> {
    this.encode(hidden);
    return await this.readOutput();
  }

  /**
   * Encode one decode step into the queue WITHOUT reading the result back.
   *
   * This is the form a real engine wants. The map readback that `forward` does is
   * ~24 ms on this backend -- measured, and it dominates everything: a step whose
   * arithmetic is ~0.2 ms takes 29 ms end to end, so 99% of a forward() call is
   * the host waiting to see a number it does not need yet. Chaining layers means
   * calling encode() per layer and reading back once, at the end of the shard.
   *
   * `hidden` is optional: omit it to consume whatever is already in the output
   * buffer, which is how one layer feeds the next without a host round-trip.
   */
  encode(hidden?: Float32Array, encoder?: GPUCommandEncoder): void {
    const { dev, cfg, buf } = this;
    const { nHeads, nKvHeads, headDim, ffn } = cfg;
    const pos = this.nKeys;
    if (pos >= cfg.maxKeys) throw new Error(`KV cache full at ${cfg.maxKeys} keys`);

    if (hidden) writeView(dev, buf.h, 0, hidden);

    // Position-dependent uniforms are REWRITTEN, not reallocated. Allocating them
    // per step would make their bind groups uncacheable, which is the expensive
    // part; the write itself is ~10 us for all three.
    dev.queue.writeBuffer(buf.d_rope_q, 0, new Uint32Array([1, nHeads, headDim, pos]));
    dev.queue.writeBuffer(buf.d_rope_k, 0, new Uint32Array([1, nKvHeads, headDim, pos]));
    dev.queue.writeBuffer(buf.d_attn, 0, new Uint32Array([nHeads, nKvHeads, headDim, pos + 1]));

    // The caller may pass an encoder so that several layers -- or several decode
    // steps -- share one command buffer. That is worth a lot: 28 steps in one
    // command buffer cost 1.39 ms each, versus 5.43 ms each in their own.
    const enc = encoder ?? dev.createCommandEncoder();
    const p = enc.beginComputePass();

    // --- attention block ---------------------------------------------------
    this.norm(p, buf.h, buf["g_attn_norm.weight"], buf.nh, buf.d_norm_hidden, 1);
    this.matvec(p, "attn_q.weight", buf.nh, buf.q);
    this.matvec(p, "attn_k.weight", buf.nh, buf.k);
    this.matvec(p, "attn_v.weight", buf.nh, buf.v);

    // Per-head RMSNorm on q and k -- the Qwen3-specific step. The gain is headDim
    // long and applies to every head independently, so this is one dispatch of
    // n_heads workgroups. Destination differs from source: see the buffer note.
    this.norm(p, buf.q, buf["g_attn_q_norm.weight"], buf.qn, buf.d_norm_q, nHeads);
    this.norm(p, buf.k, buf["g_attn_k_norm.weight"], buf.kn, buf.d_norm_k, nKvHeads);

    // RoPE, after the per-head norm, on q and k only -- never on v. This one IS
    // in place, and legally so: rope.wgsl declares x as read_write and reads no
    // other storage buffer, so there is no read/read_write conflict.
    const rope = (dims: GPUBuffer, x: GPUBuffer, heads: number) => {
      p.setPipeline(this.pipes.rope);
      p.setBindGroup(0, this.bind(this.pipes.rope, [x, dims, buf.invFreq]));
      p.dispatchWorkgroups(Math.ceil((heads * headDim / 2) / 64));
    };
    rope(buf.d_rope_q, buf.qn, nHeads);
    rope(buf.d_rope_k, buf.kn, nKvHeads);
    p.end();

    // Append this token's k/v to the cache. A buffer-to-buffer copy cannot be
    // encoded inside a compute pass, so the pass is split here rather than
    // writing a copy kernel. The cache is [key][kv_head][head_dim], so one
    // token's k is a single contiguous run at offset pos * kvBytes.
    const kvBytes = nKvHeads * headDim * 4;
    enc.copyBufferToBuffer(buf.kn, 0, buf.kcache, pos * kvBytes, kvBytes);
    enc.copyBufferToBuffer(buf.v, 0, buf.vcache, pos * kvBytes, kvBytes);

    const p2 = enc.beginComputePass();
    p2.setPipeline(this.pipes.attn);
    p2.setBindGroup(
      0,
      this.bind(this.pipes.attn, [buf.qn, buf.kcache, buf.vcache, buf.attn, buf.d_attn]),
    );
    p2.dispatchWorkgroups(nHeads);

    // Output projection and the first residual. h2 = h + proj.
    this.matvec(p2, "attn_output.weight", buf.attn, buf.proj);
    this.elementwise(p2, "add", buf.h, buf.proj, buf.h2, buf.d_ew_hidden, cfg.hidden);

    // --- feed-forward block ------------------------------------------------
    // Pre-norm: the residual adds to h2 (the block's input), not to its norm.
    this.norm(p2, buf.h2, buf["g_ffn_norm.weight"], buf.nh, buf.d_norm_hidden, 1);
    this.matvec(p2, "ffn_gate.weight", buf.nh, buf.gate);
    this.matvec(p2, "ffn_up.weight", buf.nh, buf.up);
    this.elementwise(p2, "swiglu", buf.gate, buf.up, buf.act, buf.d_ew_ffn, ffn);
    this.matvec(p2, "ffn_down.weight", buf.act, buf.proj);
    this.elementwise(p2, "add", buf.h2, buf.proj, buf.out, buf.d_ew_hidden, cfg.hidden);
    p2.end();

    // The layer's output becomes the next step's input. Copying out -> h here
    // rather than swapping the two means the caller can chain encode() calls with
    // no argument and no host involvement.
    enc.copyBufferToBuffer(buf.out, 0, buf.h, 0, cfg.hidden * 4);
    // Only submit if we own the encoder. When the caller supplied one, they
    // decide when to submit -- which is the whole point of sharing it.
    if (!encoder) dev.queue.submit([enc.finish()]);
    this.nKeys = pos + 1;
  }

  /**
   * Encode a PREFILL of `nTokens` positions starting at `this.nKeys`.
   *
   * Same op sequence as decode -- it is the same layer -- with four differences,
   * each of which is a place a prefill implementation can be quietly wrong:
   *
   *  1. Every matvec is batched (workgroup_id.y over tokens), so the seven
   *     projections are seven dispatches and not 7N.
   *  2. RMSNorm's n_vecs is multiplied by nTokens. The per-head q/k norms become
   *     N*nHeads and N*nKvHeads independent 128-wide vectors -- still one dispatch
   *     each, because the kernel was written around "n_vecs vectors of width n"
   *     for exactly this reason.
   *  3. RoPE gets n_tokens = N and pos0 = this.nKeys, so token i is rotated by
   *     position nKeys + i. This is the one place where a prefill bug is invisible
   *     in the output magnitude: every token rotated by the same angle is a
   *     self-consistent, wrong model.
   *  4. Attention is the streaming-softmax prefill kernel over an N x (pos+N)
   *     causal region, dispatched (nHeads, N). Decode's kernel would compute the
   *     wrong thing here, not just slowly: it has one query and no causal bound
   *     per query.
   *
   * The KV append is still a single contiguous copy. kn/v hold the batch as
   * [token][kv_head][head_dim] and the cache is [key][kv_head][head_dim], so N
   * tokens land as one run of N*kvBytes at pos*kvBytes -- the layouts agree by
   * construction, which is why prefill needs no scatter kernel.
   *
   * `hidden` is N*cfg.hidden long, or omitted to consume what is already in the
   * buffer (how one layer feeds the next with no host round-trip).
   */
  encodePrefill(nTokens: number, hidden?: Float32Array, encoder?: GPUCommandEncoder): void {
    const { dev, cfg, buf } = this;
    const { nHeads, nKvHeads, headDim, ffn, hidden: H } = cfg;
    const pos = this.nKeys;
    if (nTokens < 1) throw new Error(`nTokens must be >= 1, got ${nTokens}`);
    if (nTokens > cfg.maxPrefill) {
      throw new Error(`prefill of ${nTokens} exceeds maxPrefill ${cfg.maxPrefill}; chunk it`);
    }
    if (pos + nTokens > cfg.maxKeys) {
      throw new Error(`KV cache would exceed ${cfg.maxKeys} keys (${pos} + ${nTokens})`);
    }
    if (hidden && hidden.length !== nTokens * H) {
      throw new Error(`hidden is ${hidden.length}, expected ${nTokens * H}`);
    }
    if (hidden) writeView(dev, buf.h, 0, hidden);

    // Per-chunk uniform rewrites: 6 + 7 writes, independent of N. The matvec dims
    // differ per weight only in rows/cols, so each weight's own dimsPre is written.
    const u32 = (...v: number[]) => new Uint32Array(v);
    dev.queue.writeBuffer(buf.dp_norm_hidden, 0, u32(H, nTokens));
    dev.queue.writeBuffer(buf.dp_norm_q, 0, u32(headDim, nTokens * nHeads));
    dev.queue.writeBuffer(buf.dp_norm_k, 0, u32(headDim, nTokens * nKvHeads));
    dev.queue.writeBuffer(buf.dp_ew_hidden, 0, u32(nTokens * H, 0, 0, 0));
    dev.queue.writeBuffer(buf.dp_ew_ffn, 0, u32(nTokens * ffn, 0, 0, 0));
    dev.queue.writeBuffer(buf.dp_attn, 0, u32(nHeads, nKvHeads, headDim, nTokens, pos, 0, 0, 0));
    dev.queue.writeBuffer(buf.d_rope_q, 0, u32(nTokens, nHeads, headDim, pos));
    dev.queue.writeBuffer(buf.d_rope_k, 0, u32(nTokens, nKvHeads, headDim, pos));
    for (const name of Object.keys(this.wq)) {
      const w = this.wq[name];
      dev.queue.writeBuffer(w.dimsPre, 0, u32(w.rows, w.cols, nTokens, 0));
    }

    const enc = encoder ?? dev.createCommandEncoder();
    const p = enc.beginComputePass();

    // --- attention block ---------------------------------------------------
    this.norm(p, buf.h, buf["g_attn_norm.weight"], buf.nh, buf.dp_norm_hidden, nTokens);
    this.matvec(p, "attn_q.weight", buf.nh, buf.q, nTokens);
    this.matvec(p, "attn_k.weight", buf.nh, buf.k, nTokens);
    this.matvec(p, "attn_v.weight", buf.nh, buf.v, nTokens);
    this.norm(p, buf.q, buf["g_attn_q_norm.weight"], buf.qn, buf.dp_norm_q, nTokens * nHeads);
    this.norm(p, buf.k, buf["g_attn_k_norm.weight"], buf.kn, buf.dp_norm_k, nTokens * nKvHeads);

    const rope = (dims: GPUBuffer, x: GPUBuffer, heads: number) => {
      p.setPipeline(this.pipes.rope);
      p.setBindGroup(0, this.bind(this.pipes.rope, [x, dims, buf.invFreq]));
      p.dispatchWorkgroups(Math.ceil((nTokens * heads * headDim / 2) / 64));
    };
    rope(buf.d_rope_q, buf.qn, nHeads);
    rope(buf.d_rope_k, buf.kn, nKvHeads);
    p.end();

    // N tokens of k/v as ONE contiguous copy: the batch layout and the cache
    // layout are both [*][kv_head][head_dim], so they concatenate.
    const kvBytes = nKvHeads * headDim * 4;
    enc.copyBufferToBuffer(buf.kn, 0, buf.kcache, pos * kvBytes, nTokens * kvBytes);
    enc.copyBufferToBuffer(buf.v, 0, buf.vcache, pos * kvBytes, nTokens * kvBytes);

    const p2 = enc.beginComputePass();
    p2.setPipeline(this.pipes.attnPre);
    p2.setBindGroup(
      0,
      this.bind(this.pipes.attnPre, [buf.qn, buf.kcache, buf.vcache, buf.attn, buf.dp_attn]),
    );
    // (head, query): one workgroup per pair, which is what keeps prefill wide.
    p2.dispatchWorkgroups(nHeads, nTokens);

    this.matvec(p2, "attn_output.weight", buf.attn, buf.proj, nTokens);
    this.elementwise(p2, "add", buf.h, buf.proj, buf.h2, buf.dp_ew_hidden, nTokens * H);

    // --- feed-forward block ------------------------------------------------
    this.norm(p2, buf.h2, buf["g_ffn_norm.weight"], buf.nh, buf.dp_norm_hidden, nTokens);
    this.matvec(p2, "ffn_gate.weight", buf.nh, buf.gate, nTokens);
    this.matvec(p2, "ffn_up.weight", buf.nh, buf.up, nTokens);
    this.elementwise(p2, "swiglu", buf.gate, buf.up, buf.act, buf.dp_ew_ffn, nTokens * ffn);
    this.matvec(p2, "ffn_down.weight", buf.act, buf.proj, nTokens);
    this.elementwise(p2, "add", buf.h2, buf.proj, buf.out, buf.dp_ew_hidden, nTokens * H);
    p2.end();

    enc.copyBufferToBuffer(buf.out, 0, buf.h, 0, nTokens * H * 4);
    if (!encoder) dev.queue.submit([enc.finish()]);
    this.nKeys = pos + nTokens;
  }

  /** The buffer holding this layer's output, for a caller chaining layers itself. */
  outputBuffer(): GPUBuffer {
    return this.buf.out;
  }

  /**
   * The buffer this layer reads its input from.
   *
   * Exposed so a Model can chain 28 layers with device-to-device copies inside one
   * command buffer, instead of passing a Float32Array per layer -- which would mean
   * a readback and an upload per layer, ~24 ms each on this backend against a
   * whole-token budget of a few ms.
   */
  inputBuffer(): GPUBuffer {
    return this.buf.h;
  }

  /**
   * The KV cache buffers. Exposed so a test can assert that a prefill left the
   * cache BIT-IDENTICAL to what the equivalent decode steps would have written --
   * a prefill can produce the right hidden states and still corrupt the cache,
   * which then breaks every token generated afterwards instead of the prefill.
   */
  cacheBuffers(): { k: GPUBuffer; v: GPUBuffer } {
    return { k: this.buf.kcache, v: this.buf.vcache };
  }

  /**
   * Read the hidden state back to the host. This is the expensive part (~24 ms).
   *
   * `nTokens` reads a whole prefill batch; the default of 1 is the decode case.
   */
  async readOutput(nTokens = 1): Promise<Float32Array> {
    const { dev, cfg, buf } = this;
    const bytes = nTokens * cfg.hidden * 4;
    const rd = dev.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(buf.out, 0, rd, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rd.getMappedRange().slice(0));
    rd.unmap();
    rd.destroy();
    return out;
  }
}
