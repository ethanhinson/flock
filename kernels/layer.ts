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
  attnSource, coopSource, probeUnpack, probeUnpackF16, splitQ8, storageBuffer,
  uniformBuffer,
} from "./lib.ts";

export interface LayerWeights {
  /** Q8_0 packed, straight from the GGUF file, keyed by tensor suffix. */
  q8: Record<string, { rows: number; cols: number; packed: Uint8Array }>;
  /** f32 norm gains, keyed by tensor suffix. */
  f32: Record<string, Float32Array>;
}

export interface LayerConfig {
  hidden: number;      // 1024
  nHeads: number;      // 16
  nKvHeads: number;    // 8
  headDim: number;     // 128
  ffn: number;         // 3072
  eps: number;         // 1e-6
  ropeBase: number;    // 1e6
  maxKeys: number;     // KV cache capacity
}

// maxKeys is 2048, not the 4096 that 16 KB of workgroup memory allows, and that
// is deliberate. 16 KB is exactly the point where only one workgroup fits per
// core, so attention loses all latency hiding: measured 328 us per dispatch at
// 4096 versus 21 us at 2048, a 15x cliff for 2x the context. Raise it only with
// that number in hand.
export const QWEN3_06B: LayerConfig = {
  hidden: 1024, nHeads: 16, nKvHeads: 8, headDim: 128, ffn: 3072,
  eps: 9.999999974752427e-7, ropeBase: 1e6, maxKeys: 2048,
};

const ROWS_PER_WG = 4;     // must match q8_coop.wgsl
const COOP_LANES = 64;     // must match q8_coop.wgsl

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
  private wq: Record<string, { qs: GPUBuffer; sc: GPUBuffer; dims: GPUBuffer; rows: number }> = {};
  /** How many keys are currently in the cache. */
  nKeys = 0;

  private constructor(dev: GPUDevice, cfg: LayerConfig) {
    this.dev = dev;
    this.cfg = cfg;
  }

  static async create(
    dev: GPUDevice, w: LayerWeights, cfg: LayerConfig = QWEN3_06B,
  ): Promise<Layer> {
    const L = new Layer(dev, cfg);
    const unpack8 = await probeUnpack(dev);
    const unpackF16 = await probeUnpackF16(dev);
    const src = (f: string) => Deno.readTextFile(new URL("./" + f, import.meta.url));
    const mk = (code: string, entryPoint = "main") => dev.createComputePipeline({
      layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint },
    });
    const ew = await src("elementwise.wgsl");
    L.pipes = {
      matvec: mk(coopSource(await src("q8_coop.wgsl"), { unpack8, unpackF16 })),
      rmsnorm: mk(await src("rmsnorm.wgsl")),
      rope: mk(await src("rope.wgsl")),
      // The scores array is sized to maxKeys, not to a fixed maximum: workgroup
      // memory caps occupancy, and declaring the full 16 KB made attention cost
      // 328 us instead of 21 us. Measured; see attention.wgsl.
      attn: mk(attnSource(await src("attention.wgsl"), cfg.maxKeys)),
      swiglu: mk(ew, "swiglu"),
      add: mk(ew, "add"),
    };

    // Weights: repack each Q8_0 tensor into split qs/scales once, here.
    for (const [name, t] of Object.entries(w.q8)) {
      const { qs, scales } = splitQ8(t.packed, t.rows, t.cols);
      L.wq[name] = {
        qs: storageBuffer(dev, qs), sc: storageBuffer(dev, scales),
        dims: uniformBuffer(dev, [t.rows, t.cols, 0, 0]), rows: t.rows,
      };
    }
    for (const [name, v] of Object.entries(w.f32)) L.buf["g_" + name] = storageBuffer(dev, v);

    const { hidden, nHeads, nKvHeads, headDim, ffn, maxKeys } = cfg;
    const rw = (n: number) => dev.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // WebGPU forbids binding one buffer as both `read` and `read_write` in a
    // single dispatch, so no op can write its own input. Every stage therefore
    // has a distinct destination -- qn/kn for the per-head norms, h2 for the
    // residual -- rather than working in place. This is not a micro-optimization
    // being skipped: the in-place version is a validation error, not merely slow.
    L.buf.h = rw(hidden);          // hidden state in / residual source
    L.buf.h2 = rw(hidden);         // after the attention residual
    L.buf.out = rw(hidden);        // after the ffn residual: the layer's output
    L.buf.nh = rw(hidden);         // normalized hidden
    L.buf.q = rw(nHeads * headDim);
    L.buf.qn = rw(nHeads * headDim);     // q after per-head norm
    L.buf.k = rw(nKvHeads * headDim);
    L.buf.kn = rw(nKvHeads * headDim);   // k after per-head norm
    L.buf.v = rw(nKvHeads * headDim);
    L.buf.attn = rw(nHeads * headDim);
    L.buf.proj = rw(hidden);
    L.buf.gate = rw(ffn);
    L.buf.up = rw(ffn);
    L.buf.act = rw(ffn);
    // KV cache, laid out [key][kv_head][head_dim] so appending a token is one
    // contiguous write -- which is what a decode step does every time.
    L.buf.kcache = rw(maxKeys * nKvHeads * headDim);
    L.buf.vcache = rw(maxKeys * nKvHeads * headDim);
    L.buf.invFreq = storageBuffer(dev, invFreqOf(headDim, cfg.ropeBase));

    // Uniforms that never change. eps is an f32 in a u32 array, so it is written
    // as raw bits at its byte offset.
    const normDims = (n: number, nVecs: number) => {
      const b = dev.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
    const posU = () => dev.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    L.buf.d_rope_q = posU();
    L.buf.d_rope_k = posU();
    L.buf.d_attn = posU();
    for (const [name, pipe] of Object.entries(L.pipes)) L.pipeKey.set(pipe, name);
    return L;
  }

  /** Reset the KV cache, e.g. for a new sequence. */
  reset() { this.nKeys = 0; }

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
      if (id === undefined) { id = this.bufId.size; this.bufId.set(b, id); }
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

  private matvec(
    p: GPUComputePassEncoder, wName: string, xBuf: GPUBuffer, outBuf: GPUBuffer,
  ) {
    const w = this.wq[wName];
    const pipe = this.pipes.matvec;
    p.setPipeline(pipe);
    p.setBindGroup(0, this.bind(pipe, [w.qs, w.sc, xBuf, outBuf, w.dims]));
    p.dispatchWorkgroups(Math.ceil(w.rows / ROWS_PER_WG));
  }

  private norm(
    p: GPUComputePassEncoder, xBuf: GPUBuffer, gain: GPUBuffer, outBuf: GPUBuffer,
    dims: GPUBuffer, nVecs: number,
  ) {
    const pipe = this.pipes.rmsnorm;
    p.setPipeline(pipe);
    p.setBindGroup(0, this.bind(pipe, [xBuf, gain, outBuf, dims]));
    p.dispatchWorkgroups(nVecs);
  }

  private elementwise(
    p: GPUComputePassEncoder, kind: "swiglu" | "add", a: GPUBuffer, b: GPUBuffer,
    out: GPUBuffer, dims: GPUBuffer, n: number,
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

    if (hidden) dev.queue.writeBuffer(buf.h, 0, hidden);

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
    p2.setBindGroup(0, this.bind(this.pipes.attn,
      [buf.qn, buf.kcache, buf.vcache, buf.attn, buf.d_attn]));
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

  /** Read the current hidden state back to the host. This is the expensive part. */
  async readOutput(): Promise<Float32Array> {
    const { dev, cfg, buf } = this;
    const rd = dev.createBuffer({
      size: cfg.hidden * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(buf.out, 0, rd, 0, cfg.hidden * 4);
    dev.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rd.getMappedRange().slice(0));
    rd.unmap(); rd.destroy();
    return out;
  }
}
