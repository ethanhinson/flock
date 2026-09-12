// Model: a whole Qwen3 forward pass in WGSL, from GGUF weights. No ONNX.
//
//   token ids -> embedding -> 28 layers -> output_norm -> logits -> argmax -> id
//
// Everything between the ids going in and the id coming out stays on the GPU. The
// host sees one number per generated token, and that is the entire point of the
// shape of this class: a map readback is ~24 ms on this backend, so a design that
// read the hidden state between layers would spend 28 * 24 ms per token on
// nothing. `step()` reads back 4 bytes; `logits()` exists for tests and pays the
// full price knowingly.
//
// THE TIED LM HEAD. Qwen3-0.6B sets tie_word_embeddings, so the vocab projection
// IS token_embd.weight, read as 151936 rows of 1024 columns -- which is exactly
// the [rows][cols] layout the matvec kernel already wants. So the LM head is not a
// new kernel: it is q8_coop.wgsl at (151936 x 1024), the largest single dispatch
// in the model by a wide margin (37984 workgroups against 768 for ffn_gate), and
// the embedding lookup is a gather over the same buffer. One repacked upload,
// 155.6 MB, serves both. Uploading it twice, or dequantizing it, would be 622 MB
// for the dequantized copy alone.
//
// That one buffer is also why this needs a device with raised limits: the split
// `qs` array is 155.6 MB and the DEFAULT maxStorageBufferBindingSize is 128 MiB.
// lib.ts getDevice() asks for the adapter's real limits; see the note there,
// including why an oversized binding is a silent wrong answer rather than a throw.
//
// WHERE output_norm GOES, because the ONNX export puts it somewhere surprising and
// copying that layout would be wrong. flock_export_coordinator.py builds
// head.onnx as `Head(model.model.norm, model.lm_head)` whose forward is
// `self.head(hidden).argmax(-1)` -- it stores the norm and never calls it. The
// final RMSNorm is instead applied by the LAST SHARD (flock_export.py, `hidden =
// self.norm(hidden) if self.is_last`). So the real ONNX pipeline is
// embed -> layers -> [last shard applies output_norm] -> lm_head -> argmax, and
// that is what this reproduces: output_norm here, before the projection, exactly
// once.
//
// PREFILL vs DECODE. A prompt goes through encodePrefill in chunks of maxPrefill;
// generation then goes one token at a time through encode(). Only the LAST
// position of a prefill needs logits -- the earlier ones are just there to fill
// the KV cache -- so the head runs on one hidden state regardless of prompt
// length, and `lastRow` is what selects it.

import {
  attnSource, coopSource, probeUnpack, probeUnpackF16, splitQ8, storageBuffer,
  uniformBuffer,
} from "./lib.ts";
import { Layer, QWEN3_06B, type LayerConfig, type LayerWeights } from "./layer.ts";
import type { RealModel } from "./real_weights.ts";

const ROWS_PER_WG = 4;      // must match q8_coop.wgsl
const ARGMAX_WG = 256;      // must match argmax.wgsl
/**
 * Stage-1 workgroups for the argmax. 594 is ceil(151936 / 256), i.e. one element
 * per thread with no strided loop -- the point where stage 1 stops being a
 * reduction and becomes a single comparison per thread, and where stage 2 still
 * fits in one workgroup (594 partials, 256 threads, a 3-iteration strided load).
 */
const ARGMAX_GROUPS = 594;

export interface ModelConfig extends LayerConfig {
  vocab: number;
  nLayers: number;
}

export class Model {
  private dev: GPUDevice;
  private cfg: ModelConfig;
  private layers: Layer[] = [];
  private pipes!: Record<string, GPUComputePipeline>;
  private buf: Record<string, GPUBuffer> = {};
  private bgCache = new Map<string, GPUBindGroup>();
  private bufId = new Map<GPUBuffer, number>();
  private pipeKey = new Map<GPUComputePipeline, string>();
  /** Absolute position of the next token; the KV cache length. */
  pos = 0;

  private constructor(dev: GPUDevice, cfg: ModelConfig) {
    this.dev = dev;
    this.cfg = cfg;
  }

  static async create(
    dev: GPUDevice, m: RealModel, base: LayerConfig = QWEN3_06B,
  ): Promise<Model> {
    const cfg: ModelConfig = { ...base, vocab: m.vocab, nLayers: m.nLayers };
    if (m.hidden !== cfg.hidden) {
      throw new Error(`model hidden ${m.hidden} != config ${cfg.hidden}`);
    }
    if (m.embd.cols !== cfg.hidden) {
      throw new Error(`token_embd cols ${m.embd.cols} != hidden ${cfg.hidden}`);
    }
    const M = new Model(dev, cfg);
    const unpack8 = await probeUnpack(dev);
    const unpackF16 = await probeUnpackF16(dev);
    const src = (f: string) => Deno.readTextFile(new URL("./" + f, import.meta.url));
    const mk = (code: string, entryPoint = "main") => dev.createComputePipeline({
      layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint },
    });
    const argmaxSrc = await src("argmax.wgsl");
    M.pipes = {
      embed: mk(coopSource(await src("embed.wgsl"), { unpack8, unpackF16 })),
      // The LM head reuses the layer matvec unchanged. It is the same operation at
      // a different shape, and a second copy of a bit-exact kernel is a second
      // thing to keep right.
      matvec: mk(coopSource(await src("q8_coop.wgsl"), { unpack8, unpackF16 })),
      rmsnorm: mk(await src("rmsnorm.wgsl")),
      argmax1: mk(argmaxSrc, "pass1"),
      argmax2: mk(argmaxSrc, "pass2"),
    };
    for (const [name, pipe] of Object.entries(M.pipes)) M.pipeKey.set(pipe, name);

    // ONE repack of token_embd, shared by the embedding gather and the LM head.
    const { qs, scales } = splitQ8(m.embd.packed, m.embd.rows, m.embd.cols);
    M.buf.embdQs = storageBuffer(dev, qs);
    M.buf.embdSc = storageBuffer(dev, scales);
    M.buf.outputNorm = storageBuffer(dev, m.outputNorm);

    const rw = (n: number) => dev.createBuffer({
      size: Math.max(16, n * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const P = cfg.maxPrefill;
    M.buf.ids = rw(P);                    // u32 token ids for the embedding gather
    M.buf.embOut = rw(P * cfg.hidden);    // the embedding's output / layer 0's input
    M.buf.normed = rw(cfg.hidden);        // one hidden state after output_norm
    M.buf.lastRow = rw(cfg.hidden);       // the position the head projects
    M.buf.logits = rw(cfg.vocab);
    M.buf.amVal = rw(ARGMAX_GROUPS);
    M.buf.amIdx = rw(ARGMAX_GROUPS);
    M.buf.tokVal = rw(1);
    M.buf.tokIdx = rw(1);
    // argmax pass1 declares xidx (so both passes share a bind group layout) but
    // reads nothing live from it. It still needs a buffer that pass1 does not also
    // write: read + read_write on one buffer is a validation error.
    M.buf.amDummy = rw(1);

    M.buf.dEmbed = uniformBuffer(dev, [1, cfg.hidden, 0, 0]);
    M.buf.dEmbedPre = dev.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    M.buf.dHead = uniformBuffer(dev, [cfg.vocab, cfg.hidden, 1, 0]);
    M.buf.dNorm = (() => {
      const b = dev.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(b, 0, new Uint32Array([cfg.hidden, 1, 0, 0]));
      dev.queue.writeBuffer(b, 8, new Float32Array([cfg.eps]));
      return b;
    })();
    M.buf.dAm1 = uniformBuffer(dev, [cfg.vocab, ARGMAX_GROUPS, 0, 0]);
    M.buf.dAm2 = uniformBuffer(dev, [ARGMAX_GROUPS, ARGMAX_GROUPS, 0, 0]);

    for (let i = 0; i < m.nLayers; i++) {
      M.layers.push(await Layer.create(dev, m.layers[i], cfg));
    }
    return M;
  }

  /** Start a new sequence: clear every layer's KV cache and the position. */
  reset() {
    this.pos = 0;
    for (const l of this.layers) l.reset();
  }

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

  /**
   * Push `ids` through the embedding and all 28 layers, leaving the last token's
   * hidden state ready for the head. Returns nothing: the result stays on the GPU.
   *
   * Chunked at maxPrefill so an arbitrarily long prompt works with fixed buffers.
   * Each chunk is one command buffer covering the embedding and all 28 layers --
   * one submit per chunk, not per layer, which is the 20.8x the layer benchmark
   * measured for sharing a command buffer.
   */
  private encodeTokens(ids: number[]): void {
    const { dev, cfg, buf } = this;
    for (let off = 0; off < ids.length; off += cfg.maxPrefill) {
      const chunk = ids.slice(off, off + cfg.maxPrefill);
      const n = chunk.length;
      if (this.pos + n > cfg.maxKeys) {
        throw new Error(`sequence would exceed maxKeys ${cfg.maxKeys} at ${this.pos + n}`);
      }
      dev.queue.writeBuffer(buf.ids, 0, new Uint32Array(chunk));
      dev.queue.writeBuffer(buf.dEmbedPre, 0, new Uint32Array([n, cfg.hidden, 0, 0]));

      const enc = dev.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.pipes.embed);
      p.setBindGroup(0, this.bind(this.pipes.embed, [
        buf.embdQs, buf.embdSc, buf.ids, buf.embOut,
        n === 1 ? buf.dEmbed : buf.dEmbedPre,
      ]));
      p.dispatchWorkgroups(n);
      p.end();

      // The embedding's output is the first layer's input; from then on each layer
      // consumes the previous one's, with no host involvement. The copy into
      // layer 0's `h` is a device-to-device copy of at most maxPrefill * 1024
      // floats, which is 2 MB at the default and cheap next to 28 layers of work.
      const first = this.layers[0];
      enc.copyBufferToBuffer(buf.embOut, 0, first.inputBuffer(), 0, n * cfg.hidden * 4);
      for (let i = 0; i < this.layers.length; i++) {
        const L = this.layers[i];
        // Layer i > 0 reads what layer i-1 wrote. encode/encodePrefill with no
        // hidden argument consume whatever is already in the layer's own `h`, so
        // the chain is one copy per boundary.
        if (i > 0) {
          enc.copyBufferToBuffer(
            this.layers[i - 1].outputBuffer(), 0, L.inputBuffer(), 0, n * cfg.hidden * 4);
        }
        if (n === 1) L.encode(undefined, enc);
        else L.encodePrefill(n, undefined, enc);
      }
      // Only the LAST position's hidden state is projected: the earlier ones exist
      // to fill the KV cache. So the head's input is one row, taken from the end.
      const lastOut = this.layers[this.layers.length - 1].outputBuffer();
      enc.copyBufferToBuffer(
        lastOut, (n - 1) * cfg.hidden * 4, buf.lastRow, 0, cfg.hidden * 4);
      dev.queue.submit([enc.finish()]);
      this.pos += n;
    }
  }

  /**
   * output_norm, then the tied vocab projection, then argmax -- all on the GPU.
   *
   * Argmax on the device rather than reading 151936 floats back: the reduction is
   * two ~20 us dispatches returning 8 bytes, against a ~24 ms map readback plus
   * 608 KB of copy. bench_model.ts measures both rather than asserting it.
   */
  private encodeHead(enc: GPUCommandEncoder): void {
    const { buf, cfg } = this;
    const p = enc.beginComputePass();
    // The final RMSNorm. Applied HERE, once -- see the note at the top about the
    // ONNX export putting it in the last shard rather than in head.onnx.
    p.setPipeline(this.pipes.rmsnorm);
    p.setBindGroup(0, this.bind(this.pipes.rmsnorm,
      [buf.lastRow, buf.outputNorm, buf.normed, buf.dNorm]));
    p.dispatchWorkgroups(1);

    // The tied projection: token_embd read as 151936 x 1024. 37984 workgroups.
    p.setPipeline(this.pipes.matvec);
    p.setBindGroup(0, this.bind(this.pipes.matvec,
      [buf.embdQs, buf.embdSc, buf.normed, buf.logits, buf.dHead]));
    p.dispatchWorkgroups(Math.ceil(cfg.vocab / ROWS_PER_WG), 1);

    p.setPipeline(this.pipes.argmax1);
    p.setBindGroup(0, this.bind(this.pipes.argmax1,
      [buf.logits, buf.amDummy, buf.amVal, buf.amIdx, buf.dAm1]));
    p.dispatchWorkgroups(ARGMAX_GROUPS);
    p.end();

    // Stage 2 in its own pass: it reads the buffers stage 1 wrote, through a
    // different bind group, so a pass boundary is what makes the write visible
    // without relying on in-pass ordering across bind groups.
    const p2 = enc.beginComputePass();
    p2.setPipeline(this.pipes.argmax2);
    p2.setBindGroup(0, this.bind(this.pipes.argmax2,
      [buf.amVal, buf.amIdx, buf.tokVal, buf.tokIdx, buf.dAm2]));
    p2.dispatchWorkgroups(1);
    p2.end();
  }

  /** Read `n` u32 out of a buffer. */
  private async readU32(src: GPUBuffer, n: number): Promise<Uint32Array> {
    const bytes = n * 4;
    const rd = this.dev.createBuffer({
      size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rd, 0, bytes);
    this.dev.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(rd.getMappedRange().slice(0));
    rd.unmap(); rd.destroy();
    return out;
  }

  private async readF32(src: GPUBuffer, n: number): Promise<Float32Array> {
    const bytes = n * 4;
    const rd = this.dev.createBuffer({
      size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rd, 0, bytes);
    this.dev.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rd.getMappedRange().slice(0));
    rd.unmap(); rd.destroy();
    return out;
  }

  /**
   * Feed `ids` and return the next token id, greedily.
   *
   * The host round-trip is 4 bytes. Everything else -- 151936 logits, 28 layers of
   * hidden state, the KV cache -- never leaves the GPU.
   */
  async step(ids: number[]): Promise<number> {
    this.encodeTokens(ids);
    const enc = this.dev.createCommandEncoder();
    this.encodeHead(enc);
    this.dev.queue.submit([enc.finish()]);
    return (await this.readU32(this.buf.tokIdx, 1))[0];
  }

  /**
   * The full logit vector for the last position. 608 KB and a map readback, so
   * this is for tests and for diffing against ONNX -- `step` is the path a
   * generation takes.
   */
  async logits(ids: number[]): Promise<Float32Array> {
    this.encodeTokens(ids);
    const enc = this.dev.createCommandEncoder();
    this.encodeHead(enc);
    this.dev.queue.submit([enc.finish()]);
    return await this.readF32(this.buf.logits, this.cfg.vocab);
  }

  /** The last position's hidden state after all layers, BEFORE output_norm. */
  async hiddenState(ids: number[]): Promise<Float32Array> {
    this.encodeTokens(ids);
    return await this.readF32(this.buf.lastRow, this.cfg.hidden);
  }

  /** The last position's hidden state AFTER output_norm, before the projection. */
  async normedState(ids: number[]): Promise<Float32Array> {
    this.encodeTokens(ids);
    const enc = this.dev.createCommandEncoder();
    this.encodeHead(enc);
    this.dev.queue.submit([enc.finish()]);
    return await this.readF32(this.buf.normed, this.cfg.hidden);
  }

  /**
   * K decode steps encoded behind ONE map readback, for measurement only.
   *
   * NOT a generation path, and it is important to be clear about why: greedy decode
   * needs each token id on the host before it can choose the next input, so it
   * cannot batch steps -- it pays one ~26 ms readback per token by construction.
   * This runs K steps on a FIXED input id instead, which computes the wrong tokens
   * on purpose but does exactly the right amount of GPU work, and so isolates that
   * work from the readback. bench_model.ts uses it to show what the readback costs;
   * the honest tok/s figure is the unbatched one.
   */
  async stepsAmortized(ids: number[]): Promise<number> {
    for (const id of ids) {
      this.encodeTokens([id]);
      const enc = this.dev.createCommandEncoder();
      this.encodeHead(enc);
      this.dev.queue.submit([enc.finish()]);
    }
    return (await this.readU32(this.buf.tokIdx, 1))[0];
  }

  /**
   * Drop the KV cache back to `nKeys`, so a benchmark's repeated trials all run at
   * the same context length.
   *
   * Legal because the cache is append-only: keys past `nKeys` are never read again
   * once nKeys says they are not there. Nothing is zeroed, which is why this is a
   * measurement aid and not a correctness-preserving public operation -- a
   * generation should use reset().
   */
  rewind(nKeys: number) {
    this.pos = nKeys;
    for (const l of this.layers) l.nKeys = nKeys;
  }

  /**
   * The token id the GPU argmax currently holds, without recomputing anything.
   *
   * For the one comparison that isolates the reduction: run it on the SAME logits a
   * host scan reads, so a disagreement is the reduction and not the logits. Calling
   * `step` again instead would recompute the logits and advance the KV cache.
   */
  async currentToken(): Promise<number> {
    return (await this.readU32(this.buf.tokIdx, 1))[0];
  }

  /** Just the embedding of `ids`, for validating the gather end to end. */
  async embed(ids: number[]): Promise<Float32Array> {
    const { dev, cfg, buf } = this;
    const n = ids.length;
    dev.queue.writeBuffer(buf.ids, 0, new Uint32Array(ids));
    dev.queue.writeBuffer(buf.dEmbedPre, 0, new Uint32Array([n, cfg.hidden, 0, 0]));
    const enc = dev.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipes.embed);
    p.setBindGroup(0, this.bind(this.pipes.embed,
      [buf.embdQs, buf.embdSc, buf.ids, buf.embOut, buf.dEmbedPre]));
    p.dispatchWorkgroups(n);
    p.end();
    dev.queue.submit([enc.finish()]);
    return await this.readF32(buf.embOut, n * cfg.hidden);
  }

  /**
   * Greedy decode: prefill `prompt`, then generate up to `maxTokens` ids, stopping
   * at `eos`.
   *
   * The prompt is one prefill (chunked internally); each generated token is one
   * decode step. `onToken` sees each id as it is produced, which is what lets a
   * caller stream without changing this loop.
   */
  async generate(
    prompt: number[], maxTokens: number,
    opts: { eos?: number[]; onToken?: (id: number, i: number) => void } = {},
  ): Promise<number[]> {
    const eos = new Set(opts.eos ?? []);
    const out: number[] = [];
    let next = await this.step(prompt);
    for (let i = 0; i < maxTokens; i++) {
      if (eos.has(next)) break;
      out.push(next);
      opts.onToken?.(next, i);
      if (out.length >= maxTokens) break;
      next = await this.step([next]);
    }
    return out;
  }
}
