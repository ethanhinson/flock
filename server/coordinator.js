// flock coordinator — the machine that starts and ends each lap.
//
// Holds the embedding, layers 0..cut-1, output_norm and the tied vocab
// projection, all as WGSL compute kernels over Q8_0 weights read straight out of
// a GGUF file. No ONNX, no export, no build step, no artifacts on disk
// beyond the cached model file itself.
//
// WHY THIS RUNS UNDER DENO AND NOT NODE. The coordinator needs a GPU, and Node
// has no WebGPU: there is no `navigator` at all on node v20, and no flag that
// adds one (`--experimental-webgpu` is not a Node option; that is Deno's). This
// was checked rather than assumed. Deno 2.1.2 gives a real device AND runs the
// rest of the stack unchanged through its node compatibility layer -- express,
// ws, node-datachannel and @huggingface/transformers all import and work -- so
// the whole server moved to one runtime rather than being split across two
// processes with a socket between them. See README.md.
//
// The alternative designs were considered and rejected: a browser-hosted
// coordinator (what swarmllm does) makes the thing you have to keep open a tab
// rather than a server, and a CPU coordinator means writing a second
// implementation of 24 layers that nothing validates. Deno costs neither.
//
// WHAT IS SHARED WITH THE ENGINE THE TESTS PIN. This class is a thin driver over
// kernels/model.ts with `cut` set. It adds the tokenizer, the chat template and
// the conversation bookkeeping; it adds no arithmetic. kernels/test_model.ts
// proves that engine generates the same token ids as the ONNX pipeline, and
// kernels/test_split.ts proves cutting it changes nothing -- so the inference
// path here inherits both.
import { getDevice } from '../kernels/lib.ts';
import { Model } from '../kernels/model.ts';
import { QWEN3_06B } from '../kernels/layer.ts';
import { realModel } from '../kernels/real_weights.ts';
import { AutoTokenizer } from '@huggingface/transformers';

/** Where the tokenizer comes from.
 *
 *  ONE HONEST CAVEAT about "no ONNX in the process". @huggingface/transformers
 *  bundles its own copy of onnxruntime-node and dlopens libonnxruntime at import
 *  time, so the .dylib IS mapped into this process -- visible in lsof. Nothing
 *  here ever asks it to run anything: the only API used from that package is
 *  AutoTokenizer, never AutoModel or pipeline(), so no InferenceSession is ever
 *  constructed and no ONNX graph is ever loaded. The inference path is WGSL end to
 *  end. Removing the mapping entirely would mean replacing transformers.js with a
 *  standalone tokenizer, which buys nothing for correctness and would put the chat
 *  template -- the thing that must match the ONNX comparison exactly -- into new
 *  code.
 *
 *  The GGUF carries the vocabulary but not the chat template, and getting the
 *  template wrong changes the prompt and therefore the text -- which looks exactly
 *  like an engine bug. So the tokenizer is the HuggingFace one, fetched once and
 *  cached by transformers.js, and it is the SAME template the ONNX comparison
 *  used: verified to produce the identical 16 prompt ids for "Capital of France?".
 */
const TOKENIZER = 'Qwen/Qwen3-0.6B';

export class Coordinator {
  /**
   * Load the embedding, layers 0..cut-1, output_norm and the head onto the GPU.
   *
   * `cut` comes from the flock's layer plan rather than from a file, because with
   * GGUF the split is just a choice about byte ranges -- there is nothing exported
   * per-split any more, so it can be decided at startup.
   */
  static async load(cut = 24) {
    const c = new Coordinator();
    c.dev = await getDevice();
    const weights = await realModel();
    c.model = await Model.create(c.dev, weights, QWEN3_06B, { cut });
    c.tok = await AutoTokenizer.from_pretrained(TOKENIZER);
    // The shape the rest of the server reads. Taken from the GGUF metadata and the
    // kernel config rather than from an exported coord.json, so there is one
    // source of truth for it.
    c.meta = {
      hidden: QWEN3_06B.hidden,
      kv_heads: QWEN3_06B.nKvHeads,
      head_dim: QWEN3_06B.headDim,
      n_layers: cut,
      n_total: weights.nLayers,
      vocab: weights.vocab,
      eos: weights.eos,
    };
    return c;
  }

  /** Raw text -> token ids, with no chat template applied. */
  ids(text) {
    return Array.from(this.tok(text, {add_special_tokens: false})
                          .input_ids.data).map(Number);
  }

  /** The tokens to feed for ONE user turn, continuing whatever is already cached.
   *
   *  Deliberately not `apply_chat_template(wholeHistory)`: with
   *  enable_thinking:false the template injects a `<think></think>` scaffold for
   *  the CURRENT turn only, so re-encoding the history produces a token stream
   *  that DIVERGES from what we already fed (measured: they split at token 9 of
   *  13). Feeding that against a warm K/V cache would silently compute garbage.
   *
   *  So we append exactly what the model saw next. The cost is that earlier
   *  turns keep their scaffold, which is well-formed and is what the model
   *  itself generated in context. The benefit is that the sharded K/V cache
   *  stays valid across turns, which is the whole point of sharding it.
   */
  turnTokens(prompt, isFirst) {
    const open = isFirst ? '' : '<|im_end|>\n';
    return this.ids(`${open}<|im_start|>user\n${prompt}<|im_end|>\n` +
                    `<|im_start|>assistant\n<think>\n\n</think>\n\n`);
  }

  decode(ids) {
    return this.tok.decode(ids, {skip_special_tokens: true});
  }

  reset() { this.model.reset(); }

  /** token ids -> hidden state, through the embedding and our layers.
   *
   *  Returns every position, not just the last: the next device in the chain has
   *  to fill its own K/V cache for all of them. `offset` is passed so a
   *  disagreement between the server's idea of the conversation and this cache
   *  throws instead of computing against keys for other positions. */
  async forward(ids, offset) {
    return await this.model.encodePartial(ids, offset);
  }

  /** hidden state (from the last bird) -> next token id.
   *
   *  output_norm, the tied projection and the argmax, in that order and exactly
   *  once -- see kernels/model.ts on why the ONNX layout put the final norm
   *  somewhere else and why copying it would be wrong. */
  async project(flat, seq) {
    return await this.model.projectHidden(flat, seq);
  }
}
