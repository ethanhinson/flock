// flock coordinator — the machine that starts and ends each lap.
//
// Holds the embedding, layers 0..cut-1, and the tied vocab projection, all as
// ONNX graphs exported by flock_export_coordinator.py. Because this runs on
// Node rather than Python, it can be a real WebRTC peer -- so every link in
// the chain is a data channel, exactly like swarmllm.
import ort from 'onnxruntime-node';
import {readFileSync} from 'fs';
import {AutoTokenizer} from '@huggingface/transformers';

export class Coordinator {
  static async load(dir = 'web/coord') {
    const c = new Coordinator();
    c.meta = JSON.parse(readFileSync(`${dir}/coord.json`, 'utf8'));
    c.tok = await AutoTokenizer.from_pretrained(`${dir}/tok`, {local_files_only: true});
    // ORT resolves the .data sidecars relative to the graph path.
    [c.embed, c.layers, c.head] = await Promise.all([
      ort.InferenceSession.create(`${dir}/embed.onnx`),
      ort.InferenceSession.create(`${dir}/layers.onnx`),
      ort.InferenceSession.create(`${dir}/head.onnx`),
    ]);
    c.past = null;
    return c;
  }

  encode(prompt) {
    const text = this.tok.apply_chat_template(
      [{role: 'user', content: prompt}],
      {add_generation_prompt: true, enable_thinking: false, tokenize: false});
    const enc = this.tok(text, {add_special_tokens: false});
    return Array.from(enc.input_ids.data).map(Number);
  }

  decode(ids) {
    return this.tok.decode(ids, {skip_special_tokens: true});
  }

  reset() { this.past = null; }

  empty() {
    return new ort.Tensor('float32', new Float32Array(0),
                          [1, this.meta.kv_heads, 0, this.meta.head_dim]);
  }

  /** token ids -> hidden state, through the embedding and our layers. */
  async forward(ids, offset) {
    const n = ids.length;
    const emb = await this.embed.run({
      ids: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, n]),
    });
    const feed = {
      hidden: emb.hidden,
      position_ids: new ort.Tensor('int64',
        BigInt64Array.from({length: n}, (_, i) => BigInt(offset + i)), [1, n]),
    };
    // Our own K/V cache -- the birds each keep their own for their layers.
    for (let i = 0; i < this.meta.n_layers; i++) {
      feed[`past_k${i}`] = this.past ? this.past[`new_k${i}`] : this.empty();
      feed[`past_v${i}`] = this.past ? this.past[`new_v${i}`] : this.empty();
    }
    const out = await this.layers.run(feed);
    this.past = {};
    for (let i = 0; i < this.meta.n_layers; i++) {
      this.past[`new_k${i}`] = out[`new_k${i}`];
      this.past[`new_v${i}`] = out[`new_v${i}`];
    }
    return out.output.data;   // Float32Array
  }

  /** hidden state (last position) -> next token id */
  async project(flat, seq) {
    const h = this.meta.hidden;
    const last = flat.slice((seq - 1) * h, seq * h);
    const out = await this.head.run({
      hidden: new ort.Tensor('float32', Float32Array.from(last), [1, 1, h]),
    });
    return Number(out.token.data[0]);
  }
}
