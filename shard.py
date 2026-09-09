"""
shard.py — one node in the swarm. Holds a CONTIGUOUS SLICE of the model's layers.

The whole idea in one paragraph:
A transformer is a stack of identical layers. Token ids become a hidden-state
tensor of shape [batch, seq, hidden]; each layer reads that tensor and writes a
same-shaped tensor. So you can cut the stack anywhere, put layers 0..k on
machine A and k+1..n on machine B, and ship the hidden state between them. The
tensor is tiny compared to the weights -- that asymmetry is the entire trick.

Run:  python shard.py --start 0 --end 13 --port 8001
"""
import argparse, json, time
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from flask import Flask, request, jsonify

p = argparse.ArgumentParser()
p.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
p.add_argument("--start", type=int, required=True)   # first layer this node owns
p.add_argument("--end", type=int, required=True)     # last layer, inclusive
p.add_argument("--port", type=int, required=True)
p.add_argument("--host", default="0.0.0.0")
args = p.parse_args()

print(f"[shard :{args.port}] loading {args.model} ...")
model, tokenizer = load(args.model)
inner = model.model                      # the Qwen3Model (embed + layers + norm)
n_total = len(inner.layers)

# --- THE SPLIT -------------------------------------------------------------
# Keep only our slice. Everything else is freed, so each node's memory scales
# with its share of the model, not the whole thing. This is what lets a phone
# participate in a model it could never hold alone.
my_layers = inner.layers[args.start : args.end + 1]
inner.layers = my_layers
is_first = args.start == 0               # only the first node embeds tokens
is_last = args.end == n_total - 1        # only the last node norms + projects

# Qwen3 ties input/output embeddings: the SAME matrix embeds tokens and projects
# the final hidden state back to vocab. So the last node needs it too, even
# though it never embeds anything. Middle nodes drop it and save the memory.
if not is_first and not is_last:
    del inner.embed_tokens
mx.clear_cache()

cache = make_prompt_cache(inner)         # one KV cache per layer WE own
HIDDEN = model.args.hidden_size
print(f"[shard :{args.port}] layers {args.start}-{args.end} of {n_total} "
      f"| first={is_first} last={is_last} | hidden={HIDDEN}")

app = Flask(__name__)


def to_wire(a):
    """MLX array -> plain JSON-able floats.

    The cast to float32 is required: models run in bfloat16, which numpy has no
    native dtype for. Real systems ship raw bf16 bytes instead of JSON floats --
    that is ~4x smaller and much faster, but far less readable than this.
    """
    return np.array(a.astype(mx.float32), copy=False).tolist()


@app.post("/forward")
def forward():
    """Run the hidden state through OUR layers only, then hand it back."""
    body = request.get_json()
    t0 = time.perf_counter()

    if is_first:
        # We own the embedding table: turn token ids into the first hidden state.
        h = inner.embed_tokens(mx.array([body["tokens"]]))
    else:
        # Someone upstream already did the earlier layers. Pick up their tensor.
        h = mx.array(np.array(body["hidden"], dtype=np.float32))

    from mlx_lm.models.base import create_attention_mask
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask, c)

    mx.eval(h)   # MLX is lazy; force the compute before we time or serialize it
    payload = {"shard": f"{args.start}-{args.end}"}
    if is_last:
        # Final node: normalize, project to vocab, and return ONE token id.
        h = inner.norm(h)
        logits = model.lm_head(h) if "lm_head" in model else inner.embed_tokens.as_linear(h)
        payload["token"] = int(mx.argmax(logits[:, -1, :]).item())
    else:
        payload["hidden"] = to_wire(h)

    payload["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    payload["bytes"] = len(json.dumps(payload.get("hidden", "")))
    return jsonify(payload)


@app.post("/reset")
def reset():
    """New conversation -> drop the KV cache for our layers."""
    global cache
    cache = make_prompt_cache(inner)
    return jsonify({"ok": True})


@app.get("/info")
def info():
    return jsonify({"start": args.start, "end": args.end, "n_total": n_total,
                    "is_first": is_first, "is_last": is_last, "hidden": HIDDEN})


app.run(host=args.host, port=args.port, threaded=False)
