"""
swarm.py — the coordinator. Serves the phone UI and drives the relay.

It holds ZERO model weights. Its only jobs: tokenize, walk the hidden state
through each shard in order, and stream the resulting tokens out. Every token
you see took one full lap through every node in the ring.
"""
import argparse, json, time
import requests
from flask import Flask, request, Response, send_from_directory
from mlx_lm import load

p = argparse.ArgumentParser()
p.add_argument("--shards", nargs="+", required=True,   # e.g. localhost:8001 localhost:8002
               help="shard host:port, IN LAYER ORDER")
p.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
p.add_argument("--port", type=int, default=8000)
args = p.parse_args()

# We load ONLY the tokenizer here (tiny). No transformer weights on this box.
_, tokenizer = load(args.model)
SHARDS = [s if s.startswith("http") else f"http://{s}" for s in args.shards]

for s in SHARDS:
    print(f"[swarm] shard {s} -> {requests.get(s + '/info', timeout=120).json()}")

app = Flask(__name__, static_folder=".")


def relay(tokens, on_hop):
    """One lap: token ids in at node 0, next-token id out of node N.

    This function IS the concept. Note what crosses the wire: a [1, seq, 2048]
    tensor, never weights. Compare that to the gigabytes each node holds.
    """
    payload = {"tokens": tokens}
    for i, s in enumerate(SHARDS):
        t0 = time.perf_counter()
        r = requests.post(f"{s}/forward", json=payload, timeout=300).json()
        wire_ms = round((time.perf_counter() - t0) * 1000, 1)
        on_hop({"node": i, "shard": r["shard"], "compute_ms": r["ms"],
                "total_ms": wire_ms, "kb": round(r.get("bytes", 0) / 1024, 1)})
        if "token" in r:
            return r["token"]
        payload = {"hidden": r["hidden"]}          # hand off to the next node
    raise RuntimeError("last shard did not return a token")


@app.post("/chat")
def chat():
    prompt = request.get_json()["prompt"]
    max_tokens = int(request.get_json().get("max_tokens", 60))
    for s in SHARDS:
        requests.post(f"{s}/reset", timeout=60)

    msgs = [{"role": "user", "content": prompt}]
    # Qwen3 is a reasoning model; without this it burns tokens on <think> blocks.
    try:
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            enable_thinking=False)
    except TypeError:
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

    def stream():
        hops = []
        tok = relay(ids, hops.append)              # prefill: whole prompt, one lap
        yield sse({"type": "hops", "hops": hops, "phase": "prefill"})
        out = []
        for _ in range(max_tokens):
            if tok in tokenizer.eos_token_ids:
                break
            out.append(tok)
            yield sse({"type": "token", "text": tokenizer.decode([tok])})
            hops = []
            tok = relay([tok], hops.append)        # decode: ONE token per lap
            yield sse({"type": "hops", "hops": hops, "phase": "decode"})
        yield sse({"type": "done", "text": tokenizer.decode(out)})

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/")
def index():
    return send_from_directory(".", "ui.html")


@app.get("/topology")
def topology():
    return {"shards": [requests.get(s + "/info", timeout=60).json() for s in SHARDS]}


app.run(host="0.0.0.0", port=args.port, threaded=True)
