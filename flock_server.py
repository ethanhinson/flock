"""
flock_server.py — run the flock.

The coordinator holds the embedding, the first layers, and the vocab
projection; each phone (a "bird") holds a contiguous slice of the rest. Per
token the hidden state makes one lap: coordinator -> bird -> bird -> back.

Transport is a WebSocket per bird (event-driven, no polling) carrying f16
binary frames. Birds that negotiate a WebRTC data channel forward directly to
each other, taking this machine out of the middle of the chain.

Run:  python3 flock_server.py
Then open http://<this-ip>:8000/flock on each phone, and http://<this-ip>:8000
in any browser to chat.
"""
import json, os, time

import torch
from flask import Flask, request, Response, send_from_directory, jsonify
from flask_sock import Sock
from transformers import AutoModelForCausalLM, AutoTokenizer

from flock.mesh import Flock
from flock.routes import register
from flock.coordinator import Coordinator
from flock.wire import sizes

if not os.path.exists("web/flock.json"):
    raise SystemExit(
        "No shards found. Build them first:\n"
        "    python3 build_shards.py --start 24 --end 27 --peers 1")

META = json.load(open("web/flock.json"))
MODEL = META["model"]
RANGES = [(p["start"], p["end"]) for p in META["birds"]]
CUT = RANGES[0][0]
USE_CACHE = META.get("kv_cache", False)

print(f"loading {MODEL} on the coordinator (layers 0-{CUT-1}) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
coord = Coordinator(
    AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval(), CUT)
print(f"coordinator holds layers 0-{CUT-1}; birds hold " +
      ", ".join(f"{s}-{e}" for s, e in RANGES))
print(f"kv cache: {'ON' if USE_CACHE else 'OFF'}  |  wire: f16 binary")

flock = Flock(RANGES)
app = Flask(__name__, static_folder="web")
sock = Sock(app)
register(app, sock, flock, META)


@app.post("/chat")
def chat():
    body = request.get_json()
    prompt, max_new = body["prompt"], int(body.get("max_tokens", 24))
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(i) for i in ids]

    def stream():
        if not flock.ready():
            yield sse({"type": "error", "text":
                       "waiting for devices to cover layers " +
                       ", ".join(flock.missing()) +
                       " — open /flock on each phone and tap join"})
            return
        out_ids = []
        caches = coord.new_caches() if USE_CACHE else None
        flock.reset()
        step_ids, offset = ids, 0                # prefill: the whole prompt
        for _ in range(max_new):
            t0 = time.perf_counter()
            h = coord.forward(step_ids, caches, offset)
            coord_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            flat = h.flatten().tolist()
            n_floats = len(flat)
            stats = []
            try:
                # The lap: each bird's output is the next one's input. Birds
                # linked by WebRTC forward directly and only the last replies.
                for bird in flock.birds:
                    data, meta = bird.send(flat, seq=len(step_ids),
                                           hidden=h.shape[-1], offset=offset)
                    flat = data
                    stats.append({"range": f"{bird.start}-{bird.end}",
                                  "ms": meta.get("ms"), "label": bird.label,
                                  "transport": bird.transport})
            except TimeoutError as e:
                yield sse({"type": "error", "text": str(e)})
                return
            net_ms = (time.perf_counter() - t1) * 1000

            h2 = torch.tensor(flat, dtype=torch.float32).view(1, len(step_ids), -1)
            nxt = coord.project(h2[:, -1])

            w = sizes(n_floats)
            yield sse({"type": "hop", "coord_ms": round(coord_ms, 1),
                       "birds": stats, "roundtrip_ms": round(net_ms, 1),
                       "kb": round(w["fp16_binary"] / 1024, 1),
                       "kb_json": round(w["fp32_json"] / 1024, 1)})
            if nxt in (tok.eos_token_id,):
                break
            out_ids.append(nxt)
            offset += len(step_ids)
            step_ids = [nxt] if USE_CACHE else ids + out_ids
            if not USE_CACHE:
                offset = 0
        yield sse({"type": "done", "text": tok.decode(out_ids)})

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def sse(o):
    return f"data: {json.dumps(o)}\n\n"


@app.get("/")
def index():
    return send_from_directory("web", "chat.html")


@app.get("/status")
def status():
    return jsonify({"ready": flock.ready(), "missing": flock.missing(),
                    "coord_layers": f"0-{CUT-1}",
                    "birds": [b.info() for b in flock.birds],
                    "n_total": META["n_total"], "hidden": META["hidden"],
                    "kv_cache": USE_CACHE, "wire": "f16"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
