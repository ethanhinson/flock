"""
swarm_phone.py — Mac + PHONE swarm. The phone genuinely computes layers.

Split:  Mac  = embedding + layers 0..23  (+ the tied vocab projection)
        PHONE = layers 24..27 + final norm, on its own GPU via WebGPU

Per token the Mac sends a ~4KB activation to the phone, the phone runs four
real transformer layers on its GPU, and sends ~4KB back. Nothing but hidden
states crosses the wire.

Run:  python3 swarm_phone.py
Then open http://<mac-ip>:8000/phone on the phone, tap join,
and http://<mac-ip>:8000 on any browser to chat.
"""
import json, os, time, threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, Response, send_from_directory, jsonify
from phone_bridge import PhoneNode, register

if not os.path.exists("web/phone_shard.json"):
    raise SystemExit(
        "No phone shard found. Build one first:\n"
        "    python3 export_phone_shard.py --start 24 --end 27")

META = json.load(open("web/phone_shard.json"))
MODEL = META["model"]
CUT = META["start"]                     # phone owns CUT..n_total-1
USE_CACHE = META.get("kv_cache", False)

print(f"loading {MODEL} on the Mac (layers 0-{CUT-1}) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
full = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

# --- THE SPLIT: Mac drops the layers the phone owns, and frees that memory.
mac_layers = full.model.layers[:CUT]
full.model.layers = torch.nn.ModuleList(mac_layers)
embed, rotary, lm_head = full.model.embed_tokens, full.model.rotary_emb, full.lm_head
HEAD_DIM = full.config.head_dim
print(f"Mac holds layers 0-{CUT-1}; phone will hold {CUT}-{META['end']}")
print(f"kv cache: {'ON' if USE_CACHE else 'OFF (re-runs whole sequence each step)'}")

phone = PhoneNode(META)
app = Flask(__name__, static_folder="web")
register(app, phone)


class MacCache:
    """Per-layer K/V for the Mac's own layers. Same contract HF attention wants.

    Note this is HALF a cache: the phone independently keeps K/V for its four
    layers. Conversation state is sharded exactly like the weights are -- no
    single device holds the whole thing.
    """
    def __init__(self):
        self.k = self.v = None

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if self.k is not None:
            k = torch.cat([self.k, k], dim=2)
            v = torch.cat([self.v, v], dim=2)
        self.k, self.v = k, v
        return k, v


def mac_forward(ids, caches=None, offset=0):
    """Embedding + our layers -> hidden state to hand to the phone.

    With a cache, `ids` is just the NEW tokens (usually one) and `offset` says
    where they sit in the sequence, so we do O(1) work per step instead of
    re-running the whole prompt every time.
    """
    q = len(ids)
    pids = torch.arange(offset, offset + q)[None, :]
    h = embed(torch.tensor([ids]))
    pos = rotary(h, pids)
    kv = offset + q
    m = torch.full((q, kv), torch.finfo(torch.float32).min)
    m = torch.triu(m, diagonal=1 + (kv - q))[None, None]
    with torch.no_grad():
        for i, lyr in enumerate(full.model.layers):
            if caches is not None:
                out = lyr(h, attention_mask=m, position_ids=pids,
                          past_key_values=caches[i], use_cache=True,
                          position_embeddings=pos)
            else:
                out = lyr(h, attention_mask=m, position_ids=pids,
                          position_embeddings=pos)
            h = out[0] if isinstance(out, tuple) else out
    return h


def project(h_last):
    """Tied vocab projection -- kept on the Mac so the phone stays small."""
    with torch.no_grad():
        return int(lm_head(h_last).argmax(-1).item())


@app.post("/chat")
def chat():
    body = request.get_json()
    prompt, max_new = body["prompt"], int(body.get("max_tokens", 24))
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True)
    # transformers versions disagree here: plain list, BatchEncoding, or tensor.
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]                          # unwrap any batch dimension
    ids = [int(i) for i in ids]

    def stream():
        if not phone.is_alive():
            yield sse({"type": "error", "text": "no phone connected — open /phone on your phone and tap join"})
            return
        out_ids = []
        caches = [MacCache() for _ in full.model.layers] if USE_CACHE else None
        phone.reset()                                # clear the phone's K/V too
        step_ids, offset = ids, 0                    # prefill: the whole prompt
        for step in range(max_new):
            t0 = time.perf_counter()
            h = mac_forward(step_ids, caches, offset)   # Mac's 24 layers
            mac_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            try:
                r = phone.call(h.flatten().tolist(), offset=offset)  # -> PHONE's 4 layers
            except TimeoutError:
                yield sse({"type": "error", "text":
                           "phone stopped responding mid-generation — is the screen "
                           "still on and the tab in front?"})
                return
            net_ms = (time.perf_counter() - t1) * 1000

            h2 = torch.tensor(r["output"], dtype=torch.float32).view(1, len(step_ids), -1)
            nxt = project(h2[:, -1])                 # Mac projects to vocab

            yield sse({"type": "hop", "mac_ms": round(mac_ms, 1),
                       "phone_ms": r["ms"], "roundtrip_ms": round(net_ms, 1),
                       "kb": round(h.numel() * 4 / 1024, 1)})
            if nxt in (tok.eos_token_id,):
                break
            out_ids.append(nxt)
            # Next round we feed ONLY the new token; the caches hold the rest.
            offset += len(step_ids)
            step_ids = [nxt] if USE_CACHE else ids + out_ids
            if not USE_CACHE:
                offset = 0
            yield sse({"type": "token", "text": tok.decode([nxt])})
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
    return jsonify({"phone_connected": phone.is_alive(),
                    "mac_layers": f"0-{CUT-1}",
                    "phone_layers": f"{META['start']}-{META['end']}",
                    "n_total": META["n_total"], "hidden": META["hidden"],
                    "phone_last_ms": phone.last_ms})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
