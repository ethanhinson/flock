"""flock.routes — HTTP + WebSocket endpoints the birds talk to."""
import json, uuid
from flask import request, jsonify, send_from_directory, Response

from .wire import unpack


def register(app, sock, flock, meta, web_dir="web"):
    @app.post("/join")
    def join():
        b = request.get_json() or {}
        pid = b.get("peer_id") or uuid.uuid4().hex[:8]
        bird = flock.claim(pid, b.get("label", "phone"))
        if bird is None:
            return jsonify({"error": "flock full — every layer slot is taken"}), 409
        return jsonify({"peer_id": pid, "slot": bird.slot,
                        "start": bird.start, "end": bird.end,
                        "n_layers": bird.end - bird.start + 1,
                        **{k: meta[k] for k in ("hidden", "kv_heads", "head_dim",
                                                "kv_cache", "n_total", "model")}})

    @sock.route("/ws")
    def ws(ws):
        """One long-lived socket per bird. Frames in, frames out, no polling."""
        pid = ws.receive()                       # first message: our peer id
        try:
            hello = json.loads(pid)
        except Exception:
            return
        bird = flock.by_peer(hello.get("peer_id"))
        if bird is None:
            ws.send(json.dumps({"error": "unknown peer — rejoin"}))
            return
        bird.attach(ws, hello["peer_id"], hello.get("label", bird.label))
        try:
            while True:
                msg = ws.receive()
                if msg is None:
                    break
                if isinstance(msg, str):
                    # control: signaling relay + timing reports
                    m = json.loads(msg)
                    if m.get("t") == "signal":
                        flock.post_signal(m["to"], {"from": hello["peer_id"],
                                                    "data": m["data"]})
                    elif m.get("t") == "stats":
                        bird.note_stats(m.get("ms"))
                        if m.get("transport"):
                            bird.transport = m["transport"]
                    elif m.get("t") == "pull":
                        for s in flock.take_signals(hello["peer_id"]):
                            ws.send(json.dumps({"t": "signal", **s}))
                        ws.send(json.dumps({"t": "chain",
                                            **(flock.chain_of(hello["peer_id"]) or {})}))
                else:
                    bird.deliver(msg)            # binary: activations coming back
        finally:
            bird.detach()

    @app.get("/flock")
    def flock_page():
        return send_from_directory(web_dir, "bird.html")

    @app.get("/shard/<int:slot>.onnx")
    def shard_onnx(slot):
        return send_from_directory(web_dir, f"shard{slot}.onnx")

    @app.get("/shard/<int:slot>.onnx.data")
    def shard_data(slot):
        # torch.onnx puts the WEIGHTS in this sidecar; the .onnx is just the
        # graph. ONNX Runtime fetches it by name, so it MUST be served.
        return send_from_directory(web_dir, f"shard{slot}.onnx.data")

    @app.get("/js/<path:f>")
    def js(f):
        return send_from_directory(f"{web_dir}/js", f)
