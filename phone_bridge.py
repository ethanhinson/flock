"""
phone_bridge.py — makes BROWSERS usable as pipeline nodes.

The problem: shard.py works because a Mac can accept inbound HTTP. A phone
browser cannot. So we invert it -- the phone long-polls /work, computes, and
POSTs to /result. To the coordinator this still looks like "send activations
somewhere, get activations back", so the generation loop needs no special case.

A Swarm holds N such peers, each owning a contiguous layer range. Peers claim
a free slot when they join and release it when they go quiet, so devices can
come and go without restarting the server.
"""
import threading, uuid, time
from queue import Queue, Empty


class PhoneNode:
    """One browser acting as a shard. Blocking call(), long-poll fed."""

    def __init__(self, start, end, slot):
        self.start, self.end, self.slot = start, end, slot
        self.jobs = Queue()
        self.results = {}
        self.cv = threading.Condition()
        self.peer_id = None            # set when a browser claims this slot
        self.last_seen = 0.0
        self.last_ms = None
        self.label = "?"
        self.reset_pending = False

    # --- coordinator side --------------------------------------------------
    def call(self, hidden_flat, offset, timeout=120):
        """Send activations to this peer, block until they come back."""
        jid = uuid.uuid4().hex[:8]
        self.jobs.put({"id": jid, "hidden": hidden_flat, "offset": offset,
                       "reset": self.reset_pending, "start": self.start,
                       "end": self.end})
        self.reset_pending = False
        deadline = time.time() + timeout
        with self.cv:
            while jid not in self.results:
                if not self.cv.wait(timeout=max(0, deadline - time.time())):
                    raise TimeoutError(
                        f"peer for layers {self.start}-{self.end} stopped "
                        f"responding -- is its screen on and the tab in front?")
            return self.results.pop(jid)

    def reset(self):
        """New generation: tell the peer to drop its K/V cache.

        Sent as a flag on the next job rather than its own round trip, since
        the peer is long-polling and has no inbound channel of its own.
        """
        self.reset_pending = True

    # --- peer side ---------------------------------------------------------
    def next_job(self, wait=25):
        """Long-poll: hold the request open until there's work (or time out)."""
        self.last_seen = time.time()
        try:
            return self.jobs.get(timeout=wait)
        except Empty:
            return None

    def submit(self, jid, output, ms):
        self.last_ms = ms
        with self.cv:
            self.results[jid] = {"output": output, "ms": ms}
            self.cv.notify_all()

    def is_alive(self, grace=40):
        return self.peer_id is not None and (time.time() - self.last_seen) < grace

    def info(self):
        return {"slot": self.slot, "start": self.start, "end": self.end,
                "alive": self.is_alive(), "last_ms": self.last_ms,
                "label": self.label}


class Swarm:
    """The set of browser-held shards, in layer order."""

    def __init__(self, ranges):
        self.nodes = [PhoneNode(s, e, i) for i, (s, e) in enumerate(ranges)]
        self.lock = threading.Lock()

    def claim(self, peer_id, label="phone"):
        """Give a joining browser the first free slot (or its own, on reload)."""
        with self.lock:
            for n in self.nodes:
                if n.peer_id == peer_id:
                    n.last_seen = time.time()
                    return n
            for n in self.nodes:
                if not n.is_alive():
                    n.peer_id, n.label = peer_id, label
                    n.last_seen = time.time()
                    return n
        return None

    def ready(self):
        return all(n.is_alive() for n in self.nodes)

    def missing(self):
        return [f"{n.start}-{n.end}" for n in self.nodes if not n.is_alive()]

    def reset(self):
        for n in self.nodes:
            n.reset()

    def by_peer(self, peer_id):
        return next((n for n in self.nodes if n.peer_id == peer_id), None)


def register(app, swarm, meta, web_dir="web"):
    """Attach the peer endpoints to an existing Flask app."""
    from flask import request, jsonify, send_from_directory, Response

    @app.post("/join")
    def join():
        b = request.get_json() or {}
        pid = b.get("peer_id") or uuid.uuid4().hex[:8]
        node = swarm.claim(pid, b.get("label", "phone"))
        if node is None:
            return jsonify({"error": "swarm full — every layer slot is taken"}), 409
        return jsonify({"peer_id": pid, "slot": node.slot,
                        "start": node.start, "end": node.end,
                        **{k: meta[k] for k in
                           ("hidden", "kv_heads", "head_dim", "kv_cache",
                            "n_total", "model")},
                        "n_layers": node.end - node.start + 1})

    @app.post("/work")
    def work():
        node = swarm.by_peer((request.get_json() or {}).get("peer_id"))
        if node is None:
            return jsonify({"error": "unknown peer — rejoin"}), 409
        job = node.next_job()
        if job is None:
            return Response(status=204)          # nothing to do, poll again
        return jsonify(job)

    @app.post("/result")
    def result():
        b = request.get_json()
        node = swarm.by_peer(b.get("peer_id"))
        if node is None:
            return jsonify({"error": "unknown peer"}), 409
        node.submit(b["id"], b["output"], b["ms"])
        return jsonify({"ok": True})

    @app.get("/phone")
    def phone():
        return send_from_directory(web_dir, "phone.html")

    @app.get("/shard/<int:slot>.onnx")
    def shard_onnx(slot):
        return send_from_directory(web_dir, f"shard{slot}.onnx")

    @app.get("/shard/<int:slot>.onnx.data")
    def shard_data(slot):
        # torch.onnx puts the actual WEIGHTS in this sidecar; the .onnx is just
        # the graph. ONNX Runtime fetches it by name, so it MUST be served or
        # the peer loads a graph with no weights and fails.
        return send_from_directory(web_dir, f"shard{slot}.onnx.data")
