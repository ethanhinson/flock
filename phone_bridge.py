"""
phone_bridge.py — makes a BROWSER usable as a pipeline node.

The problem: shard.py works because a Mac can accept inbound HTTP. A phone
browser cannot. So we invert it -- the phone long-polls /work, computes, and
POSTs to /result. To the coordinator this still looks like "send activations
somewhere, get activations back", so swarm.py needs no special case.

This module exposes exactly that: a queue plus a blocking call() that the
coordinator uses just like a requests.post to a Mac shard.
"""
import threading, uuid, time
from queue import Queue, Empty


class PhoneNode:
    """One browser acting as a shard. Blocking call(), long-poll fed."""

    def __init__(self, meta):
        self.meta = meta               # start/end/hidden/is_last from the export
        self.jobs = Queue()
        self.results = {}
        self.cv = threading.Condition()
        self.connected = False
        self.last_seen = 0.0
        self.last_ms = None

    # --- coordinator side --------------------------------------------------
    def call(self, hidden_flat, offset, timeout=120):
        """Send activations to the phone, block until it sends them back."""
        jid = uuid.uuid4().hex[:8]
        self.jobs.put({"id": jid, "hidden": hidden_flat, "offset": offset})
        deadline = time.time() + timeout
        with self.cv:
            while jid not in self.results:
                if not self.cv.wait(timeout=max(0, deadline - time.time())):
                    raise TimeoutError("phone did not respond -- is it awake "
                                       "and still on the page?")
            return self.results.pop(jid)

    # --- phone side --------------------------------------------------------
    def next_job(self, wait=25):
        """Long-poll: hold the request open until there's work (or time out)."""
        self.connected, self.last_seen = True, time.time()
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
        return self.connected and (time.time() - self.last_seen) < grace


def register(app, node, web_dir="web"):
    """Attach the phone endpoints to an existing Flask app."""
    from flask import request, jsonify, send_from_directory, Response

    @app.post("/work")
    def work():
        job = node.next_job()
        if job is None:
            return Response(status=204)          # nothing to do, poll again
        return jsonify(job)

    @app.post("/result")
    def result():
        b = request.get_json()
        node.submit(b["id"], b["output"], b["ms"])
        return jsonify({"ok": True})

    @app.get("/phone")
    def phone():
        return send_from_directory(web_dir, "phone.html")

    @app.get("/phone_shard.onnx")
    def onnx():
        return send_from_directory(web_dir, "phone_shard.onnx")

    @app.get("/phone_shard.onnx.data")
    def onnx_data():
        # torch.onnx puts the actual WEIGHTS in this sidecar file; the .onnx is
        # just the graph. ONNX Runtime fetches it by name, so it MUST be served
        # or the phone loads a graph with no weights and fails.
        return send_from_directory(web_dir, "phone_shard.onnx.data")

    @app.get("/phone_shard.json")
    def onnx_meta():
        return send_from_directory(web_dir, "phone_shard.json")
