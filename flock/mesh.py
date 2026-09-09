"""
flock.mesh — the flock of devices, and how work reaches them.

Each bird in the flock holds a contiguous range of transformer layers. The
coordinator (this machine) holds the embedding, the first layers, and the vocab
projection; every other range lives on a phone.

TRANSPORT
Two paths, because a browser can't accept inbound connections:

  websocket  the coordinator pushes a frame to a bird and awaits the reply.
             Event-driven -- no polling latency.
  webrtc     birds talk DIRECTLY to each other. The coordinator hands out
             offers/answers (signaling only) and then leaves the data path,
             so bird A -> bird B never round-trips through this machine.

Long-polling was the previous design; it cost a full extra hop through the
coordinator per link. See docs in README.
"""
import json, threading, time, uuid
from queue import Queue, Empty

from .wire import pack, unpack


class Bird:
    """One device holding a contiguous slice of layers."""

    def __init__(self, start, end, slot):
        self.start, self.end, self.slot = start, end, slot
        self.peer_id = None
        self.label = "?"
        self.last_seen = 0.0
        self.last_ms = None
        self.transport = "none"        # ws | webrtc-relay | none
        self.reset_pending = False

        self._ws = None                # live websocket, if this bird has one
        self._inbox = Queue()          # frames coming back from the bird
        self._lock = threading.Lock()

    # --- coordinator side --------------------------------------------------
    def attach(self, ws, peer_id, label):
        with self._lock:
            self._ws = ws
            self.peer_id, self.label = peer_id, label
            self.transport = "ws"
            self.last_seen = time.time()

    def detach(self):
        with self._lock:
            self._ws = None
            self.transport = "none"

    def send(self, floats, seq, hidden, offset, timeout=120):
        """Push activations to this bird; block until they come back."""
        if self._ws is None:
            raise TimeoutError(f"no device holding layers {self.start}-{self.end}")
        frame = pack(floats, seq=seq, hidden=hidden, offset=offset,
                     reset=self.reset_pending)
        self.reset_pending = False
        while not self._inbox.empty():        # drop anything stale
            self._inbox.get_nowait()
        try:
            self._ws.send(frame)
        except Exception as e:
            self.detach()
            raise TimeoutError(f"layers {self.start}-{self.end} went away: {e}")
        try:
            return self._inbox.get(timeout=timeout)
        except Empty:
            raise TimeoutError(
                f"device holding layers {self.start}-{self.end} stopped "
                f"responding -- is its screen on and the tab in front?")

    def deliver(self, buf):
        """A frame arrived from the bird."""
        self.last_seen = time.time()
        data, meta = unpack(buf)
        self.last_ms = meta.get("ms")
        self._inbox.put((data, meta))

    def note_stats(self, ms):
        self.last_ms = ms
        self.last_seen = time.time()

    def reset(self):
        """New generation: tell this bird to drop its K/V cache."""
        self.reset_pending = True

    def is_alive(self, grace=40):
        return self._ws is not None and (time.time() - self.last_seen) < grace

    def info(self):
        return {"slot": self.slot, "start": self.start, "end": self.end,
                "alive": self.is_alive(), "last_ms": self.last_ms,
                "label": self.label, "transport": self.transport}


class Flock:
    """The ordered chain of birds covering the non-coordinator layers."""

    def __init__(self, ranges):
        self.birds = [Bird(s, e, i) for i, (s, e) in enumerate(ranges)]
        self.lock = threading.Lock()
        self.signals = {}              # peer_id -> [pending signaling messages]

    def claim(self, peer_id, label="phone"):
        """Give a joining device the first free slot (or its own, on reload)."""
        with self.lock:
            for b in self.birds:
                if b.peer_id == peer_id:
                    b.last_seen = time.time()
                    return b
            for b in self.birds:
                if not b.is_alive():
                    b.peer_id, b.label = peer_id, label
                    b.last_seen = time.time()
                    return b
        return None

    def by_peer(self, peer_id):
        return next((b for b in self.birds if b.peer_id == peer_id), None)

    def ready(self):
        return all(b.is_alive() for b in self.birds)

    def missing(self):
        return [f"{b.start}-{b.end}" for b in self.birds if not b.is_alive()]

    def reset(self):
        for b in self.birds:
            b.reset()

    # --- WebRTC signaling ---------------------------------------------------
    # The coordinator only introduces birds to each other. Once the data
    # channel opens, activations flow bird-to-bird and this machine is out of
    # that path entirely.
    def post_signal(self, to_peer, msg):
        with self.lock:
            self.signals.setdefault(to_peer, []).append(msg)

    def take_signals(self, peer_id):
        with self.lock:
            out = self.signals.pop(peer_id, [])
        return out

    def chain_of(self, peer_id):
        """Who this bird should forward to, and who it expects frames from."""
        b = self.by_peer(peer_id)
        if b is None:
            return None
        nxt = self.birds[b.slot + 1] if b.slot + 1 < len(self.birds) else None
        return {"slot": b.slot,
                "next_peer": nxt.peer_id if nxt else None,
                "next_range": f"{nxt.start}-{nxt.end}" if nxt else None,
                "is_last": nxt is None}
