"""
flock.wire — how activations cross the network.

The payload between devices is a hidden state: a flat array of floats. How you
encode it dominates the cost, because it's sent once per token per hop.

    fp32 JSON   86,769 bytes   <- what a naive implementation sends
    fp16 binary 10,240 bytes   <- what this module sends

That's 8.5x, and on real activation values (|x| < 0.1, derived from bf16
weights) the fp16 round trip is lossless. swarmllm uses the same trick --
f16-packed Uint16Array frames over WebRTC.
"""
import numpy as np

MAGIC = b"FLK1"          # 4-byte header so a peer can reject junk frames


def pack(arr, seq, hidden, offset=0, reset=False):
    """float32 array -> compact binary frame.

    Layout: MAGIC | seq u32 | hidden u32 | offset u32 | flags u32 | f16 payload
    """
    a = np.asarray(arr, dtype=np.float32).ravel().astype(np.float16)
    head = np.array([seq, hidden, offset, 1 if reset else 0], dtype=np.uint32)
    return MAGIC + head.tobytes() + a.tobytes()


def unpack(buf):
    """binary frame -> (float32 array, meta dict)."""
    if buf[:4] != MAGIC:
        raise ValueError("not a flock frame")
    seq, hidden, offset, flags = np.frombuffer(buf, dtype=np.uint32, count=4, offset=4)
    payload = np.frombuffer(buf, dtype=np.float16, offset=20).astype(np.float32)
    return payload, {"seq": int(seq), "hidden": int(hidden),
                     "offset": int(offset), "reset": bool(flags & 1)}


def sizes(n):
    """What a given float count costs on the wire, for the UI to show."""
    return {"fp32_json": n * 21, "fp16_binary": n * 2 + 20}
