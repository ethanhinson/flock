# flock (Node runtime)

The all-JavaScript flock: the coordinator runs on Node, so it can be a real
WebRTC peer. Every link in the chain is a data channel; the websocket carries
only signaling.

```bash
cd node && npm install
npm start
```

Requires the exports first (from the repo root):

```bash
python3 build_shards.py --start 24 --end 27 --birds 2
python3 flock_export_coordinator.py --cut 24
```

Python remains only as the **export and verification tool** — PyTorch is the
reference every ONNX graph is checked against, so a bad export can't be
silently wrong.
