# flock

**Run one language model split across your laptop and your phone.**

Your phone's GPU actually computes four transformer layers. Activations cross
your wifi. It is a real, if small, distributed inference system — about 600
lines you can read in a sitting.

This is a *learning* prototype, built to make the ideas behind
[swarmllm.ai](https://swarmllm.ai) and [exo](https://github.com/exo-explore/exo)
concrete. It is not production software. See [Limitations](#limitations).

---

## Measured on real hardware

M3 Max + an iPhone on the same wifi, Qwen3-0.6B split 24/4:

```
tok     mac ms  phone ms  wire KB  total ms
1        116.8     123.0     64.0     306.9
2         71.6      87.0     68.0     205.3
3         78.3      92.0     72.0     197.5
...
10        77.3      94.0    100.0     247.7

avg  mac 81ms | phone 97ms | roundtrip 229ms
phone share of compute: 54%
network overhead: 132ms/token
```

Output: `The capital of France is **Paris**.`

The phone runs 4 of 28 layers but spends *more* time than the Mac's 24 — that
asymmetry is what real heterogeneous hardware looks like.

---

## The idea in one paragraph

A transformer is a stack of identical layers. Token ids become a hidden state
of shape `[batch, seq, hidden]`; each layer reads that tensor and writes one
the same shape. So you can cut the stack anywhere, put layers 0–23 on one
device and 24–27 on another, and ship the hidden state between them. **The
tensor is kilobytes; the weights are hundreds of megabytes.** That asymmetry is
the entire trick — it's why a phone can hold a slice of a model it could never
download in full.

---

## Two demos

### 1. Coordinator + phones (the real distributed one)

```bash
pip install -r requirements-phone.txt              # export tooling only
python3 build_shards.py --start 24 --end 27 --birds 1
python3 flock_export_coordinator.py --cut 24
cd node && npm install && npm start
```

Two phones? `--birds 2` splits the same range in half (126MB each) and each
device claims a free slot when it joins:

```bash
python3 build_shards.py --start 24 --end 27 --birds 2
```

- **phone** → `http://<your-ip>:8000/flock` → tap **join the flock**
- **any browser** → `http://<your-ip>:8000` → chat

Mac holds the embedding, layers 0–23, and the vocab projection. The phone holds
layers 24–27 and the final norm, executed by **ONNX Runtime Web on WebGPU**
(iOS 26+ enables WebGPU by default; older iOS falls back to WASM, slower but
working — the page tells you which backend it got).

### 2. Mac only, N processes (the concept, no phone needed)

```bash
uv tool install --with flask --with requests --with numpy mlx-lm
./run.sh 2      # or ./run.sh 3
```

Splits Qwen3-1.7B across N local processes via MLX. Useful for seeing the
layer split without a second device — but note it is **not** a distributed
test: one GPU, loopback traffic.

---

## Transport: f16 frames over WebRTC

Activations cross the network as **f16 binary frames**, not JSON floats:

```
fp32 JSON    21.0 KB per decode step
f16 binary    2.0 KB per decode step
```

On real activation values (|x| < 0.1, derived from bf16 weights) the f16 round
trip is effectively lossless. The Python and JS implementations are
byte-compatible and verified against each other in both directions.

The coordinator runs on **Node**, so it is itself a WebRTC peer: it offers a
data channel to each bird, and once that opens the websocket carries nothing
but offers/answers/ICE — the same role PeerJS plays for swarmllm. Replacing
long-polling with event-driven sockets took the roundtrip through two birds
from **132ms to ~5ms**.

Python survives only as the export and verification tool: PyTorch is the
reference every ONNX graph is checked against, so a bad export can't be
silently wrong.

## The KV cache is sharded too

Each device caches K/V for **only its own layers**, so conversation state is
sharded exactly like the weights are — no single device holds all of it.

An ONNX graph is static, so the phone's cache can't live inside it as hidden
state. Past K/V come in as explicit graph inputs, new K/V go out as outputs,
and the phone's JS holds them between steps. Measured effect per decode step:

```
              before          after
wire      64→100KB growing    flat 4.0KB
mac ms         ~81             ~28
```

## Design decisions worth understanding

**The phone never gets `lm_head`.** It's 151936×1024 (~600MB fp32) and is
*tied* to the embedding matrix the Mac already holds. Shipping it would
quadruple the phone's download to buy nothing, so the phone returns a hidden
state and the Mac does the vocab projection. First export was 874MB; this
brought it to 252MB.

**The phone long-polls `/work`.** A browser cannot accept inbound connections,
so the phone *asks* for activations rather than receiving them. That's the one
real concession to running a node inside a web page — swarmllm.ai avoids it
with WebRTC, at the cost of a signaling layer.

**A screen wake lock is held while joined.** iOS suspends a backgrounded or
locked tab: timers stop, fetches never return, and the node silently vanishes.
This is a genuine constraint of browser-based compute nodes, not an oversight.

**The export is verified against PyTorch** (`max abs err ~1e-4`) on every run,
so a phone can't be silently wrong.

---

## Limitations

Read these before drawing conclusions from it.

- **This is slower than either device alone.** Pipeline parallelism buys
  capacity, not throughput. Only one node computes at a time. That is the
  single most misunderstood thing about distributed inference, and you can feel
  it here in ~30 seconds.
- **Flask dev servers, no auth, no TLS, no input validation.** LAN-only toy.
  Do not expose to the internet.
- **No re-sharding on peer loss.** If a device leaves, generation stops with a
  message naming the uncovered layers, and resumes when *some* device claims
  that slot. It does not redistribute those layers onto the survivors — that
  redistribution is most of the real complexity in production P2P swarms.
- **fp32 weights.** No quantization on the phone shard, so the download is
  bigger than it needs to be.
- **Single conversation, no batching, no concurrency.**

---

## Files

| file | role |
|---|---|
| `build_shards.py` | splits a layer range across N birds, writes `web/flock.json` |
| `flock_export.py` | carves one layer range into ONNX; verifies vs PyTorch |
| `flock_export_coordinator.py` | exports the coordinator's embed/layers/head graphs |
| `node/src/server.js` | the coordinator: chat loop, signaling, HTTP |
| `node/src/coordinator.js` | embedding + layers 0–23 + vocab projection |
| `node/src/mesh.js` | the flock, slot claiming, WebRTC links |
| `node/src/wire.js` · `web/js/wire.js` · `flock/wire.py` | the f16 frame format |
| `web/bird.html` | a bird — ONNX Runtime Web on WebGPU |
| `web/chat.html` | chat UI showing the per-token lap |
| `flock_mlx_*.py` / `run.sh` | the Mac-only MLX demo |

## Things to try

- Move the split: `--start 20 --end 27` gives the phones 8 layers — watch them
  become the bottleneck.
- Kill a peer mid-generation: the swarm reports exactly which layers are
  uncovered, and recovers when a device claims that slot.
- Export with `--no-cache` and compare `wire KB` growth against the cached run.
- Compare `./run.sh 2` vs `./run.sh 3` tok/s. It gets *slower*.

## Prior art

- [swarmllm](https://github.com/Nehanth/swarmllm) — browser P2P via WebGPU + WebRTC; the direct inspiration
- [exo](https://github.com/exo-explore/exo) — Apple-silicon mesh, mDNS discovery
- [Petals](https://github.com/bigscience-workshop/petals) — BitTorrent-style, over the public internet
- [llama.cpp RPC](https://github.com/ggml-org/llama.cpp) — same idea in C++
- [prima.cpp](https://arxiv.org/pdf/2504.08791) — 30–70B on heterogeneous home clusters

## License

MIT
