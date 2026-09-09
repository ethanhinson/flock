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

### 1. Mac + phone (the real distributed one)

```bash
pip install -r requirements-phone.txt
python3 export_phone_shard.py --start 24 --end 27   # once, ~252MB ONNX
python3 swarm_phone.py
```

- **phone** → `http://<mac-ip>:8000/phone` → tap **join the swarm**
- **any browser** → `http://<mac-ip>:8000` → chat

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

- **No KV cache on the phone.** Each step re-runs the whole sequence — watch
  `wire KB` climb 64 → 68 → 72 → 100 as it generates. This is why it slows as
  output lengthens. Fixing it is ~10x faster and ~80 more lines on both sides;
  it was left out to keep the phone client ~150 readable lines.
- **This is slower than either device alone.** Pipeline parallelism buys
  capacity, not throughput. Only one node computes at a time. That is the
  single most misunderstood thing about distributed inference, and you can feel
  it here in ~30 seconds.
- **Flask dev servers, no auth, no TLS, no input validation.** LAN-only toy.
  Do not expose to the internet.
- **No fault tolerance.** Kill any node and generation stops. Real P2P swarms
  re-shard on peer loss, which is most of their actual complexity.
- **fp32 weights.** No quantization on the phone shard, so the download is
  bigger than it needs to be.
- **Single conversation, no batching, no concurrency.**

---

## Files

| file | role |
|---|---|
| `export_phone_shard.py` | carves N layers into ONNX for the phone; verifies vs PyTorch |
| `swarm_phone.py` | Mac side: embedding, layers 0–23, vocab projection, SSE chat |
| `phone_bridge.py` | work queue that lets a browser act as a pipeline peer |
| `web/phone.html` | the phone node — ONNX Runtime Web on WebGPU |
| `web/chat.html` | chat UI showing the per-token hop |
| `shard.py` / `swarm.py` / `run.sh` | the Mac-only MLX demo |

## Things to try

- Move the split: `--start 20 --end 27` gives the phone 8 layers — watch it
  become the bottleneck.
- Kill a node mid-generation and see the swarm stop.
- Compare `./run.sh 2` vs `./run.sh 3` tok/s. It gets *slower*.

## Prior art

- [swarmllm](https://github.com/Nehanth/swarmllm) — browser P2P via WebGPU + WebRTC; the direct inspiration
- [exo](https://github.com/exo-explore/exo) — Apple-silicon mesh, mDNS discovery
- [Petals](https://github.com/bigscience-workshop/petals) — BitTorrent-style, over the public internet
- [llama.cpp RPC](https://github.com/ggml-org/llama.cpp) — same idea in C++
- [prima.cpp](https://arxiv.org/pdf/2504.08791) — 30–70B on heterogeneous home clusters

## License

MIT
