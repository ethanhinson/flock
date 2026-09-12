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

M3 Max + an iPhone on the same wifi, Qwen3-0.6B split 24/4, all links WebRTC:

```
tok    coord ms  phone ms  wire KB  roundtrip
1          57.0     151.0     58.0      280.3   <- prefill
2          19.3      13.0      2.0       22.7
3          18.3      12.0      2.0       18.8
...
60         18.7      13.0      2.0       19.3

decode averages over 59 tokens:
  coordinator (24 layers, node)  :  19.6 ms
  phone       ( 4 layers, webgpu):  15.4 ms
  roundtrip incl. network        :  26.4 ms
  network overhead               :  11.0 ms
  wire per token                 :   2.0 KB f16  (21.0 KB as json)

  60 tokens in 3.1s = 19.6 tok/s
  roundtrip min 18.1 / median 22.7 / max 131.5 ms
```

Every one of those 60 hops crossed a **WebRTC data channel** to a real iPhone.

The phone runs 4 of 28 layers and spends nearly as long as the coordinator's
24 — that asymmetry is what real heterogeneous hardware looks like. The 131ms
outlier is wifi, not compute: it is the honest tail of a wireless link.

For contrast, the same workload under the original long-polling + fp32-JSON
design ran at ~132ms of network overhead per token. It is now 11ms.

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

## Run it

```bash
pip install -r requirements-export.txt          # export tooling only
python3 build_shards.py --start 24 --end 27 --birds 1 --int8
python3 flock_export_coordinator.py --cut 24
cd node && npm install && npm start
```

Two phones? `--birds 2` splits the same range in half (126MB each) and each
device claims a free slot when it joins:

```bash
python3 build_shards.py --start 24 --end 27 --birds 2 --int8
```

`--int8` quantizes the weights: a bird downloads **63MB instead of 252MB** and
generates *identical* text (verified by a full greedy decode against the fp32
shard). This is what makes the ONNX path competitive with GGUF on size without
writing a single GPU kernel — ONNX Runtime already ships quantized matmul.

`npm start` prints the LAN addresses to use:

- **phone** → `http://<your-ip>:8000/flock` → tap **join the flock**
- **any browser** → `http://<your-ip>:8000` → chat

Mac holds the embedding, layers 0–23, and the vocab projection. The phone holds
layers 24–27 and the final norm, executed by **ONNX Runtime Web on WebGPU**
(iOS 26+ enables WebGPU by default; older iOS falls back to WASM, slower but
working — the page tells you which backend it got).

No phone handy? `npm run solo` runs every bird in one Node process, so the whole
chain works on one machine:

```bash
npm start                   # one shell
npm run solo                # another: claims every layer slot
npm run health              # is the flock covered?
npm test                    # syntax + range planning
npm run test:ui             # drives both pages against the live coordinator
```

Set `PORT` to run somewhere other than 8000.

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

## Weights can come straight from HuggingFace, with no build step

The ONNX path above needs Python to run first: export the layer range,
quantize it, write 63MB per bird to disk, and let the coordinator serve those
bytes. A GGUF file does not need any of that. The first ~6MB of it is a
directory listing every tensor's exact byte offset, and everything after is raw
weights — so **a bird range-fetches its own layers directly from the model
host** and the coordinator never touches them.

```bash
FLOCK_GGUF=https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf npm start
# then open the bird page with ?gguf=1
```

Four layers of Qwen3-0.6B is **66.9MB in one HTTP range request** — one, not
44, because layers are laid out contiguously in GGUF and the loader opens a
single response over the merged range and hands it to each tensor in turn.
Removing those 43 round trips also made the download 5x faster.

It also makes the split *dynamic*: layer ranges are just byte ranges, so the
coordinator can decide them when devices join instead of baking them into
exported files.

### Peak memory is the staging buffer, not the tensor

The obvious loader — `await res.arrayBuffer()` then `writeBuffer` — holds the
whole slice in JS while the driver makes its own copy. That is what killed an
iPad on the ONNX path: ~250MB peak for a 126MB shard. So nothing here ever
holds a tensor. The response is read as a **stream** and each 4MB staging chunk
is copied into the GPU buffer as it arrives, de-interleaved on the way through
into the separate `qs`/`scales` buffers the WGSL kernels want.

Measured against the real 639MB file, comparing JS memory held per byte
delivered:

```
                        slice     peak JS    per byte
naive (arrayBuffer)     16.7MB     30.3MB      1.81x
streaming               66.9MB     15.7MB      0.24x
```

Streaming four layers costs *less* JS memory than holding one. Above a measured
floor for Deno's own fetch buffering, the loader's share is 3.7MB — the
coalescing buffer plus one flush of quants and scales, independent of slice
size, which is the whole claim.

The bytes are verified byte-identical to `fetchRange` + `splitQ8`, the reference
the kernels are already validated against, for every tensor of a layer.

`npm run test:gpu` runs all of it against real hardware and the real file.

**The ONNX path is untouched and still the default.** It is the reference the
WGSL engine is validated against, so the GGUF path is opt-in until that engine
lands, and a GGUF failure logs and falls through rather than costing a bird its
slot.

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

Conversations continue against that warm cache: turn two feeds only the new
tokens, not the whole history. The subtle part is that you cannot simply
re-encode the conversation each turn — with `enable_thinking:false` the Qwen3
chat template injects a `<think></think>` scaffold for the *current* turn only,
so a re-encoded history diverges from what was already fed (measured: the two
token streams split at token 9 of 13) and every cache would be keyed to
different tokens. So the coordinator appends exactly what the model saw next:

```
turn 1  "My favorite color is blue."   21 prompt tokens   cache 35
turn 2  "What is my favorite color?"   20 prompt tokens   cache 64   -> "blue"
turn 3  "Name one fruit of that color" 21 prompt tokens   cache 101  -> "blueberry"
```

Turn 2 fed 20 tokens instead of re-prefilling 55. `POST /reset` drops every
device's shard of the cache.

## Design decisions worth understanding

**The phone never gets `lm_head`.** It's 151936×1024 (~600MB fp32) and is
*tied* to the embedding matrix the Mac already holds. Shipping it would
quadruple the phone's download to buy nothing, so the phone returns a hidden
state and the Mac does the vocab projection. First export was 874MB; this
brought it to 252MB.

**A browser cannot accept inbound connections.** The original design had the
phone long-poll for work, which cost ~132ms per token. Now the coordinator is
itself a WebRTC peer: it *offers* each bird a data channel over the websocket,
and once that opens the websocket carries only signaling. This is the same role
PeerJS plays for swarmllm, and it is why the network overhead is now ~11ms.

**A screen wake lock is held while joined.** iOS suspends a backgrounded or
locked tab: timers stop, fetches never return, and the node silently vanishes.
This is a genuine constraint of browser-based compute nodes, not an oversight.

**The export is verified against PyTorch** (`max abs err ~1e-4`) on every run,
so a phone can't be silently wrong.

---

## Why there is any Python

Only at build time, and only because there is no alternative: `torch.onnx.export`
has no JavaScript equivalent. ONNX Runtime *runs* models in every language, but
Python is the only thing that *produces* them from PyTorch weights.

Keeping it also buys something real — PyTorch is the reference every exported
graph is checked against (`max abs err ~1e-4` on every build), which is what
catches an export that would otherwise produce plausible-looking garbage.

Nothing Python runs at inference time. Export once, then `npm start`.

## Limitations

Read these before drawing conclusions from it.

- **This is slower than either device alone.** Pipeline parallelism buys
  capacity, not throughput. Only one node computes at a time. That is the
  single most misunderstood thing about distributed inference, and you can feel
  it here in ~30 seconds.
- **No auth, no TLS, barely any input validation.** LAN-only toy. Do not expose
  to the internet.
- **No re-sharding on peer loss.** If a device leaves, generation stops with a
  message naming the uncovered layers, and resumes when *some* device claims
  that slot. It does not redistribute those layers onto the survivors — that
  redistribution is most of the real complexity in production P2P swarms.
- **One conversation, no batching, no concurrency.** Turns are multi-turn and
  continue from the warm cache, but there is exactly one of them: a second
  simultaneous `/chat` is refused rather than interleaving writes into the same
  K/V cache. The context also only grows — nothing evicts it but `/reset`.

---

## Files

| file | role |
|---|---|
| `build_shards.py` | splits a layer range across N birds, writes `web/flock.json` |
| `flock_export.py` | carves one layer range into ONNX; verifies vs PyTorch |
| `flock_export_coordinator.py` | exports the coordinator's embed/layers/head graphs |
| `node/src/server.js` | the coordinator: chat loop, signaling, HTTP |
| `node/src/coordinator.js` | embedding + layers 0–23 + vocab projection |
| `node/src/mesh.js` | the flock, slot claiming, chain topology, WebRTC links |
| `web/js/wire.mjs` | the f16 frame format — one copy, imported by node and browsers |
| `web/js/gguf-dir.mjs` | reads a GGUF directory in a browser, over range requests |
| `web/js/gguf-stream.mjs` | streams GGUF tensors into GPU buffers, ~4MB at a time |
| `web/bird.html` | a bird — ONNX Runtime Web on WebGPU |
| `web/chat.html` | chat UI: conversation, live topology, per-token stats |
| `node/sim_bird.mjs` | a bird without a browser, for testing the chain |
| `node/test/` | range planning, and both pages driven against a live coordinator |

## Things to try

- Move the split: `--start 20 --end 27` gives the phones 8 layers — watch them
  become the bottleneck.
- Kill a peer mid-generation: the swarm reports exactly which layers are
  uncovered, and recovers when a device claims that slot.
- Export with `--no-cache` and compare `wire KB` growth against the cached run.
- Add a second device with `--birds 2` and watch tok/s go *down* — pipeline
  parallelism buys capacity, not speed.

## Prior art

- [swarmllm](https://github.com/Nehanth/swarmllm) — browser P2P via WebGPU + WebRTC; the direct inspiration
- [exo](https://github.com/exo-explore/exo) — Apple-silicon mesh, mDNS discovery
- [Petals](https://github.com/bigscience-workshop/petals) — BitTorrent-style, over the public internet
- [llama.cpp RPC](https://github.com/ggml-org/llama.cpp) — same idea in C++
- [prima.cpp](https://arxiv.org/pdf/2504.08791) — 30–70B on heterogeneous home clusters

## License

MIT
