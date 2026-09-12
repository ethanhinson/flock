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
cd node && npm install && npm start
```

That is the whole setup. **There is no build step** — no export, no Python, no
artifacts on disk. The coordinator reads Q8_0 weights straight out of a GGUF file
and runs Qwen3's first 24 layers as WGSL compute kernels; each bird range-fetches
its own layers from HuggingFace.

`npm start` runs **Deno**, not Node, and that is a requirement rather than a
preference: the coordinator needs a GPU and Node has no WebGPU at all. See
[Why the server runs under Deno](#why-the-server-runs-under-deno).

`npm start` prints the LAN addresses to use:

- **phone** → `http://<your-ip>:8000/flock` → tap **join the flock**
- **any browser** → `http://<your-ip>:8000` → chat

The Mac holds the embedding, layers 0–23, `output_norm` and the tied vocab
projection. The phone holds layers 24–27, executed as **WGSL compute shaders on
WebGPU** — the same kernels the test suite validates against ONNX token for token.
WebGPU is required on a bird; there is no CPU fallback any more, and the page says
so plainly rather than promising a slower one.

Two phones? `FLOCK_BIRDS=2` splits the same range in half and each device claims a
free slot when it joins. Because a split is now just a choice about byte ranges,
this is an environment variable rather than a rebuild:

```bash
FLOCK_BIRDS=2 npm start                 # 2 birds, 2 layers each
FLOCK_BIRD_LAYERS=8 npm start           # give the birds 8 layers instead of 4
FLOCK_GGUF=<url> npm start              # a different GGUF
```

No phone handy? `npm run solo` runs every bird in one process, so the whole chain
works on one machine:

```bash
npm start                   # one shell
npm run solo                # another: claims every layer slot
npm run health              # is the flock covered?
npm test                    # syntax + range planning + the bird's load path
npm run test:ui             # drives both pages against the live coordinator
```

Set `PORT` to run somewhere other than 8000.

---

---

## Transport: f16 frames over WebRTC

Activations cross the network as **f16 binary frames**, not JSON floats:

```
fp32 JSON    21.0 KB per decode step
f16 binary    2.0 KB per decode step
```

On real activation values (|x| < 0.1, derived from bf16 weights) the f16 round
trip is effectively lossless. There is exactly one implementation
(`web/js/wire.mjs`), imported by the server and by the browser pages alike, so the
two ends cannot disagree about the format.

The coordinator is a real WebRTC peer: it offers a data channel to each bird, and
once that opens the websocket carries nothing but offers/answers/ICE — the same
role PeerJS plays for swarmllm. Replacing long-polling with event-driven sockets
took the roundtrip through two birds from **132ms to ~5ms**.

## Weights come straight from HuggingFace, with no build step

There used to be a Python build step here: export the layer range with
`torch.onnx.export`, quantize it, write 63MB per bird to disk, and have the
coordinator serve those bytes. A GGUF file needs none of that. The first ~6MB of
it is a directory listing every tensor's exact byte offset, and everything after
is raw weights — so **every device range-fetches its own layers directly from the
model host** and the coordinator never touches them.

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

**This is now the only path.** ONNX is gone from inference entirely — no
`onnxruntime-web` in the bird page, no `onnxruntime-node` in the coordinator, no
exported shards. What remains of it is test-only: see
[ONNX survives as the reference](#onnx-survives-as-the-reference).

## The KV cache is sharded too

Each device caches K/V for **only its own layers**, so conversation state is
sharded exactly like the weights are — no single device holds all of it.

The cache now lives in GPU memory inside each `Layer` object and is never
marshalled: a decode step appends one key and one value to a device buffer, and
the host sees neither. That is a simplification the WGSL engine bought — under
ONNX the graph was static, so past K/V had to come in as explicit graph inputs and
new K/V go out as outputs, with the page's JS holding a dict of tensors between
steps. Measured effect per decode step, when the cache was first added:

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
state and the Mac does the vocab projection. A bird's whole download is 66.9MB in
one range request.

**A browser cannot accept inbound connections.** The original design had the
phone long-poll for work, which cost ~132ms per token. Now the coordinator is
itself a WebRTC peer: it *offers* each bird a data channel over the websocket,
and once that opens the websocket carries only signaling. This is the same role
PeerJS plays for swarmllm, and it is why the network overhead is now ~11ms.

**A screen wake lock is held while joined.** iOS suspends a backgrounded or
locked tab: timers stop, fetches never return, and the node silently vanishes.
This is a genuine constraint of browser-based compute nodes, not an oversight.

**A bird's weights never sit in the JS heap.** Each ~4MB staging chunk goes
straight into its GPU buffer as it streams, because the obvious loader
(`arrayBuffer()` then `writeBuffer`) held a 126MB shard while the driver made its
own copy — ~250MB peak, which is what killed an iPad.

---

## Why the server runs under Deno

The coordinator holds layers 0–23, so it needs a GPU. **Node has no WebGPU.**
Measured on node v20.11.0: there is no `navigator` at all, `globalThis.GPUDevice`
is undefined, and `--experimental-webgpu` is not a Node flag (that is Deno's —
Node rejects it with `bad option`).

Deno 2.1.2 solves it with no compromise. It provides a real adapter whose
`maxStorageBufferBindingSize` is 4 GiB — which the 155.6MB tied LM head needs,
because overflowing the 128 MiB default is a **silent** wrong answer rather than a
throw — and it runs the rest of the stack unchanged through its node compatibility
layer: `express`, `ws`, `node-datachannel` and `@huggingface/transformers` all
import and work. So the whole server stayed one process on one runtime.

The alternatives were considered and are worse. A browser-hosted coordinator
(swarmllm's design) makes the thing you must keep open a browser tab rather than a
server. A CPU coordinator means writing a second implementation of 24 layers that
nothing validates. Neither was necessary.

## ONNX survives as the reference

Nothing in the inference path imports ONNX. But `onnxruntime-node` is kept as a
**devDependency**, and that is deliberate: `kernels/onnx_full.mjs` runs the whole
five-graph ONNX pipeline under Node, and `kernels/test_model.ts` diffs the WGSL
engine against it token for token. It is the only *independent* implementation in
the repo.

Delete it and the engine could only ever be compared against itself — a CPU
reference built from the same per-op references shares their assumptions, and
`kernels/README.md` documents a real bug (RoPE's pairing convention) that exactly
that kind of self-consistency test could not see. So the regression proof is worth
keeping a test-only dependency and a set of gitignored graphs for.

```
prompt  "Capital of France?"
WGSL    [785,6722,315,9625,374,3070,59604,334,13]   "The capital of France is **Paris**."
ONNX    [785,6722,315,9625,374,3070,59604,334,13]   identical, both stop at EOS
```

One honest caveat about "no ONNX in the process": `@huggingface/transformers`
bundles its own `onnxruntime-node` and dlopens `libonnxruntime` when imported, so
the library *is* mapped into the coordinator — `lsof` shows it. Nothing ever asks
it to run anything. The only API used from that package is `AutoTokenizer`, never
`AutoModel` or `pipeline()`, so no `InferenceSession` is constructed and no ONNX
graph is ever loaded. Removing the mapping would mean hand-rolling a tokenizer and
moving the chat template — the one thing that must match the reference exactly —
into new code, which buys nothing.

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
| `kernels/` | the WGSL engine: 9 compute kernels, `Layer`, `Model`, and their tests |
| `kernels/README.md` | what each kernel is validated against, and ten traps worth reading |
| `node/src/server.js` | the coordinator: chat loop, signaling, HTTP, kernel serving |
| `node/src/coordinator.js` | embedding + layers 0–23 + output_norm + tied head |
| `node/src/mesh.js` | the flock, slot claiming, chain topology, WebRTC links |
| `node/src/gguf.mjs` | GGUF range planning: which bytes does each device need |
| `web/js/wire.mjs` | the f16 frame format — one copy, imported by server and browsers |
| `web/js/gguf-dir.mjs` | reads a GGUF directory in a browser, over range requests |
| `web/js/gguf-stream.mjs` | streams GGUF tensors into GPU buffers, ~4MB at a time |
| `web/bird.html` | a bird — WGSL compute shaders on WebGPU |
| `web/chat.html` | chat UI: conversation, live topology, per-token stats |
| `node/sim_bird.mjs` | a bird without a browser, for testing the chain |
| `node/test/` | range planning, and both pages driven against a live coordinator |

## Things to try

- Move the split: `FLOCK_BIRD_LAYERS=8` gives the phones 8 layers — watch them
  become the bottleneck. No rebuild; a split is just a byte range now.
- Kill a peer mid-generation: the swarm reports exactly which layers are
  uncovered, and recovers when a device claims that slot.
- Add a second device with `FLOCK_BIRDS=2` and watch tok/s go *down* — pipeline
  parallelism buys capacity, not speed.
- Run `kernels/test_model.ts` to see the WGSL engine and the ONNX pipeline
  generate the same token ids, then read `kernels/README.md` on why a per-token
  cosine that is perfect at position 0 and decays after it indicts RoPE.

## Prior art

- [swarmllm](https://github.com/Nehanth/swarmllm) — browser P2P via WebGPU + WebRTC; the direct inspiration
- [exo](https://github.com/exo-explore/exo) — Apple-silicon mesh, mDNS discovery
- [Petals](https://github.com/bigscience-workshop/petals) — BitTorrent-style, over the public internet
- [llama.cpp RPC](https://github.com/ggml-org/llama.cpp) — same idea in C++
- [prima.cpp](https://arxiv.org/pdf/2504.08791) — 30–70B on heterogeneous home clusters

## License

MIT
