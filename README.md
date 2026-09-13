# flock

flock runs one language model across the devices in the room: a laptop holds
the first few transformer layers and every phone, tablet or second laptop that
opens a web page holds a contiguous run of the rest, computing them as WebGPU
compute shaders and passing the hidden state along over WebRTC. It is a small,
real pipeline-parallel inference system -- Qwen3-0.6B today, greedy decode, one
conversation at a time -- built to make the ideas behind
[swarmllm](https://github.com/Nehanth/swarmllm) and
[exo](https://github.com/exo-explore/exo) concrete enough to read in an afternoon.

## What you need

- **Deno 2.1 or newer** for the coordinator, and **Node 20 + npm** for the
  dependencies and the tests. The coordinator runs under Deno because it holds
  layers on a GPU and Node has no WebGPU at all (checked: no `navigator`, no
  flag). Deno provides a real adapter and runs express, ws and node-datachannel
  unchanged through its Node compatibility layer, so the whole server is one
  process on one runtime.
- **A machine with a GPU** for the coordinator. Apple Silicon is what this was
  built on.
- **Devices with WebGPU in the browser.** Safari on iOS/iPadOS 26 and macOS 26
  has it on by default; on iOS/iPadOS 17 and 18 turn it on under Settings >
  Safari > Advanced > Feature Flags > WebGPU. Chrome and Edge have it, but only
  in a **secure context** -- see the certificate step below. There is no CPU
  fallback: a bird runs transformer layers as compute shaders or it does not run.
- **Everything on one wifi.** Devices reach the coordinator by its LAN address,
  and weights come straight from HuggingFace (639 MB once for the coordinator,
  a few tens of MB per device), so the first run also needs the internet.

## Run it

```bash
npm install
npm run cert      # once per machine; see why below
npm start
```

`npm start` reads the model's header, loads the coordinator's layers onto its
GPU (the whole GGUF is downloaded once into `kernels/.cache/`), and prints the
addresses to use:

```
  https://192.168.1.20:8000
flock listening on port 8000 -- chat at / , birds join at /flock, device check at /check
  birds share layers 4-27; however many devices show up, the split follows them
```

On each device open `https://<that address>/flock`. The page probes the GPU,
asks the coordinator for a range of layers, streams exactly those bytes out of
the GGUF file on HuggingFace straight into GPU buffers (about 4 MB at a time, so
a phone never holds a whole slice in memory), builds the layers, and reports
**ready**. Tap **join the flock**. Then open `https://<that address>/` anywhere
and type.

**The certificate step, and why.** Chrome and Edge expose WebGPU only to pages
in a secure context, and `http://192.168.1.20:8000` is not one -- on Chrome a
bird page over plain http sees no `navigator.gpu` and looks exactly like a
device with no GPU. Safari does not gate it this way, which is why the flock
worked on iPhones before anyone noticed. `npm run cert` writes a self-signed
certificate for your LAN address into `.certs/`; the coordinator serves https
whenever that directory exists, and each device has to open the address once
and accept the warning. If you only use Safari you can skip it and use `http://`.

Other things you can set:

| variable | default | what it does |
|---|---|---|
| `PORT` | 8000 | where the coordinator listens |
| `FLOCK_GGUF` | Qwen3-0.6B-Q8_0 on HuggingFace | the model every device reads |
| `FLOCK_COORD_LAYERS` | 4 | how many layers the coordinator keeps; the birds share the rest |
| `FLOCK_BIRD_LAYERS` | all but the coordinator's | an explicit size for the birds' share |
| `FLOCK_READY_WAIT_MS` | 90000 | how long a turn waits for reassigned devices to finish streaming |
| `FLOCK_NO_TLS` | | serve plain http even when `.certs/` exists (the test runner uses this) |
| `FLOCK_ONNX_REF` | `kernels/.ref` | where the ONNX reference export lives, for the engine's regression test |

No phone handy? `npm run solo` runs a simulated bird in one process that holds
every bird layer, with the same kernels; `npm run solo -- 3` runs three. Both
need this machine's GPU. `npm run fake -- --restream 3000` runs a bird with no
GPU that echoes the hidden state back and pretends each re-stream takes three
seconds -- useless for text, exactly right for testing the membership protocol.

## What you will see

On a bird page: the probe's numbers (how big a tensor this device can bind),
`joined as bird 1 -- I own layers 16-27`, a download bar that fills as the
layers stream, `ready: holding layers 16-27`, then a frame counter that ticks
once per token, the time each frame took, and the size of the K/V cache this
device -- and no other -- holds for the conversation.

On the chat page: a strip showing the coordinator and every bird with its
layers, the transport carrying its frames (`webrtc` once the data channel is
up, `ws` before), and per-token timings; a panel explaining why each device got
the share it did; and, under each reply, a table of every token's cost per hop.
If a device is still loading when you send, the reply shows `waiting for iPhone
to load layers 16-27` until it is not.

In the terminal: one line per join with the device's reported limits, the
split it produced and the predicted per-token cost, `[ready]` as each device
confirms its layers, and `[nack]` if a device ever refuses a frame.

`/status` has all of it as JSON; `/health` answers 200 when a prompt could be
served right now.

Measured on an M3 Max and an iPhone on the same wifi, Qwen3-0.6B, the phone
holding four layers over a WebRTC data channel:

```
decode averages over 59 tokens:
  coordinator (24 layers)      19.6 ms
  phone       ( 4 layers)      15.4 ms
  roundtrip incl. network      26.4 ms      60 tokens in 3.1s = 19.6 tok/s
  wire per token                2.0 KB      (f16; 21 KB as JSON floats)
```

The phone spends nearly as long on 4 layers as the laptop does on 24. That
asymmetry is what real heterogeneous hardware looks like, and it is why the
split is not even.

## How the layers are divided

The coordinator always holds the embedding, layers `0..FLOCK_COORD_LAYERS-1`,
the final norm and the tied vocabulary projection, because the projection is
the same 155 MB matrix as the embedding and shipping it to a phone would buy
nothing. Everything else is divided among whoever is present, by
`server/allocate.js`:

- **By bytes, not layer count.** Layers are not the same size (Qwen3-14B's range
  185-210 MB), so every budget and weight is in bytes.
- **By measured speed.** Each device's throughput is tracked as bytes per
  millisecond from its own per-token timings (a rate, so it does not change when
  the device's share does), smoothed, and not trusted until it has three
  samples. A pipeline runs at the pace of its slowest stage, so the objective is
  the **makespan**: the split that minimises the slowest device's time, found
  exactly by dynamic programming over contiguous runs.
- **Under each device's real limits.** A device reports its
  `maxStorageBufferBindingSize` before it joins; a layer whose largest tensor
  exceeds it is never assigned to that device, and a flock that cannot hold the
  model is told so -- naming the tensor and both numbers -- before any weights
  download.
- **The coordinator's machine is charged for its own work.** A browser on the
  coordinator's machine can join as a bird; its GPU is already doing the
  embedding and the projection, so it is charged that time (`reservedMs`)
  rather than being handed the biggest share for looking fastest.

Two rules keep that from thrashing, and both were learned the hard way:

- **A join or a leave is a local edit, not a re-partition.** A device that joins
  takes a contiguous run off one incumbent; a device that leaves hands its run
  to its neighbours; nobody else moves. Every device that moves has to
  re-stream its weights, and holds nothing for the 3-18 seconds that takes, so
  the number of movers is the real cost of a change.
- **Speed rebalancing never fires during a conversation.** Timings are recorded
  every token, but acting on them moves layers, and the K/V cache is sharded by
  layer, so a move drops the conversation. The exact split is only applied when
  a conversation starts (the clear button, or the first message), and only if
  it promises at least 15% and the last move was 20 s ago.

A device that has been assigned layers is not yet holding them. Every bird
confirms its range once the weights are on its GPU, the coordinator sends
nothing to a device that has not confirmed, and a turn waits -- saying who it
is waiting for -- rather than failing. A join that arrives while an answer is
being generated is applied when the answer finishes. This is the protocol that
made the flock usable: before it, one phone joining moved every device, the
next frame arrived at one that was still streaming, that device left, and the
leave moved everyone again.

## What it cannot do

- **A reassignment drops the conversation.** Each device caches keys and values
  for its own layers only, so when a layer moves its cache stays behind. The
  turn after a join or a leave starts from an empty context, and the chat page
  says so. Re-feeding the history is not an option either: the chat template
  scaffolds the current turn differently from past ones, so a re-encoded
  conversation diverges from what was actually fed.
- **The first turn after a change waits.** A moved device streams its new
  layers first; a phone takes 3-18 seconds for a few layers, longer for a big
  share.
- **It is slower than either device alone.** Pipeline parallelism buys
  capacity, not throughput; one device computes at a time. Adding a device
  lowers tokens per second. What it buys is running a model no single device
  could hold.
- **No CPU fallback, WebGPU required.** Chrome needs https; iOS before 26 needs
  the feature flag; a device without either cannot be a bird, and the page says
  which it is missing.
- **Greedy decoding only.** No temperature, no sampling. Decode attention caps
  at 2048 keys.
- **One conversation.** A second `/chat` while one is generating is refused.
  The context only grows until it is cleared.
- **The coordinator's share is fixed at startup.** Its layers are loaded once
  onto its GPU; only the birds' share is divided live.
- **LAN only.** No authentication and almost no input validation. Do not expose
  it to the internet.

## Layout

```
server/    the coordinator: HTTP + websocket signaling + the chat loop
           (server.js), its layers and tokenizer (coordinator.js), the flock
           and the WebRTC links (mesh.js), who holds what (allocate.js),
           whether the timings justify moving anything (speed.js), and the
           GGUF directory reader (gguf.mjs)
web/       what a device opens: bird.html (a bird), chat.html, inspect.html
           (/check); web/js/ holds the modules the browser and the server share
           -- the f16 wire format, the GGUF streamer, the capability probe
kernels/   the WGSL engine: nine compute kernels, Layer, Model, and the suite
           that validates them against strict-f32 references and an ONNX export
tools/     sim_bird.mjs (a bird without a browser), plan.mjs (what each device
           would fetch), check_html.mjs, cert.sh
test/      unit/ (node, no GPU), e2e/ (need a live coordinator), gpu/ (need a
           device), and run.mjs, which runs all of it
```

The split is by role rather than by runtime because the runtime is the same
everywhere: the server is Deno, the birds are browsers running the very files
under `kernels/` (transpiled on request, no build step), and the tests are
Node where they can be and Deno where they need a GPU. `kernels/` stays its own
directory with its own [README](kernels/README.md) because it is a self-contained
piece of work -- the engine, its validation, and the traps found on the way --
that is worth reading on its own.

## Tests

```bash
npm test               # everything, with a summary table at the end
npm run test:unit      # node, no GPU: the allocator, the flock, the probe, the
                       # bird page's load path, both GGUF parsers (network)
npm run test:e2e       # starts its own coordinator on a spare port and its own
                       # simulated birds, and kills them all when done
npm run test:kernels   # the WGSL engine's 13 suites, 316 assertions, on the GPU
npm run test:gpu       # the streaming loader against real hardware and the real file
```

The e2e group is the one that matters for the membership protocol.
`test/e2e/churn.test.mjs` runs real `tools/sim_bird.mjs` processes that hold
nothing for 2.5 s after every assignment, joins devices between turns and
mid-turn, and asserts on what each bird *says it holds* and on whether a
five-turn chat completes -- because the cascade it guards against was
invisible to every `/status` check. The runner never touches port 8000, where
a real flock might be, and every simulated bird it starts is in its own
process group and killed on the way out; stand-ins left holding layers have
blocked real phones from joining before.

`onnxruntime-node` is a devDependency for one reason: `kernels/test_model.ts`
diffs the WGSL engine, token for token, against an ONNX export of the same
model, the only implementation in the repo that shares none of its
assumptions. The export is not in the repo; the test skips when it is absent.

## Prior art

- [swarmllm](https://github.com/Nehanth/swarmllm) -- browser P2P via WebGPU + WebRTC; the direct inspiration
- [exo](https://github.com/exo-explore/exo) -- Apple-silicon mesh, mDNS discovery
- [Petals](https://github.com/bigscience-workshop/petals) -- BitTorrent-style, over the public internet
- [llama.cpp RPC](https://github.com/ggml-org/llama.cpp) -- the same idea in C++
- [prima.cpp](https://arxiv.org/pdf/2504.08791) -- 30-70B on heterogeneous home clusters

## License

MIT
