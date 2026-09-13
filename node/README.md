# flock (server runtime)

The all-JavaScript flock: the coordinator is a real WebRTC peer, so every link in
the chain is a data channel and the websocket carries only signaling.

```bash
cd node && npm install
npm start
```

There is **no build step**. No export, no Python, no artifacts on disk: the
coordinator reads Q8_0 weights straight out of a GGUF file and runs Qwen3's first
24 layers as WGSL compute kernels, and each bird range-fetches its own layers from
HuggingFace. The layer split is computed from the model's header at startup, so
changing it is an environment variable rather than a rebuild.

## It runs under Deno, not Node

`npm start` invokes Deno. That is not a preference, it is a requirement, and the
reason is worth stating because it drove the design:

**The coordinator needs a GPU, and Node has no WebGPU.** Measured on node
v20.11.0 — there is no `navigator` at all, `globalThis.GPUDevice` is undefined,
and `--experimental-webgpu` is not a Node flag (that is Deno's; Node rejects it
with `bad option`).

Deno 2.1.2 solves it without a compromise. It provides a real adapter with
`maxStorageBufferBindingSize` of 4 GiB — which the 155.6 MB tied LM head needs,
since overflowing the 128 MiB default is a **silent** wrong answer rather than a
throw — and it runs the rest of the stack unchanged through its node
compatibility layer: `express`, `ws`, `node-datachannel` and
`@huggingface/transformers` all import and work. So the whole server is one
process on one runtime.

The alternatives are worse. A browser-hosted coordinator (what swarmllm does)
makes the thing you must keep open a browser tab rather than a server. A CPU
coordinator means a second implementation of 24 layers that nothing validates.
Both were avoidable at no cost.

The npm dependencies are still npm dependencies, installed by `npm install`;
`deno.json` sets `nodeModulesDir: "manual"` so Deno resolves them out of
`node_modules` rather than fetching its own copies.

## Configuration

| variable | default | what it does |
|---|---|---|
| `PORT` | 8000 | HTTP port |
| `FLOCK_GGUF` | Qwen3-0.6B-Q8_0 on HuggingFace | where every device reads weights |
| `FLOCK_BIRD_LAYERS` | 4 | how many layers the birds hold in total |

**There is no `FLOCK_BIRDS`.** It used to fix the device count at startup, so a third
phone pointed at a coordinator started with 2 got "flock full" and changing the count
meant a restart. Any number of devices can now join, and the layers are re-split
across whoever is present. `FLOCK_BIRD_LAYERS` is still a startup choice because the
coordinator's half is loaded onto this GPU once and cannot move.

## How the split is decided

Not evenly, and not by layer count. Three measured reasons:

- **Devices differ by ~5x.** A phone did 4 layers in 15.4 ms; this Mac did 24 in
  19.6 ms. A pipeline runs at the pace of its slowest stage, so an even split wastes
  the fast device.
- **Layers are not the same size.** Qwen3-14B Q4_K_M layers range **185.8-210.2 MB**,
  a 13% spread, so "half the layers" is not "half the bytes".
- **Some tensors cannot be divided and are huge.** That model's `output.weight` is
  **638 MB as one tensor** and WebGPU's default `maxStorageBufferBindingSize` is
  128 MiB, so a device can be excluded from a layer at any split.

So `src/allocate.js` minimises the **makespan** -- `max(bytes assigned / measured
bytes-per-ms)` -- over contiguous runs of layers, subject to each device's real
`maxStorageBufferBindingSize` and memory budget. It is an exact DP, not a heuristic
(`test/allocate.test.mjs` brute-forces it against an exhaustive search).

`src/speed.js` decides whether to act on the timings, and it is built against
**oscillation** rather than slowness:

| brake | why |
|---|---|
| a **rate** (bytes/ms), not a time | size-invariant, so the controller cannot read its own last decision as new evidence |
| EWMA, α = 0.25 | one slow token (a GC pause, the user switching apps) moves the estimate by a quarter, not all the way |
| 3 samples before a rate is trusted | the first token after a join cannot trigger a reshuffle |
| a 15% predicted-gain gate | near the optimum every move is a small gain, every small gain is refused, and the assignment stops moving |
| a 20 s cooldown | a move costs the moved layers a re-download and the flock its K/V cache |

Measured: the same closed loop on two similar-speed devices makes **56 moves in 300
tokens** with the gates off and **1** with them on.

`/status` reports the whole decision -- each device's bytes, share, measured rate,
sample count and reported limits, plus the predicted per-token cost next to what an
even split of the same layers would have cost. The chat page renders it under the
topology strip.

## Joining and leaving mid-generation

Membership can change at any time; the **assignment** only changes at a token
boundary. The token in flight completes against the topology it started with.

Any reallocation that actually moves layers **drops the conversation context**, and
this is a deliberate, reported cost rather than a bug. The K/V cache is sharded *by
layer*, so a layer that moves leaves its keys on the device that no longer holds it,
while the device that now does starts from an empty cache at an offset the rest of the
flock believes is already filled. Continuing would mix keys for the same positions
computed on two different devices -- text that is wrong without looking wrong.
Re-prefilling the history instead is not available: `Coordinator.turnTokens` documents
why re-encoding a conversation diverges from what was actually fed.

So:

| event | what happens |
|---|---|
| join between turns | layers re-split immediately; context dropped if anything moved |
| join mid-token | staged; the token finishes, then the split lands, the turn ends with `stop: "rebalanced"` and a `note` saying the context was dropped |
| `POST /leave` | layers handed on at once (or at the next boundary if a token is in flight) |
| socket closes | **nothing**, for up to 40 s. A refresh and a departure look identical, and reallocating on the close would drop the conversation for what is about to be the same device holding the same range. The sweeper removes it once it has really been silent. |
| a device stops answering mid-lap | the turn fails naming the device and the layers it held; the caches are reset |
| a device joins that makes the flock unsatisfiable | it is **refused and un-admitted**, so one bad device cannot poison a working flock |

## Scripts

| command | what it does |
|---|---|
| `npm start` | run the coordinator (`PORT` to move off 8000) |
| `npm run solo` | one process holding every layer — no phone needed; `-- solo 3` for three devices |
| `npm run bird` | one simulated bird |
| `npm run fake` | a bird that reports arbitrary limits and speed and needs no GPU |
| `npm run health` | is the flock covered? exits non-zero if not |
| `npm run status` | full state: birds, transports, cache, recent device errors |
| `npm test` | syntax, allocation, membership, the capability probe, GGUF range planning, both parsers cross-checked, the bird's load path |
| `npm run test:membership` | dynamic membership against a live coordinator (fake birds, no GPU) |
| `npm run test:ui` | drives chat.html and bird.html against a live coordinator |
| `npm run test:gpu` | streams real weights into real GPU buffers |
| `npm run plan` | what each device would fetch, reading only a GGUF header |

`npm test` runs under Node and needs no GPU, so it works in CI and over ssh.
`npm run test:ui` and `test:gpu` need Deno and a real device: `bird_ui.test.mjs`
runs bird.html's own module with a real WebGPU device and real streamed weights,
so it covers the inference and not only the UI state.

## Why onnxruntime-node is still a devDependency

Nothing in the inference path imports it. It is kept so
`kernels/onnx_full.mjs` and `kernels/onnx_truth.mjs` still run, because those are
the **independent reference** that `kernels/test_model.ts` diffs the WGSL engine
against token for token. Delete it and the engine could only ever be compared
against itself, which would catch no future kernel regression at all.

## Layout

| file | role |
|---|---|
| `src/server.js` | HTTP, SSE chat loop, websocket signaling, kernel serving |
| `src/coordinator.js` | embedding, layers 0..cut-1, output_norm, tied head, tokenizer |
| `src/mesh.js` | the flock: slot claiming, liveness, chain topology, WebRTC links |
| `src/allocate.js` | who holds which layers: makespan over bytes, under each device's limits |
| `src/speed.js` | per-token timings -> a rate, and whether to act on it |
| `src/gguf.mjs` | GGUF range planning over HTTP range requests (npm import) |
| `../kernels/` | the WGSL engine — the same kernels the test suite validates |
| `../web/js/probe.mjs` | what a device can do: GPU limits plus a real compute pass |
| `../web/inspect.html` | that probe as a page, at `/check` |
| `../web/js/gguf-dir.mjs` | the same directory parse, browser-safe |
| `../web/js/gguf-stream.mjs` | streams GGUF tensors from HuggingFace into GPU buffers |
| `sim_bird.mjs` | a bird without a browser |
| `check_html.mjs` | `node --check` over the inline `<script>` of each page |

## Endpoints

| route | purpose |
|---|---|
| `GET /` | the chat page |
| `GET /flock` | the bird page |
| `GET /kernels/*.wgsl` | the compute kernels, as-is |
| `GET /kernels/*.ts.js` | the same kernels, transpiled on request for browsers |
| `GET /check` | what can this device do? the same probe a bird runs before joining |
| `POST /join` | join the flock with your measured `caps`; get a range, model dims and the GGUF url. Answers `{wait: true}` if a token is in flight (send the `peer_id` back when you retry) |
| `POST /leave` | leave on purpose, so the layers are handed on without waiting out the 40 s grace period |
| `POST /evict` | throw a device out from the coordinator side |
| `POST /chat` | SSE: `turn`, `hop`, `token`, `rebalanced`, `done`, `error` |
| `POST /reset` | drop every device's shard of the KV cache |
| `GET /status` | birds, transports, cached tokens, the full allocation decision, recent `/diag` reports |
| `GET /health` | 200 when the layers are covered, 503 when not |
| `POST /diag` | where birds report failures; phones have no readable console |
| `GET /ws` | websocket: hello, signaling, `ping`, `stats`, `caps`, `chain` (which can carry a NEW range), binary frames |

### Serving TypeScript to a browser

Browsers do not execute TypeScript, and adding a bundler would put back the build
step this design removes. So `/kernels/x.ts.js` transpiles `kernels/x.ts` on
request (`@deno/emit`, in memory, keyed by mtime) and rewrites relative `./y.ts`
specifiers to `./y.ts.js` so the browser walks the same module graph Deno does.
Nothing is written to disk and there is no artifact to rebuild: the file a bird
runs is derived from the file the tests run, on every request. Transpiling is type
**stripping** only — no type checking.
