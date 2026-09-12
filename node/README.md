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
| `FLOCK_BIRDS` | 1 | how many bird slots the tail is split across |
| `FLOCK_BIRD_LAYERS` | 4 | how many layers the birds hold in total |

## Scripts

| command | what it does |
|---|---|
| `npm start` | run the coordinator (`PORT` to move off 8000) |
| `npm run solo` | one process holding every bird slot — no phone needed |
| `npm run bird` | one simulated bird, claiming the lowest free slot |
| `npm run health` | is the flock covered? exits non-zero if not |
| `npm run status` | full state: birds, transports, cache, recent device errors |
| `npm test` | syntax, GGUF range planning, both parsers cross-checked, the bird's load path |
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
| `src/gguf.mjs` | GGUF range planning over HTTP range requests (npm import) |
| `../kernels/` | the WGSL engine — the same kernels the test suite validates |
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
| `POST /join` | claim a layer slot, get its range, model dims and the GGUF url |
| `POST /chat` | SSE: `turn`, `hop`, `token`, `done`, `error` |
| `POST /reset` | drop every device's shard of the KV cache |
| `GET /status` | birds, transports, cached tokens, recent `/diag` reports |
| `GET /health` | 200 when the layers are covered, 503 when not |
| `POST /diag` | where birds report failures; phones have no readable console |
| `GET /ws` | websocket: hello, signaling, `ping`, `stats`, binary frames |

### Serving TypeScript to a browser

Browsers do not execute TypeScript, and adding a bundler would put back the build
step this design removes. So `/kernels/x.ts.js` transpiles `kernels/x.ts` on
request (`@deno/emit`, in memory, keyed by mtime) and rewrites relative `./y.ts`
specifiers to `./y.ts.js` so the browser walks the same module graph Deno does.
Nothing is written to disk and there is no artifact to rebuild: the file a bird
runs is derived from the file the tests run, on every request. Transpiling is type
**stripping** only — no type checking.
