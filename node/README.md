# flock (Node runtime)

The all-JavaScript flock: the coordinator runs on Node, so it can be a real
WebRTC peer. Every link in the chain is a data channel; the websocket carries
only signaling.

```bash
cd node && npm install
npm start
```

Requires the exports first (from the repo root, or `npm run build` from here):

```bash
python3 build_shards.py --start 24 --end 27 --birds 2
python3 flock_export_coordinator.py --cut 24
```

Python remains only as the **export and verification tool** — PyTorch is the
reference every ONNX graph is checked against, so a bad export can't be
silently wrong.

## Scripts

| command | what it does |
|---|---|
| `npm start` | run the coordinator (`PORT` to move off 8000) |
| `npm run solo` | one process holding every bird slot — no phone needed |
| `npm run bird` | one simulated bird, claiming the lowest free slot |
| `npm run health` | is the flock covered? exits non-zero if not |
| `npm run status` | full state: birds, transports, cache, recent device errors |
| `npm test` | syntax, GGUF range planning, both parsers cross-checked, the bird's GGUF gate |
| `npm run test:ui` | drives chat.html and bird.html against a live coordinator |
| `npm run test:gpu` | streams real weights into real GPU buffers (needs Deno) |
| `npm run plan` | what each device would fetch, reading only a GGUF header |
| `npm run build` | export the shards and the coordinator graphs |

## Layout

| file | role |
|---|---|
| `src/server.js` | HTTP, SSE chat loop, websocket signaling |
| `src/coordinator.js` | embedding, layers 0..cut-1, vocab projection, tokenizer |
| `src/mesh.js` | the flock: slot claiming, liveness, chain topology, WebRTC links |
| `src/gguf.mjs` | GGUF range planning over HTTP range requests (Node-only: npm import) |
| `../web/js/gguf-dir.mjs` | the same directory parse, browser-safe |
| `../web/js/gguf-stream.mjs` | streams GGUF tensors from HuggingFace into GPU buffers |
| `sim_bird.mjs` | a bird without a browser |
| `check_html.mjs` | `node --check` over the inline `<script>` of each page |

## Endpoints

| route | purpose |
|---|---|
| `GET /` | the chat page |
| `GET /flock` | the bird page |
| `POST /join` | claim a layer slot, get its range and model dims |
| `POST /chat` | SSE: `turn`, `hop`, `token`, `done`, `error` |
| `POST /reset` | drop every device's shard of the KV cache |
| `GET /status` | birds, transports, cached tokens, recent `/diag` reports |
| `GET /health` | 200 when the layers are covered, 503 when not |
| `POST /diag` | where birds report failures; phones have no readable console |
| `GET /ws` | websocket: hello, signaling, `ping`, `stats`, binary frames |
