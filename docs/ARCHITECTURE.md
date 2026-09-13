# flock architecture

Accurate to the code as of commit `c3db86b` ("Make flock a real project:
readiness handshake, sticky joins, real layout"). Session-durability work on
`server/`, `web/`, `tools/` and `test/` is in flight on another branch and will
change parts of the membership and identity story below; where that is likely,
the section says so in one line.

This is the document for an engineer who has read README.md and now has to
change something. It says what each process holds, what crosses each wire, and
which facts are load-bearing -- the ones where a plausible-looking change
produces text that is wrong without looking wrong.

## The one-paragraph version

One transformer, split by **layer** across a coordinator and any number of
browser tabs ("birds"). The coordinator (Deno, `server/`) holds the embedding,
layers `0..CUT-1`, the final norm and the tied vocabulary projection on its own
GPU. Each bird (a browser, `web/bird.html`) holds a contiguous run of the
remaining layers as WGSL compute shaders, with weights range-fetched straight
from the GGUF file on HuggingFace into GPU buffers. Per token, the hidden state
makes one **lap**: coordinator, then each bird in layer order, then back to the
coordinator, which projects it to a token id. Activations travel as f16 over
WebRTC data channels when they are up and over the websocket when they are not.
Every device keeps the K/V cache for **its own layers only**, which is why a
layer that moves between devices drops the conversation.

## Processes and what each one holds

```
  coordinator (one Deno process, server/server.js)          birds (one browser tab each, web/bird.html)
  ---------------------------------------------------       ------------------------------------------
  GPU: Model with cut                                       GPU: Layer[] for [start, end]
       token_embd (155.6 MB, also the LM head)                   weights streamed from the GGUF
       layers 0..CUT-1 with their K/V                            K/V cache for THESE layers only
       output_norm, argmax buffers
  tokenizer (transformers.js, HF chat template)             one websocket to /ws (signalling, ping,
  convo = {fed, turns, busy}    one conversation                 ready/nack/stats; frames if no channel)
  Flock: Bird[] in chain order (server/mesh.js)             rtc:    inbound PeerConnection FROM the
    per Bird: peerId, label, start/end, confirmed,                  coordinator (or previous bird)
              caps, Rate (EWMA bytes/ms), ws, pc, chan,     outRtc: outbound PeerConnection TO the
              lastSeen, failures, resetPending                      next bird, if there is one
  express: /join /leave /evict /diag /reset /chat           meta = the /join answer (range, gguf url,
           /status /health / /flock /check /js /kernels            hidden, kv_heads, head_dim, ...)
  ws: WebSocketServer at /ws
  node-datachannel: one PeerConnection per bird             chat page (web/chat.html)
  DIAG ring buffer (60 entries), TS_CACHE (transpiled       ----------------------------------
       kernels/*.ts per mtime), lastRebalance               polls GET /status every 2.5 s
  sweeper: setInterval 5 s                                  POST /chat, reads the SSE stream
```

Nothing is persisted anywhere. A coordinator restart forgets every peer id
(birds notice via `unknown peer -- rejoin` and reload); a bird reload is a new
member. The only cache on disk is `kernels/.cache/model.gguf` (the coordinator's
one-time 639 MB download) and, in each browser, the raw GGUF byte ranges the
streaming loader keeps in IndexedDB.

## The lap

`POST /chat {prompt, max_tokens, reset}` is one turn. The response is a
`text/event-stream`; each token is one lap.

```
  step 0 (prefill: seq = number of prompt tokens)      steps 1.. (decode: seq = 1)

  coordinator                  bird A          bird B          coordinator
  ----------                   ------          ------          -----------
  turnTokens(prompt) -> ids
  coord.forward(ids, offset)
    embedding + layers 0..CUT-1
    -> f32[seq * hidden]
  flock.lap(flat, seq, hidden, offset)
    pack() -> "FLK1" frame, f16  ---------->  layers a..b
                                 (K/V grows)  ---------->  layers c..d
                                                           (K/V grows)
                                                            -----------------> unpack()
                                                                               coord.project(flat, seq)
                                                                                 output_norm, tied head,
                                                                                 argmax -> next id
                                                                               sse {type:'hop', ...}
                                                                               sse {type:'token', text}
  offset += seq; stepIds = [next]; repeat
```

Where the frames go is decided per hop in `Flock.lap()`:

- **chained**: `coord -> A -> B -> coord`. A forwards straight to B over its
  own outbound data channel (`bird.forwardsDirectly`), and the coordinator awaits
  the **tail** of the longest directly-linked run, not each bird. One round trip
  however many birds are in the chain.
- **relayed**: `coord -> A -> coord -> B -> coord`. Fallback when A has no
  channel to B (no WebRTC in that browser, or the link is not up yet).

The K/V cache is filled as a side effect of every hop. That is what makes
`offset` (the absolute position of the first token in the frame) part of the
frame: every device's cache must be at the same position, and
`Model.encodePartial` throws if the coordinator's is not.

`Coordinator.turnTokens` appends only the new turn's tokens to a warm cache. It
does **not** re-encode the whole history through the chat template: with
`enable_thinking:false` the template scaffolds only the current turn, so a
re-encoded history diverges from the tokens that were actually fed (measured:
at token 9 of 13). This is why a conversation cannot be re-fed after a
reassignment; see "K/V is sharded by layer" below.

## The wire format (`web/js/wire.mjs`)

One file, imported by the coordinator directly and fetched by browsers at
`/js/wire.mjs`, so the two ends cannot drift.

```
  offset  size  field
  0       4     magic   u32 LE  0x314b4c46  ("FLK1")
  4       4     seq     u32 LE  tokens in this frame (prompt length for prefill, 1 for decode)
  8       4     hidden  u32 LE  floats per token (1024 for Qwen3-0.6B)
  12      4     offset  u32 LE  absolute position of the first token
  16      4     flags   u32 LE  bit 0 = reset: clear your K/V before this frame
  20      2*n   payload f16 LE, n = seq * hidden
```

f32 to f16 is **round-to-nearest-even**, not truncation. Truncating (`man >> 13`)
costs ~3 bits of an 11-bit mantissa: 5.8e-2 worst-case relative error against
4.9e-4. That bug shipped once, so every activation on the wire was 100x less
accurate than f16 allows; it is now a comment in `wire.mjs` and a line in
`kernels/README.md`.

The `reset` bit is how a conversation restart reaches the birds. `resetConvo()`
on the coordinator sets `resetPending` on every Bird; the next frame to each one
carries `reset = 1`, and the bird calls `Layer.reset()` on each of its layers.
There is no separate "clear" message, so a bird that receives no frame keeps a
stale cache until it does -- harmless, because the reset bit arrives with the
first frame that would have used it.

Everything else on the websocket is JSON, one object per message:

```
  bird -> coordinator                          coordinator -> bird
  --------------------                         --------------------
  {peer_id, label}            hello            {t:'chain', slot, start, end,
  {t:'ready', start, end}     I hold [s,e]        next_peer, next_range}   your range and successor
  {t:'nack', reason, detail,  I refuse this    {t:'signal', from?, data}   relayed SDP/ICE
     seq, offset, start, end}    frame         {error: 'unknown peer -- rejoin'}
  {t:'stats', ms}             my last frame
  {t:'caps', caps}            my limits changed
  {t:'forwards', direct}      I have/lost a direct link to my successor
  {t:'signal', to?, data}     SDP/ICE for the coordinator (no `to`) or a bird (`to`)
  {t:'ping'}                  heartbeat, every 4 s while joined
```

`reason` on a nack is `not-ready` (between assignments, or still streaming) or
`failed` (tried to compute and could not). The distinction matters: a failed
frame counts toward `bird.failures`, and three in a row removes the device; a
not-ready nack never does.

## HTTP surface

| route | who | body / answer |
|---|---|---|
| `POST /join` | bird | `{peer_id?, label, caps}` -> `{peer_id, slot, start, end, n_layers, hidden, kv_heads, head_dim, kv_cache, n_total, model, gguf, why, bytes, devices}`, or `{peer_id, wait: true, retry_ms, reason}` while a turn is running, or 409 `{error, detail, peer_id}` when this device cannot be placed |
| `POST /leave` | bird | `{peer_id}`; layers handed on immediately instead of after the liveness grace |
| `POST /evict` | anyone | `{peer_id}`; the coordinator-side version of leave |
| `POST /diag` | bird | `{stage, detail, ua, slot}`; a phone has no console, so failures travel here |
| `POST /reset` | chat | drops every cache; the one moment a speed rebalance may run |
| `POST /chat` | chat | `{prompt, max_tokens = 96, reset}` -> SSE: `turn`, `waiting`, `hop`, `token`, `done`, `error` |
| `GET /status` | chat, tests | everything: birds, allocation and its reasoning, pending, diag, convo |
| `GET /health` | scripts | 200 if a prompt could be served right now, else 503 |
| `GET /`, `/flock`, `/check` | browsers | chat.html, bird.html, inspect.html |
| `GET /js/*` | browsers | the shared modules (wire, gguf-stream, gguf-dir, probe) |
| `GET /kernels/*.ts.js` | birds | `kernels/*.ts` transpiled on request (type-stripping only, cached per mtime); relative `.ts` imports rewritten to `.ts.js` |
| `GET /kernels/*` | birds | the `.wgsl` sources, as-is |

A bird runs the **same** `kernels/layer.ts` the tests validate, served through
that transpile route, so there is no second implementation of a layer anywhere.

## The readiness handshake

Being **assigned** a range and **holding** it are different states, separated
by a weight stream of 3-18 s on a phone. Before this handshake the coordinator
assigned, announced and sent the next frame at once; the frame arrived at a
device holding nothing, the device treated that as fatal and left, the leave
reallocated everyone, and the next device was caught mid-stream. The log read
`joined -> left -> dropped -> joined` in a loop.

```
  bird                                  coordinator
  ----                                  -----------
  POST /join  ------------------------> claim(); rebalance(force) -> [start, end]
              <------------------------ {peer_id, start, end, gguf, ...}
  open /ws, hello {peer_id} ----------> bird.ws = ws; send chainFor(peer)   (ALWAYS: /join's range may be stale)
                                        openRTC(): offer a data channel
  stream weights for [start, end]       ... a turn that starts now WAITS on this bird ...
  Layer.fromBuffers() x n
  {t:'ready', start, end} ------------> bird.confirm(start, end)
                                        isReady() = confirmed matches assigned
                                        flock.ready() = covered && pending().length === 0
```

`Flock.awaitReady()` is the gate every turn passes: it polls every 200 ms for up
to `FLOCK_READY_WAIT_MS` (90 s), emitting an SSE `waiting` event about once a
second naming the birds still pending, and gives up with a reason that names
them. A bird that receives a frame it cannot serve answers `nack` rather than
leaving, which ends that turn at once instead of at the 120 s frame timeout, and
the device stays a member.

A confirmation for a range the bird is no longer assigned is kept but does not
count: it is true, just stale, and counts again if the assignment returns.

## Identity is the coordinator's to assign

`/join` mints an 8-byte random hex `peer_id`. The bird sends nothing persistent:
there is no localStorage outside the weights cache. Two reasons, both learned
on real devices:

- a stale id kept in the browser let one device silently take over another's
  slot;
- two devices reporting `navigator.platform === "MacIntel"` (a Mac and an iPad)
  were indistinguishable while knocking each other out of the flock.

The one exception is a **retry**: a device that joins while a turn is running is
answered `wait` with the id just minted, and must send it back. Minting a fresh
id per retry admitted one phantom member per retry (measured: 27 members from
one phone waiting out one 30-token turn). An id is reused only if that member
has **never** connected (`everLinked === false`); a stale id from anywhere else
is refused.

*In flight: the durability work will let a bird survive a coordinator restart
or a page reload without becoming a new member; the "every /join is a new
member" rule above is what that changes.*

## Sticky allocation: a join is a local edit

`server/allocate.js` has two entry points and the difference is the whole
membership story.

- `allocate(layers, devices)` is the **exact** split: dynamic programming over
  contiguous runs minimising the **makespan** (the slowest device's per-token
  time), in bytes, under each device's `bind` (largest tensor it can bind) and
  `budget` (total bytes). Its optimum for N+1 devices generally shares no
  boundary with its optimum for N, so using it on every join moved **every**
  bird, and every moved bird re-streams and holds nothing meanwhile.
- `adjust(layers, current, devices)` is what a **membership change** uses: a
  newcomer takes a prefix or suffix off **one** incumbent (donor and size chosen
  by makespan, ties broken so the donor and newcomer balance against each
  other); a leaver's run is split between its two neighbours; nobody else
  moves. It falls back to the exact search only when no local edit fits the
  limits, and says so (`sticky: false`).

`Flock.plan({force})` chooses: `force` (membership changed) means `adjust` with
no gain gate; not forced means the exact split under `server/speed.js`'s
brakes -- a rate per device rather than a time (so halving a device's layers
does not read as new evidence), an EWMA, three samples before a rate is trusted,
a 15% predicted gain and a 20 s cooldown. The speed path runs **only** at
`/reset` and at a `/chat` with `reset: true`, i.e. when there is no context to
lose.

A membership change that arrives **mid-turn** is deferred (`flock.defer()`) and
applied in `/chat`'s `finally`, after the answer completes. `test/e2e/churn.test.mjs`
asserts exactly this from the birds' side: it reads what each simulated bird
prints it holds, not what `/status` claims, because the cascade was invisible to
every `/status` check.

## Liveness, and the sweeper

- Each bird sends `{t:'ping'}` every 4 s while joined; any message updates
  `lastSeen`.
- `Bird.alive(grace = 40 s)`: has a link (socket or open channel) and was heard
  from inside the grace.
- `Bird.claimed(hold = 15 s)`: alive, **or** admitted over HTTP and never yet
  connected (the window between `/join` and the websocket opening).
- `Flock.members()` is the union the allocator plans over; `Flock.sweep()` is
  its exact complement, run every `SWEEP_MS` (5 s) from `server.js`, and it
  re-plans on its own so a removed device's layers are not left assigned to
  nobody. Announcing (and dropping the conversation) stays the caller's job.
- A websocket close does **not** remove a member: a refresh and a departure look
  identical at that moment, and the refreshed page comes back with the same id
  within seconds. Only the sweep, `/leave`, `/evict` or three failed frames end
  membership.

## K/V is sharded by layer, and everything that follows from it

Each `Layer` owns the keys and values for its layer, on the GPU of the device
that holds it. There is no `past` tensor on any wire; the frame is the hidden
state only. Consequences, each of which is a design decision elsewhere:

1. **Any reassignment drops the conversation.** A layer that moves leaves its
   keys on the device that no longer holds it, and the new holder starts from an
   empty cache at a position the rest of the flock believes is filled.
   `rebalance()` therefore calls `resetConvo()` whenever `moved.length > 0`, and
   `/status.last_rebalance` says so, so the chat page can explain a context that
   went to zero.
2. **Re-feeding the history is not available** (the chat-template divergence
   above), so the drop is real and is said out loud rather than papered over.
3. **Nothing moves during a turn.** A join mid-turn is deferred to the end of
   the answer; the speed rebalancer waits for a conversation to start.
4. **One conversation at a time.** `convo.busy` refuses a second `/chat`;
   interleaved turns would write two conversations into the same caches.
5. **A torn token ends the turn.** A nack or a dropped device mid-lap leaves the
   devices before it one position ahead of the ones after it, so `/chat` resets
   rather than continuing.
6. **`offset` is checked, not trusted.** `Model.encodePartial(ids, offset)`
   throws if the coordinator's cache is not at that position.
7. **Decode attention caps at 2048 keys** (`attention.wgsl`'s non-streaming
   softmax); that is the context ceiling for the whole flock.

## Why Deno

The coordinator holds 4 layers, the embedding and the LM head on a GPU. Node has
no WebGPU (no `navigator`, no flag; checked, not assumed). Deno 2.1 provides a
real adapter through wgpu and runs express, ws, node-datachannel and
transformers.js unchanged through its Node compatibility layer, so the whole
server is one process on one runtime. The alternatives -- a browser tab as
coordinator (what swarmllm does), or a CPU coordinator that would be a second
implementation of 24 layers nothing validates -- were rejected on those grounds.

The dependencies are still npm packages (`nodeModulesDir: "manual"` in
`deno.json`), the tests are Node where they can be and Deno where they need a
GPU, and `onnxruntime-node` -- a native addon that will not load under Deno --
runs in a Node subprocess (`kernels/onnx_full.mjs`) when the ONNX reference is
present.

## Why https

Chrome and Edge expose WebGPU only in a **secure context**. A bird page at
`http://192.168.1.20:8000/flock` sees no `navigator.gpu` at all, which is
indistinguishable from a device without a GPU; Safari does not gate it this way,
which is why iPhones worked before anyone noticed. `npm run cert` writes a
self-signed certificate naming the LAN address into `.certs/`, the coordinator
serves https whenever that directory exists, and the bird page opens `wss://`
on an https page because a browser blocks an insecure `ws://` from a secure
context (that silent block is how every bird once kept the range `/join` gave
it and never heard about a reassignment). `FLOCK_NO_TLS=1` forces plain http;
the test runner uses it on loopback. What a real deployment needs instead is in
`docs/PRODUCTION_AUDIT.md`.

## The WebRTC signalling relay

The coordinator is the only introducer, in the role PeerJS plays for swarmllm.
A `{t:'signal'}` with no `to` is for the coordinator's own PeerConnection to
that bird; with `to` it is relayed verbatim to that peer as
`{t:'signal', from: <sender>, data}`. Bird-to-bird links are opened by the
**upstream** bird when it learns its `next_peer` from a chain message; the
downstream bird answers. A bird therefore holds up to two PeerConnections:
`rtc` (inbound, from the coordinator or the previous bird) and `outRtc`
(outbound, to the next bird). ICE uses a public STUN server; on one LAN the host
candidates are what connect.

## Measurement traps, summarised

`kernels/README.md` is the reference; these are the ones that will bite anyone
benchmarking or validating a change, with the section to read.

| trap | one line | read |
|---|---|---|
| `onSubmittedWorkDone()` does not wait | on Deno's wgpu it returns before the GPU finishes; only `mapAsync` on a buffer the pass wrote is a barrier. Produced a 4400 GFLOP/s matvec | [The measurement traps, 1](../kernels/README.md#the-measurement-traps) |
| dispatch floor | an empty kernel costs ~18 us, the same as a 3072x1024 matvec; real numbers need thousands of dispatches behind one fence | [2](../kernels/README.md#the-measurement-traps) |
| f32 vs f64 references | compare against an FMA-modelled strict-f32 reference; then agreement is exact and the tolerance argument disappears | [3](../kernels/README.md#the-measurement-traps) |
| relative error under cancellation | judge signed sums by max absolute error over the data's scale (`absErrScaled`) | [4](../kernels/README.md#the-measurement-traps) |
| the readback is the measurement | ~26 ms per map readback regardless of size; splitting a token by differencing two readbacks cancels to noise | [5](../kernels/README.md#the-measurement-traps) |
| signal below noise | RoPE at positions 0-5 with base 1e6 is one ULP; assert at pos 900 | [6](../kernels/README.md#the-measurement-traps) |
| thousands of passes wedge the backend | dispatches batch, passes do not; put repetitions inside one pass | [7](../kernels/README.md#the-measurement-traps) |
| an error constant in N is not a sharding error | it is in the input | [8](../kernels/README.md#the-measurement-traps) |

And the correctness traps that produce a **working model that is wrong**:
RoPE pairing follows the weights, not the file format (this GGUF needs NEOX);
`queue.writeBuffer` ignores a TypedArray view's offset on Deno 2.1.2; an
oversized binding is a validation error that writes zeros, not an exception;
`output_norm` is applied by the last shard, not the head. See
[The correctness traps](../kernels/README.md#the-correctness-traps).

## Where to look

```
server/server.js      the HTTP and websocket surface, /chat's turn loop, the sweeper
server/mesh.js        Bird and Flock: liveness, readiness, lap(), the signalling relay
server/allocate.js    allocate() exact split, adjust() sticky edit, Infeasible
server/speed.js       Rate (EWMA bytes/ms) and shouldRebalance()'s four brakes
server/coordinator.js Model with a cut, tokenizer, turnTokens()
web/js/wire.mjs       the frame format
web/js/gguf-stream.mjs range-fetch GGUF tensors into GPU buffers, 4 MB at a time
web/js/probe.mjs      the capability probe both /check and /join use
web/bird.html         a bird: probe, join, stream, build, serve frames, reassign
kernels/              the engine; kernels/README.md for its validation and traps
test/e2e/churn.test.mjs  the membership protocol, asserted from the birds' side
```
