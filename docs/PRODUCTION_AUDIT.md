# Production-readiness audit

Read against commit `c3db86b`: `server/server.js`, `server/mesh.js`,
`server/allocate.js`, `server/speed.js`, `web/bird.html`, `web/chat.html`,
`web/js/wire.mjs`, `tools/sim_bird.mjs`, `tools/cert.sh`. Line numbers refer to
that commit. Nothing here is fixed; this is the list for the next agent, ordered
by severity within each section, and it is deliberately short. Every finding
names the file and line, says what goes wrong, and proposes one fix.

The README's own framing -- "LAN only. No authentication and almost no input
validation. Do not expose it to the internet." -- is accurate today. The
findings below are what separates that from something a team could run.

Severity: **High** = crashes the coordinator, lets one participant read or
corrupt another's work, or produces wrong text without an error. **Medium** =
degrades service, leaks state or fails badly under a plausible misuse.
**Low** = hygiene.

## 1. Crashes and correctness

### 1.1 High: any malformed websocket message kills the coordinator

`server/server.js:768` -- `const m = JSON.parse(data.toString());` runs inside
the `ws.on('message')` handler with no `try`. A text frame that is not JSON
throws, the exception escapes the event handler, and under Deno an uncaught
exception terminates the process: every bird loses its coordinator over one bad
frame from one client. The same handler reaches into unvalidated shapes:
`server/mesh.js:238` `this.pc.setRemoteDescription(d.sdp.sdp, d.sdp.type)`
throws `TypeError` on `{t:'signal', data:{kind:'offer'}}`, and
`server/server.js:806` relays `m.data` whatever it is. `/chat` is the only
place with a catch-all (`server.js:716`).

**Fix.** Wrap the whole message handler: parse in a `try`, validate `m` against
a small per-`t` schema (a function per message type that returns the typed
message or `null`), close the socket with code 1008 on the second bad message,
and log the peer. Add `process.on('uncaughtException')` /
`'unhandledRejection'` handlers that log with the stack and exit non-zero, so a
future crash is at least attributable.

### 1.2 High: the liveness sweeper re-plans in the middle of a turn

`server/server.js:115-123` runs `flock.sweep()` every 5 s and only afterwards
checks `convo.busy`. But `Flock.sweep()` (`server/mesh.js:382-397`) already
called `this.plan({force: true})` at line 395, which rewrote every moved bird's
`start`/`end`/`slot` and reset its `Rate`. The server's comment at lines
119-120 ("reassigning now would move layers whose K/V cache the current token
is still writing into") describes what the code was meant to avoid and what it
now does; only the `announce()` is deferred.

The consequence is in `Flock.lap()` (`mesh.js:609-645`): it walks
`this.chain()` -- the **new** assignment -- and sends frames to birds that
still hold and compute their **old** layers, because a bird only re-streams
when it is told (`web/bird.html:834-839`) and readiness (`isReady()`) is
checked once, at the start of the turn (`server.js:609`), never per lap. With
A-B-C in the chain and B swept, B's layers go to A or C on paper; the next lap
skips B's layers entirely and the remaining tokens of the answer are computed
without them. That is the "wrong layers at the right positions, plausible
garbage" failure the rest of the code is built to prevent, and it is silent.

**Fix.** Give `sweep()` a `{plan}` option and pass `plan: !convo.busy`,
deferring the re-plan like every other mid-turn membership change; and,
independently, have `lap()` refuse to send to a bird whose `isReady()` is false
(throw the same nack-style error the turn already handles). The second half is
the real guard: it turns any future variant of this race into an error rather
than wrong text. `test/e2e/churn.test.mjs` is the place to add the scenario
(kill a sim mid-turn, assert the turn ends with an error, not with text).

### 1.3 High: a peer id is the only credential, and `/status` publishes it

`server/server.js:769-772` -- a websocket hello `{peer_id}` binds that socket
to the member with that id: `bird.ws = ws`. Nothing else is checked. Every
member's `peer_id` is in `GET /status` (`server.js:465`, `468` via
`Bird.info()` at `mesh.js:144`) and `/status` is unauthenticated. So anyone on
the network can hijack a member: the coordinator sends that member's frames --
the conversation's hidden states -- to the impostor, and accepts whatever it
sends back as the answer. The same id works on `POST /leave` and `POST /evict`
(`server.js:394-427`).

**Fix.** Mint two values at `/join`: the public `peer_id` (fine to show) and a
secret `token` returned only in the join response. Require the token in the
hello and on `/leave`; keep `/evict` for the operator (section 2). Store a hash
of the token, not the token. This is the smallest change that makes identity
mean something, and the in-flight durability work -- which wants a bird to
resume across a reload -- needs exactly such a secret anyway.

### 1.4 High: the signalling relay lets any member re-wire any other

`server/server.js:804-807` relays `{t:'signal', to, data}` from any member to
any `to`, verbatim, with `from` set. `web/bird.html:760-777` then handles an
`offer` from **anyone** by creating a fresh `RTCPeerConnection` in `rtc` --
the slot that holds the link **from the coordinator or the previous bird** --
and its `ondatachannel` replaces `inChan`, the channel frames arrive on. A
member can therefore make another bird take its frames from it instead of from
the coordinator; `answer` at line 780 is applied to `outRtc || rtc` without
checking who it is from, so a stray answer can also poison a live link. The
coordinator's own PeerConnection has the same trust: `mesh.js:235-242` applies
whatever a bird sends.

**Fix.** On the coordinator, relay a signal only to the sender's chain
neighbours (`chainFor()` already knows the successor; add the predecessor) and
validate `data.kind` is one of `offer|answer|ice` with the fields each needs.
In `bird.html`, carry `prev_peer` in the chain message and accept an `offer`
only from it (or from the coordinator, which has no `from`); accept an `answer`
only on `outRtc` and only from `next_peer`.

### 1.5 Medium: a returned frame is not matched to the frame that was sent

`server/mesh.js:188-195` -- `deliver()` unpacks any binary message and hands
it to whatever `_waiter` is pending; `meta` (seq, hidden, offset) is decoded
and ignored. A frame that arrives late -- from a lap that already timed out at
120 s and reset the conversation, or from a bird that was slow while its
neighbour was retried -- is taken as the answer to the **current** lap. The
byte length is not checked either: a frame with the wrong number of floats
reaches `coord.project(flat, seq)` (`server.js:671`).

**Fix.** Check `meta.seq === seq`, `meta.hidden === hidden`,
`meta.offset === offset` and `data.length === seq * hidden` in `deliver()`
before resolving; log and drop otherwise. A per-lap nonce in the header would
be stronger (`flags` has 31 spare bits) and is a compatible change since both
ends share `wire.mjs`.

### 1.6 Medium: no pre-validation of the turn's size; oversized input costs the context

`server/server.js:556` -- `max_tokens` and `prompt` are taken as-is. The engine
does throw rather than compute past its buffers (`kernels/model.ts:447` for a
lap over `maxPrefill` 512, `kernels/model.ts:240` for a context over `maxKeys`
2048), so the failure is loud, but it lands **inside** the turn: `/chat`'s
catch at `server.js:716-720` calls `resetConvo()`, so a prompt of 600 tokens or
a conversation that crosses 2048 positions ends by erasing the whole
conversation, with an error message that names a kernel constant. And
`max_tokens` is unbounded: one `POST /chat {max_tokens: 1e9}` holds the lock
(`convo.busy`) for up to 2048 laps, roughly a minute or two.

**Fix.** Before setting `convo.busy`: reject `max_tokens` outside `1..512`
(configurable), reject a prompt over a byte cap (8 KB) or over `maxPrefill`
tokens after tokenising, and reject a turn where `convo.fed + promptTokens +
max_tokens > maxKeys` with a message that says the context is full and to
clear it. Expose `maxKeys` and `maxPrefill` in `/status` so the chat page can
show a context meter.

### 1.7 Medium: a bird can shape the split and drop everyone's context at will

Every input to the allocator is self-reported and unbounded: `caps` at `/join`
(`server.js:305`) and via `{t:'caps'}` (`server.js:843-846`), and `stats.ms`
(`server.js:816`). Reporting `ms: 0.1` makes a device the fastest in the flock
and hands it the biggest share; a `{t:'caps'}` message triggers
`rebalance({force: true})` immediately, and any move calls `resetConvo()`
(`server.js:147`). A misbehaving or buggy member can therefore reset the
conversation every few seconds.

**Fix.** Rate-limit `caps` updates per member (one per 30 s, and none while a
conversation has context), ignore `stats.ms` outside `[MIN_MS, 60000]`, and
consider not re-planning on a caps update at all unless the new limits make
the current assignment infeasible.

## 2. Authentication and authorisation

There is none. Every route in `server.js` and the websocket are open to
anyone who can reach the port, which `server.js:892` binds on `0.0.0.0`.

- `POST /evict`, `POST /leave` (`server.js:394-427`): anyone can remove any
  member by id (ids in `/status`).
- `POST /reset` (`server.js:519`): anyone can erase the conversation.
- `POST /chat` (`server.js:555`): anyone can use the flock.
- `GET /status` (`server.js:459`): exposes peer ids, labels, user-agent-derived
  device names and the last 12 diagnostics from every phone.
- **Cross-site.** Neither the websocket upgrade (`server.js:762`, no
  `verifyClient`) nor any route checks `Origin`. A web page anywhere can open
  `wss://192.168.1.20:8000/ws` from a visitor's browser, and can `POST /reset`
  or `POST /join` with a `text/plain` body (a "simple" request that needs no
  CORS preflight; `express.json()` leaves `req.body` undefined and `/join`
  still mints a member at `server.js:294`). Drive-by resets and phantom joins
  from the internet are possible today via any browser on the LAN.

**Fix**, in order of effort: (1) check `Origin` on the upgrade and on every
`POST`, allowing only the coordinator's own origins (its LAN addresses and
`localhost` on the configured port); (2) an operator token from
`FLOCK_ADMIN_TOKEN` required on `/evict`, `/reset` and, optionally, `/chat`;
(3) the join token from finding 1.3 for everything a bird does. A CORS
middleware is not the fix -- the routes are not meant to be called
cross-origin at all.

## 3. Rate limiting and denial of service

- **Unbounded joins** (`server.js:272-390`). Each `POST /join` with no
  `peer_id` mints a member; `Flock.birds` grows until the 15 s never-linked
  window (`mesh.js:95-99`) lets the sweeper drop them, and every join runs a
  re-plan and an `announce()` to every real bird. A loop of joins from one
  host keeps the flock re-planning and every real bird receiving chain
  messages. Fix: a per-source-address limit on `/join` (e.g. 5 per minute), a
  hard cap on members (`FLOCK_MAX_DEVICES`, refuse with 503 above it), and a
  cap on never-linked members per address (2).
- **`/diag` log flooding** (`server.js:433-444`). Unauthenticated, one
  `console.log` per call, `detail` capped at 400 chars but `stage` and `ua`
  not. The ring buffer is bounded (60), the log is not. Fix: cap `stage`/`ua`
  lengths and rate-limit per address.
- **Websocket payload** (`server.js:762`). `WebSocketServer` is created with
  the default `maxPayload` of 100 MiB; `Bird.deliver()` (`mesh.js:192`)
  allocates a `Float32Array` for whatever arrives. A legitimate frame is at
  most `20 + maxPrefill * hidden * 2` = 1 MB. Fix: `maxPayload: 2 * 1048596`
  and the length check from 1.5.
- **The 120 s frame timeout** (`mesh.js:168`) is the only bound on a lap. A
  websocket close resolves the waiter early (`teardown()`, `mesh.js:206`), but
  a data-channel close does not (`mesh.js:232` only nulls `chan`), so a bird
  whose WebRTC link drops while its socket stays up holds the turn for the
  full two minutes. Fix: resolve the waiter with `null` in `onClosed` too, and
  make the timeout proportional (`FLOCK_LAP_TIMEOUT_MS`, default 30 s).
- **SSE backpressure** (`server.js:559`). `res.write` return values are
  ignored; a chat client that stops reading buffers every event in the
  coordinator's memory until the turn ends. Small per turn (hundreds of
  events), but it is the pattern to fix when the journal lands: check the
  return value and stop the turn when the client cannot keep up.
- **The journal that does not exist yet.** The durability work in flight will
  add persistent state. Whatever it writes must be bounded (size or count,
  with rotation) and must not be written on the request path of `/join` or
  `/diag`, or the joins above become a disk-filling attack.

## 4. Graceful shutdown

There is none. `server/server.js` installs no `SIGTERM`/`SIGINT` handler; the
only exit paths are `process.exit(1)` on a listen error (`server.js:886-889`)
and the runtime's default. On a kill:

- an in-flight `/chat` is cut mid-stream with no `error` event, so the chat
  page shows a cursor until its fetch fails;
- every bird's socket closes without a close code; `bird.html:848-855`
  reconnects every 1.5 s indefinitely, and when the coordinator comes back
  each gets `unknown peer -- rejoin` and reloads (`bird.html:860-870`), which
  re-streams its weights from the CDN;
- the GPU device is never destroyed (harmless today; the OS reclaims it);
- nothing is flushed, because nothing is persisted -- which changes with the
  durability work, and then an unflushed journal is lost state.

**Fix.** One `shutdown(signal)` that: sets a `draining` flag so `/join` and
`/chat` answer 503 with `Retry-After`; if a turn is running, waits up to
`FLOCK_DRAIN_MS` (10 s) for it to finish, else sends `{type:'error',
text:'coordinator shutting down'}` and ends the stream; sends every bird
`{t:'shutdown', retry_ms}` and closes its socket with code 1001 (so the bird
page can say "coordinator restarting" and back off instead of hammering);
closes the WebSocketServer and the HTTP server; flushes whatever the
durability work persists; `dev.destroy()`; exits 0. Install it for `SIGTERM`
and `SIGINT`, and make a second signal exit immediately.

## 5. Logging

Everything is `console.log`/`console.error` with hand-written prefixes
(`[join]`, `[ready]`, `[nack]`, `[diag ...]`, `[chat]`, `[kernels]`,
`[flock]`) and no timestamps, so a log cannot be correlated with a phone's
`/diag` entries (which do carry `at`) or with a turn. Some lines are only
useful to a human at the terminal (`server.js:97-102`, the split explanation),
and those are the product's interface today, so the fix is not to remove them.

**Fix.** A 30-line `server/log.js`: `log.info(event, fields)`,
`log.warn`, `log.error`, each producing one line with an ISO timestamp, the
event name and `key=value` fields (human default) or one JSON object per line
when `FLOCK_LOG=json`. Give every turn a `turn` id and every member its
`peer_id` in the fields, and route the existing prefixes through it. Keep the
startup banner as it is.

## 6. Configuration

Every variable is read inline with `+(process.env.X || default)` and no
validation (`server.js:73-76`, `553`, `876`; `tools/sim_bird.mjs:62`).

- `FLOCK_COORD_LAYERS=abc` gives `Math.max(1, NaN) = NaN`, so `CUT` is `NaN`,
  `layerPlan()` returns an empty list, and the first `/status` throws inside
  `contiguous()` on `this.layers[0]`.
- `FLOCK_READY_WAIT_MS=abc` gives `NaN`, and `awaitReady`'s `waited > NaN`
  is never true: a turn waits forever for a bird that will never confirm.
- `PORT=abc` passes `NaN` to `listen()`.
- `FLOCK_GGUF` accepts any URL. `server/gguf.mjs:48` opens it with
  `allowLocalFile: true`, so a `file:` path reads the coordinator's disk --
  intended for local models, but the same URL is then handed to every bird
  (`server.js:382`, `388`), which cannot fetch it. Also unstated: the value is
  trusted to be the model the kernels are configured for (`QWEN3_06B` is
  hard-coded at `server.js:37`, `85-87`); a different architecture loads and
  computes garbage.
- The README table (`README.md`, "Other things you can set") is the only
  documentation and omits `FLOCK_URL` (tools and tests), and the test-only
  `TEST_VERBOSE`, `CHURN_RESTREAM_MS`, `CHURN_VERBOSE`, `FLOCK_E2E_FORCE`.
- Not configurable but should be: the bind address (`server.js:892`, always
  `0.0.0.0`), the sweep interval and grace (`server.js:114`, `mesh.js:73`,
  `295`, `382`), the lap timeout (`mesh.js:168`), the cert paths
  (`server.js:752-753`).

**Fix.** A `server/config.js` that reads every variable once, with a parser
per type (`int(name, default, {min, max})`, `url(...)`, `bool(...)`), throws a
one-line error naming the variable and the accepted range on bad input, and
prints the effective configuration at startup. Refuse `FLOCK_NO_TLS` unless
the bind address is loopback, and refuse a `file:` `FLOCK_GGUF` unless a
`FLOCK_GGUF_PUBLIC` URL is given for the birds. Verify at startup that the
GGUF's `general.architecture` and dimensions match the kernel config.

## 7. Error paths that swallow errors

- `mesh.js:595` (`announce()`) and `server.js:793` (the hello) wrap
  `ws.send()` in `try {} catch {}`. A bird whose send fails never learns its
  range -- the exact class of bug the readiness handshake was built around --
  and nothing records it. Fix: log the peer and the error, and mark the bird
  `needsAnnounce` so the next ping re-sends the chain message.
- `mesh.js:240` `addRemoteCandidate` failures are dropped. Fine for the
  occasional stale candidate; log at debug so an ICE failure is diagnosable.
- `bird.html:816` `JSON.parse(ev.data)` in `ws.onmessage` throws out of the
  handler on a bad message; the page keeps running, but a phone has no console,
  so the event is lost. Wrap it and `report('bad-message', ...)`.
- `server.js:239` -- the first request for any `/kernels/*.ts.js` does
  `await import('jsr:@deno/emit')`, which fetches from jsr.io at **runtime**.
  A coordinator started with internet and then moved to an offline LAN serves
  404 `cannot serve layer.ts` (`server.js:257-259`) to every bird, and the
  message does not say why. Fix: import `@deno/emit` at startup (so a missing
  network fails the start, loudly), or pre-transpile once at startup into
  `TS_CACHE`.
- `server.js:555-556` -- a `POST /chat` without a JSON body leaves `req.body`
  undefined under Express 5, so the destructuring throws before the SSE
  headers are set and the client gets an HTML 500. Return a 400 JSON error
  first.

## 8. Memory over a long-running coordinator

- **Coordinator K/V** is fixed-size (`maxKeys` 2048 per layer) and reused
  across conversations; no growth. `convo.fed` is the only counter and
  `/reset` zeroes it.
- **Members** are bounded by the sweeper (section 3 for the join flood).
  `Bird.teardown()` closes `pc` and `chan` on socket close and on `release()`,
  and `openRTC()` closes the previous pair (`mesh.js:214-216`), so
  PeerConnections do not accumulate. `node-datachannel` objects are native;
  if the process ever leaks, this is the first thing to count.
- **Bounded ring buffers**: `DIAG` (60, `server.js:442`), `TS_CACHE` (one
  entry per `.ts` file, `server.js:224`).
- **Birds are the growth risk.** `bird.html:582-586` says it: `L.destroy?.()`
  is a no-op because `kernels/layer.ts` has no `destroy()`, so a reassigned
  phone holds its old 67 MB of GPU buffers until the JS garbage collector
  gets to the `Layer` objects, and two or three reassignments in a row can
  exceed what a phone will give a tab. Fix: add `Layer.destroy()` that calls
  `destroy()` on every `GPUBuffer` it owns (weights, K/V, scratch) and clears
  the bind-group cache; it is a `kernels/` change with a natural test in
  `kernels/test_frombuffers.ts` (destroy, then assert `device.queue` accepts
  no further encode from that layer).
- **The chat page** keeps every turn's per-token `hops` in the DOM
  (`chat.html:417`, `447`); a session of hundreds of turns is a large page.
  Cap the number of stats tables kept, or drop the table when the message
  scrolls out of the last N.

## 9. TLS

Today: a self-signed certificate from `tools/cert.sh` (RSA 2048, 825 days, SAN
= one LAN IP + 127.0.0.1 + localhost), served whenever `.certs/` exists, and
every device accepts the warning once. `tools/sim_bird.mjs:66` disables
verification for that client. This is right for a LAN demo and wrong for
anything else, for three reasons: users are trained to click through
warnings; the certificate pins one IP and silently breaks when DHCP moves the
machine; and nothing verifies the coordinator a bird connects to, which is
what makes findings 1.3 and 1.4 reachable by a network attacker rather than
only by a member.

**A real deployment needs**: a DNS name for the coordinator and a certificate
for it -- either public (Let's Encrypt with DNS-01, which works for a name that
resolves to a private address) or from a private CA whose root is installed on
the devices (MDM, or a one-time profile install); `FLOCK_CERT`/`FLOCK_KEY`
paths in the configuration rather than the fixed `.certs/`; refusal to start
with `FLOCK_NO_TLS` on a non-loopback bind; a `Strict-Transport-Security`
header once the name is real; and the cert.sh script kept only as the
development path, saying so in its banner. WebRTC media is DTLS-encrypted
regardless; the signalling that sets it up is only as private as the websocket
carrying it.

## 10. Trust between server and pages

- **The bird trusts the coordinator completely.** It executes
  `/kernels/layer.ts.js` from it (`bird.html:120`), streams weights from
  whatever URL `/join` returns (`bird.html:386`, `641`; `meta.gguf`), and
  applies any chain range it is sent (`bird.html:834-839`). This is inherent
  to the design -- the coordinator is the operator's machine -- and becomes a
  problem only without TLS or with finding 1.4, where a member can act as the
  upstream. Show the GGUF host in the bird page's facts grid so a wrong URL is
  visible, and validate `start <= end < n_total` before streaming.
- **The chat page** builds HTML with `innerHTML`. `esc()` covers most
  server strings, but `chat.html:270-271` inserts `s.model` and `s.wire`, and
  line 278 inserts `s.missing.join(', ')`, unescaped. `s.model` is
  `header.metadata['general.name']` from the GGUF (`server.js:83`), so a
  crafted model file puts markup into the operator's chat page. Operator-
  controlled input, so Low; the fix is `esc()` on those three or `textContent`.
- **The server trusts the pages' self-description**: `label` (`server.js:305`,
  `772`) is any type and length and ends up in every log line and `/status`;
  `ua` in `/diag` likewise. Cap both at 64 characters, coerce to string, strip
  control characters. `caps` is normalised (`mesh.js:342-358`) except
  `vendor`/`gpu`, which are stored raw (not exposed by `info()`, so Low).
- **`onCoordinator` is inferred from the source address** (`server.js:303-304`).
  Behind any reverse proxy every bird looks local and is charged the
  coordinator's `reservedMs`, which skews the split against all of them. If a
  proxy is ever used, read `X-Forwarded-For` only from a configured trusted
  proxy, or replace the inference with an explicit `on_coordinator` flag the
  bird page sets when `location.hostname` is the coordinator's own machine.

## 11. Smaller items

- `server.js:262` serves the whole `kernels/` directory statically, including
  test sources and the README; `serve-static`'s default `dotfiles: 'ignore'`
  should keep `kernels/.cache/model.gguf` (639 MB) and `.ref/` out of reach --
  verify with a request for `/kernels/.cache/model.gguf` and add an explicit
  `dotfiles: 'deny'` either way, or serve only `*.wgsl`.
- `server.js:876-890` prints the fix for `EADDRINUSE`; do the same for
  `EACCES` (port < 1024) and for a missing/unreadable cert (`readFileSync` at
  line 760 throws a raw ENOENT today).
- `web/bird.html:277` retries `/join` up to 150 times at `retry_ms`; with the
  server's 1200 ms that is three minutes of polling during a long turn. Fine,
  but the server should send a `retry_ms` that grows.
- `tools/cert.sh:14` falls back to `127.0.0.1` when no interface is found, and
  the banner then tells the user to open `https://127.0.0.1:8000` on each
  device, which no phone can reach. Fail instead.

## What to do first

1.1 (the crash), 1.2 (the sweeper race) and 1.5 (frame matching) are small,
contained changes in `server/` that remove the ways the coordinator can die or
lie, and each has an obvious e2e assertion. 1.3 plus section 2's `Origin`
check are the minimum before anyone but the operator's own devices should be
on the network with it. Everything else can follow in the order above.
