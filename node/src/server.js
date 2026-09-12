// flock_server — runs the whole flock with WebRTC on every link.
//
// The coordinator holds the embedding, layers 0..cut-1, output_norm and the vocab
// projection; each phone (a "bird") holds a contiguous slice of the rest.
// Per token the hidden state makes one lap: coordinator -> bird -> bird -> back.
//
// Because this is a real server rather than a Python script, the coordinator is
// itself a WebRTC peer: it offers a data channel to each bird, and once that
// opens the websocket carries nothing but signaling.
//
// RUN IT WITH DENO, NOT NODE:
//
//   deno task start            (from node/, or `npm start` which calls it)
//
// The coordinator runs Qwen3's first 24 layers as WGSL compute kernels, so it
// needs a GPU, and Node has no WebGPU -- no `navigator` at all, and no flag that
// adds one. Deno provides a device and also runs express, ws and node-datachannel
// unchanged through its node compatibility layer, so the whole server is one
// process on one runtime. See node/README.md for what was measured.
//
// THERE IS NO BUILD STEP. The topology used to come from web/flock.json, written
// by build_shards.py alongside exported ONNX graphs. With GGUF a split is just a
// choice about byte ranges, so it is computed here at startup from the model's
// header: nothing is exported, nothing is written to disk, and the split can
// change without rebuilding anything.
import express from 'express';
import {WebSocketServer} from 'ws';
import {createServer} from 'node:http';
import {createServer as createHttpsServer} from 'node:https';
import {existsSync, readFileSync} from 'node:fs';
import {randomBytes} from 'node:crypto';
import {networkInterfaces} from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

import {Coordinator} from './coordinator.js';
import {Flock} from './mesh.js';
import {readModel, splitLayers} from './gguf.mjs';
import {QWEN3_06B} from '../../kernels/layer.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
process.chdir(ROOT);

// Where every device -- birds AND this coordinator -- reads its weights from.
// A plain URL rather than something we proxy, on purpose: the point of the GGUF
// path is that weights go from the model host straight to the device, so the
// coordinator never moves 67MB per bird and needs no build step and no disk.
const GGUF_URL = process.env.FLOCK_GGUF ||
  'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf';

// How the layers are divided. Only the model's HEADER is read to decide this --
// a few MB, not the weights -- which is what makes the topology a startup
// decision rather than a build artifact.
// How many devices share the bird-side layers. Every bird gets a DIFFERENT
// contiguous range: with 1 the single bird holds them all, which looks like
// "every device has the same layers" to anyone pointing two phones at it.
const N_BIRDS = +(process.env.FLOCK_BIRDS || 1);
console.log(`reading GGUF header: ${GGUF_URL.split('/').pop()}`);
const header = await readModel(GGUF_URL);
const N_TOTAL = header.nLayers;
// Birds hold the last `FLOCK_BIRD_LAYERS` layers; the coordinator holds the rest.
// Default 4, which is flock's long-standing split (coordinator 0-23, birds 24-27).
const BIRD_LAYERS = Math.min(N_TOTAL - 1, +(process.env.FLOCK_BIRD_LAYERS || 4));
const CUT = N_TOTAL - BIRD_LAYERS;
const RANGES = splitLayers(CUT, N_TOTAL - 1, N_BIRDS);

const META = {
  model: header.metadata['general.name'] || 'Qwen3-0.6B',
  n_total: N_TOTAL,
  hidden: QWEN3_06B.hidden,
  kv_heads: QWEN3_06B.nKvHeads,
  head_dim: QWEN3_06B.headDim,
  kv_cache: true,
  coord_layers: [0, CUT - 1],
  birds: RANGES.map(([s, e], i) => ({slot: i, start: s, end: e})),
};

console.log(`loading coordinator (layers 0-${CUT - 1}) on the GPU ...`);
const coord = await Coordinator.load(CUT);
const flock = new Flock(RANGES);
console.log(`coordinator holds layers 0-${CUT - 1}; birds hold ` +
            RANGES.map(([s, e]) => `${s}-${e}`).join(', '));
console.log('kv cache: ON  |  wire: f16 binary  |  links: webrtc  |  engine: wgsl');

// The LAN address is the one thing you cannot guess, and you need it to open
// /flock on a phone. Print it rather than making the user go find it.
function localAddresses() {
  return Object.values(networkInterfaces()).flat()
    .filter(i => i && i.family === 'IPv4' && !i.internal).map(i => i.address);
}

const app = express();
app.use(express.json());
app.use('/js', express.static('web/js'));

// A bird runs the SAME kernels the tests validate, fetched from here rather than
// carrying a copy -- so a kernel fix reaches every bird on reload, and there is
// no second implementation to keep in step with kernels/.
//
// The .wgsl files go out as-is: wgslSource() in kernels/layer.ts resolves them
// against its own module URL, which is a file: URL under Deno and an http: one in
// the page, so one expression serves both runtimes.
//
// The .ts files cannot go out as-is, because browsers do not execute TypeScript.
// They are transpiled ON REQUEST and their `./x.ts` import specifiers rewritten to
// `./x.ts.js`, so the browser follows the same module graph Deno does. This is
// deliberately not a build step: nothing is written to disk, there is no artifact
// to rebuild or forget to rebuild, and the file the bird runs is derived from the
// file the tests run every time it is asked for. Transpiling is type STRIPPING
// only -- no type checking, which is what `deno check` is for -- and the result is
// cached in memory per mtime so a reload does not re-emit.
const TS_CACHE = new Map();
app.get(/^\/kernels\/(.+\.ts)\.js$/, async (req, res) => {
  const rel = req.params[0];
  // Only the kernels directory, and no traversal out of it: `rel` reaches the
  // filesystem, so a crafted path must not escape.
  if (rel.includes('..') || !/^[\w./-]+$/.test(rel)) {
    return res.status(400).type('text/plain').send('bad kernel path');
  }
  const abs = path.join(ROOT, 'kernels', rel);
  try {
    const {mtimeMs} = await import('node:fs/promises').then(fs => fs.stat(abs));
    const hit = TS_CACHE.get(abs);
    if (hit && hit.mtimeMs === mtimeMs) {
      return res.type('application/javascript').send(hit.js);
    }
    const {transpile} = await import('jsr:@deno/emit');
    const url = pathToFileURL(abs);
    const out = await transpile(url);
    // Rewrite relative .ts specifiers to the .ts.js route above, so the graph
    // resolves in the browser. Only RELATIVE ones: a bare specifier would be a
    // dependency the page has no import map for, and silently rewriting it would
    // produce a 404 that looks like a missing kernel.
    // kernels/ sits next to web/ on disk, so a kernel importing
    // "../web/js/wire.mjs" is correct as a FILE path but a 404 as a URL: that
    // file is served at /js/, not /web/js/. Left alone it takes down the whole
    // module graph before any page code runs, which looks like a page that
    // simply never loads.
    const js = out.get(url.href)
      .replace(/(from\s*["'])(\.\.?\/[^"']+\.ts)(["'])/g, '$1$2.js$3')
      .replace(/(import\s*\(\s*["'])(\.\.?\/[^"']+\.ts)(["']\s*\))/g, '$1$2.js$3')
      .replace(/(["'])\.\.\/web\/js\//g, '$1/js/');
    TS_CACHE.set(abs, {mtimeMs, js});
    res.type('application/javascript').send(js);
  } catch (e) {
    console.error(`[kernels] ${rel}: ${e.message}`);
    res.status(404).type('text/plain').send(`cannot serve ${rel}: ${e.message}`);
  }
});
app.use('/kernels', express.static('kernels'));

app.post('/join', (req, res) => {
  const pid = req.body.peer_id || randomBytes(4).toString('hex');
  const bird = flock.claim(pid, req.body.label || 'phone');
  if (!bird) return res.status(409).json({error: 'flock full — every layer slot is taken'});
  res.json({peer_id: pid, slot: bird.slot, start: bird.start, end: bird.end,
            n_layers: bird.end - bird.start + 1, hidden: META.hidden,
            kv_heads: META.kv_heads, head_dim: META.head_dim,
            kv_cache: META.kv_cache, n_total: META.n_total, model: META.model,
            gguf: GGUF_URL});
});

// Birds have no readable console, so they POST failures here. Kept in a ring
// buffer as well as logged: /status serves it, so the chat page can show what a
// phone reported without anyone reading the terminal.
const DIAG = [];
// A device that cannot run must hand its slot back immediately. Without this a
// bird that fails to load holds its layers until the coordinator restarts, and
// the flock stalls on a device that will never compute anything -- which has
// blocked a real join twice.
app.post('/leave', (req, res) => {
  const {peer_id, why} = req.body || {};
  const bird = flock.byPeer(peer_id);
  if (!bird) return res.json({ok: true, released: false});
  console.log(`[flock] ${bird.label || 'a device'} released slot ${bird.slot} ` +
              `(layers ${bird.start}-${bird.end})${why ? ': ' + why : ''}`);
  bird.release();
  flock.announce();
  res.json({ok: true, released: true});
});

// Evict a slot whose device is gone but never said goodbye (a crashed tab, a
// phone that slept). Without it the only cure is restarting the coordinator.
app.post('/evict', (req, res) => {
  const slot = Number((req.body || {}).slot);
  const bird = flock.birds[slot];
  if (!bird) return res.status(404).json({error: `no slot ${slot}`});
  if (bird.alive()) {
    return res.status(409).json({error:
      `slot ${slot} is held by a live device (${bird.label}); it must leave first`});
  }
  bird.release();
  flock.announce();
  res.json({ok: true, slot, layers: `${bird.start}-${bird.end}`});
});

app.post('/diag', (req, res) => {
  const {stage, detail, ua, slot} = req.body || {};
  const dev = /iPad/.test(ua) ? 'iPad' : /iPhone/.test(ua) ? 'iPhone'
            : /Android/.test(ua) ? 'Android'
            : /Macintosh/.test(ua) ? 'Mac/iPad' : 'device';
  const text = typeof detail === 'object' ? JSON.stringify(detail) : String(detail);
  console.log(`[diag ${dev}${slot != null ? ` slot ${slot}` : ''}] ${stage}: ${text}`);
  DIAG.push({at: Date.now(), device: dev, slot: slot ?? null,
             stage: String(stage || '?'), detail: text.slice(0, 400)});
  if (DIAG.length > 60) DIAG.shift();
  res.json({ok: true});
});

// Serve relative to ROOT, not as an absolute path: send() rejects any path with
// a dot-prefixed segment as a hidden file, so an absolute path through a
// checkout under e.g. ~/.worktrees/ 404s every page. With `root` set it only
// inspects the relative part, which also keeps a crafted :slot from escaping.
const page = rel => (_, res) => res.sendFile(rel, {root: ROOT});
app.get('/', page('web/chat.html'));
app.get('/flock', page('web/bird.html'));

// A standalone hardware report. Separate from /flock because it must work even
// when the bird page cannot: it claims no slot, loads no weights, and its only
// job is to say WHY a device is or is not usable as a bird.
app.get('/check', page('web/inspect.html'));

app.get('/status', (_, res) => res.json({
  ready: flock.ready(), missing: flock.missing(),
  coord_layers: `0-${CUT - 1}`, coord_n_layers: CUT,
  birds: flock.birds.map(b => b.info()),
  n_total: META.n_total, hidden: META.hidden, kv_cache: META.kv_cache,
  wire: 'f16', runtime: 'deno', engine: 'wgsl', model: META.model,
  // Conversation state, so the chat UI can show how much context is cached and
  // whether a turn is already in flight.
  cached_tokens: convo.fed, turns: convo.turns, busy: convo.busy,
  diag: DIAG.slice(-12),
}));

// Liveness for scripts and for `curl` on a phone: one object, no model load, and
// an HTTP status that says whether the flock could answer a prompt right now.
app.get('/health', (_, res) => {
  const birds = flock.birds.map(b => b.info());
  const ok = flock.ready();
  res.status(ok ? 200 : 503).json({
    ok, uptime_s: +process.uptime().toFixed(0),
    missing: flock.missing(),
    birds_alive: birds.filter(b => b.alive).length, birds_total: birds.length,
    busy: convo.busy, cached_tokens: convo.fed,
    rss_mb: +(process.memoryUsage().rss / 1e6).toFixed(0),
  });
});

// Conversation state. `fed` is how many tokens the caches have seen, which is
// also the next position id — the coordinator's cache and every bird's cache all
// advance together, so one counter describes the whole flock.
const convo = {fed: 0, turns: 0, busy: false};

/** Drop every cache in the flock and start the conversation over. */
function resetConvo() {
  coord.reset();
  flock.reset();     // each bird clears its own shard of the cache on next frame
  convo.fed = 0;
  convo.turns = 0;
}

app.post('/reset', (_, res) => {
  if (convo.busy) return res.status(409).json({error: 'a turn is in flight'});
  resetConvo();
  res.json({ok: true});
});

app.post('/chat', async (req, res) => {
  const {prompt, max_tokens = 96, reset = false} = req.body;
  res.set({'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache',
           'X-Accel-Buffering': 'no'});
  const sse = o => res.write(`data: ${JSON.stringify(o)}\n\n`);

  if (!prompt || !String(prompt).trim()) {
    sse({type: 'error', text: 'empty prompt'});
    return res.end();
  }
  if (!flock.ready()) {
    sse({type: 'error', text: 'waiting for devices to cover layers ' +
         flock.missing().join(', ') + ' — open /flock on each phone and tap join'});
    return res.end();
  }
  // One conversation, one cache, no batching: a second concurrent turn would
  // interleave writes into the same K/V and corrupt both answers.
  if (convo.busy) {
    sse({type: 'error', text: 'the flock is already generating — one turn at a time'});
    return res.end();
  }
  convo.busy = true;

  // Stop generating if the browser goes away mid-turn. Watch the RESPONSE, not
  // the request: express.json() consumes and destroys the request stream while
  // parsing the body, so req.destroyed is already true by the time we get here
  // and would abort every turn after its first token.
  let gone = false;
  res.on('close', () => { gone = true; });

  try {
    if (reset) resetConvo();
    // Append this turn to the warm cache rather than re-prefilling the history.
    // See Coordinator.turnTokens for why re-encoding would be wrong here.
    let stepIds = coord.turnTokens(prompt, convo.turns === 0);
    let offset = convo.fed;
    const promptTokens = stepIds.length;
    const out = [];
    let stop = 'length';

    sse({type: 'turn', turn: convo.turns, prompt_tokens: promptTokens,
         cached_tokens: offset});

    for (let step = 0; step < max_tokens; step++) {
      if (gone) { stop = 'aborted'; break; }

      const t0 = performance.now();
      let flat = await coord.forward(stepIds, offset);
      const coordMs = +(performance.now() - t0).toFixed(1);

      const t1 = performance.now();
      try {
        // One lap: into the first bird, out of the last. Birds forward to each
        // other directly, so the coordinator sees a single round trip no matter
        // how many devices are in the chain.
        const {data} = await flock.lap(flat, stepIds.length, META.hidden, offset);
        flat = data;
      } catch (e) {
        // A bird dropping mid-turn leaves the caches at an offset no device
        // agrees on any more, so the conversation cannot be continued.
        resetConvo();
        sse({type: 'error', text: e.message});
        return res.end();
      }
      const stats = flock.birds.map(b => ({
        slot: b.slot, range: `${b.start}-${b.end}`, ms: b.lastMs,
        label: b.label, transport: b.transport}));
      const netMs = +(performance.now() - t1).toFixed(1);

      const nxt = await coord.project(flat, stepIds.length);
      const nFloats = stepIds.length * META.hidden;

      // Positions are consumed whether or not we keep the token, so advance the
      // cache offset before deciding to stop -- otherwise the next turn would
      // feed overlapping position ids into a cache that already holds them.
      offset += stepIds.length;
      convo.fed = offset;

      sse({type: 'hop', step, coord_ms: coordMs, birds: stats,
           roundtrip_ms: netMs, prefill: step === 0,
           seq: stepIds.length, cached: offset,
           kb: +(nFloats * 2 / 1024).toFixed(1),
           kb_json: +(nFloats * 21 / 1024).toFixed(1)});

      if (nxt === coord.meta.eos) { stop = 'eos'; break; }
      out.push(nxt);
      sse({type: 'token', text: coord.decode([nxt])});
      stepIds = [nxt];
    }

    // An aborted turn stops mid-answer: the caches hold a partial assistant
    // reply with no closing token, so the next turn cannot append cleanly.
    // Start over rather than continue from a torn conversation.
    if (stop === 'aborted') {
      resetConvo();
      return res.end();
    }

    // The generated tokens are in the cache now, so the next turn continues from
    // here. Count the turn only if it produced something to continue from.
    if (out.length) convo.turns++;
    sse({type: 'done', text: coord.decode(out), stop,
         tokens: out.length, prompt_tokens: promptTokens, cached: convo.fed,
         turn: convo.turns});
    res.end();
  } catch (e) {
    console.error(`[chat] ${e.stack || e.message}`);
    resetConvo();
    try { sse({type: 'error', text: `coordinator failed: ${e.message}`}); } catch {}
    try { res.end(); } catch {}
  } finally {
    convo.busy = false;
  }
});

// Chrome and Edge expose WebGPU only in a SECURE CONTEXT, so a bird reached over
// plain http:// on a LAN sees no navigator.gpu at all -- which looks exactly like
// a device with no GPU and is not. Safari does not gate it this way, which is why
// it works where Chrome cannot. Serving https with a self-signed cert makes every
// browser usable; the cert has to be trusted once per device.
const CERT = path.join(ROOT, '.certs', 'cert.pem');
const KEY = path.join(ROOT, '.certs', 'key.pem');
const hasCert = existsSync(CERT) && existsSync(KEY);
const SCHEME = hasCert ? 'https' : 'http';
const server = hasCert
  ? createHttpsServer({cert: readFileSync(CERT), key: readFileSync(KEY)}, app)
  : createServer(app);
const wss = new WebSocketServer({server, path: '/ws'});

wss.on('connection', ws => {
  let bird = null;
  ws.on('message', (data, isBinary) => {
    if (!isBinary) {
      const m = JSON.parse(data.toString());
      if (m.peer_id && !bird) {                 // hello
        bird = flock.byPeer(m.peer_id);
        if (!bird) return ws.send(JSON.stringify({error: 'unknown peer — rejoin'}));
        bird.ws = ws; bird.label = m.label || bird.label;
        bird.transport = 'ws'; bird.lastSeen = Date.now();
        // Offer the bird a direct data channel; the websocket then only
        // carries signaling.
        bird.openRTC(b => console.log(`webrtc link open to ${b.label} (${b.start}-${b.end})`));
        flock.announce();          // tell everyone who their successor is
        return;
      }
      if (!bird) return;
      if (m.t === 'signal') {
        // `to` means bird->bird: we are only the introducer, exactly the role
        // PeerJS plays for swarmllm. No `to` means it is meant for us.
        if (m.to) {
          const dst = flock.byPeer(m.to);
          if (dst?.ws) dst.ws.send(JSON.stringify(
            {t: 'signal', from: bird.peerId, data: m.data}));
        } else {
          bird.onSignal(m.data);
        }
      }
      else if (m.t === 'stats') { bird.lastMs = m.ms; bird.lastSeen = Date.now(); }
      // Whether this bird can hand off peer-to-peer decides how far the
      // coordinator's chained wait should reach.
      else if (m.t === 'forwards') bird.forwardsDirectly = !!m.direct;
      // note: transport is set from the send path in mesh.js, not from here --
      // what actually carried the frame is the only honest answer.
      // Liveness heartbeat. Without it an idle bird ages past alive()'s grace
      // period and the flock reports itself uncovered until a turn starts.
      // `pull` is the old name from the long-polling transport -- still accepted
      // so an un-refreshed bird page keeps its slot.
      else if (m.t === 'ping' || m.t === 'pull') bird.lastSeen = Date.now();
    } else if (bird) {
      bird.deliver(data);
    }
  });
  ws.on('close', () => {
    if (!bird) return;
    // A refresh closes the websocket; the data channel it signalled is dead
    // too, so tear the whole link down rather than leaving chan dangling.
    if (bird.ws === ws) { bird.teardown(); flock.announce(); }
  });
});

const PORT = +(process.env.PORT || 8000);

// A raw EADDRINUSE stack says nothing about what to do next, and this is the
// most common way starting the coordinator fails (a previous run still holding
// the port). Name the fix instead.
server.on('error', e => {
  if (e.code === 'EADDRINUSE') {
    console.error(`port ${PORT} is already in use — another flock is running.\n` +
                  `    lsof -nP -iTCP:${PORT} -sTCP:LISTEN     # who has it\n` +
                  `    PORT=${PORT + 1} npm start              # or use another port`);
    process.exit(1);
  }
  console.error(`server error: ${e.message}`);
  process.exit(1);
});

server.listen(PORT, '0.0.0.0', () => {
  // Only the real LAN address is reachable from a phone; bridge and vpn
  // interfaces just add noise to a line someone has to type on a tablet.
  const addrs = localAddresses();
  const lan = addrs.filter(a => /^192\.168\.|^10\.|^172\.(1[6-9]|2\d|3[01])\./.test(a)
                                && !a.endsWith('.0'));
  for (const a of (lan.length ? lan : addrs)) {
    console.log(`  ${SCHEME}://${a}:${PORT}`);
  }
  console.log(`flock listening on port ${PORT} — chat at / , birds join at /flock` +
              `, device check at /check`);
  console.log(`  ${N_BIRDS} bird slot${N_BIRDS === 1 ? '' : 's'}: ` +
              RANGES.map(([s, e]) => `${s}-${e}`).join(', ') +
              (N_BIRDS === 1
                ? '  (FLOCK_BIRDS=2 to split these across two devices)'
                : ''));
  if (!hasCert) {
    console.log('\n  NOTE: serving plain http, so Chrome and Edge will hide WebGPU');
    console.log('  (they require a secure context; Safari does not). To fix:');
    console.log('    npm run cert');
  } else {
    console.log('\n  https with a self-signed cert: each device must trust it once');
    console.log('  (open the URL, accept the warning), then WebGPU works in any browser');
  }
});
