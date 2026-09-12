// flock_server (Node) — runs the whole flock with WebRTC on every link.
//
// The coordinator holds the embedding, layers 0..cut-1 and the vocab
// projection; each phone (a "bird") holds a contiguous slice of the rest.
// Per token the hidden state makes one lap: coordinator -> bird -> bird -> back.
//
// Because this is Node rather than Python, the coordinator is itself a WebRTC
// peer: it offers a data channel to each bird, and once that opens the
// websocket carries nothing but signaling.
//
// Run:  npm start   (from node/)
import express from 'express';
import {WebSocketServer} from 'ws';
import {readFileSync, existsSync} from 'fs';
import {createServer} from 'http';
import {randomBytes} from 'crypto';
import {networkInterfaces} from 'os';
import path from 'path';
import {fileURLToPath} from 'url';

import {Coordinator} from './coordinator.js';
import {Flock} from './mesh.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
process.chdir(ROOT);

for (const f of ['web/flock.json', 'web/coord/coord.json']) {
  if (!existsSync(f)) {
    console.error(`missing ${f} — build first:\n` +
      '    python3 build_shards.py --start 24 --end 27 --birds 1\n' +
      '    python3 flock_export_coordinator.py --cut 24');
    process.exit(1);
  }
}

const META = JSON.parse(readFileSync('web/flock.json', 'utf8'));
const RANGES = META.birds.map(b => [b.start, b.end]);
const CUT = RANGES[0][0];

// Where a bird can fetch GGUF weights for itself, if it would rather do that than
// download an exported ONNX shard from us. Handed out by /join so the bird needs
// no configuration of its own, and left empty to keep every bird on the ONNX path
// -- which is still the reference the WGSL engine is validated against, so this is
// opt-in rather than a switch that flips underneath it.
//
// It is a plain URL rather than something we proxy on purpose: the point of the
// GGUF path is that weights go from the model host straight to the device, so the
// coordinator never touches 67MB per bird and needs no build step and no disk.
const GGUF_URL = process.env.FLOCK_GGUF || '';

console.log(`loading coordinator (layers 0-${CUT - 1}) ...`);
const coord = await Coordinator.load();
const flock = new Flock(RANGES);
console.log(`coordinator holds layers 0-${CUT - 1}; birds hold ` +
            RANGES.map(([s, e]) => `${s}-${e}`).join(', '));
console.log('kv cache: ON  |  wire: f16 binary  |  links: webrtc');

// The LAN address is the one thing you cannot guess, and you need it to open
// /flock on a phone. Print it rather than making the user go find it.
function localAddresses() {
  return Object.values(networkInterfaces()).flat()
    .filter(i => i && i.family === 'IPv4' && !i.internal).map(i => i.address);
}

const app = express();
app.use(express.json());
app.use('/js', express.static('web/js'));

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

// A slot is an index into the flock, so anything else is a bad request rather
// than a path to go looking for on disk.
const slotFile = suffix => (req, res) => {
  const slot = +req.params.slot;
  if (!Number.isInteger(slot) || slot < 0 || slot >= RANGES.length)
    return res.status(404).json({error: `no slot ${req.params.slot}`});
  res.sendFile(`web/shard${slot}${suffix}`, {root: ROOT}, err => {
    if (!err) return;
    console.error(`[shard] slot ${slot}${suffix}: ${err.message}`);
    if (!res.headersSent)
      res.status(404).json({error: `shard ${slot}${suffix} missing — run build_shards.py`});
  });
};
app.get('/shard/:slot.onnx', slotFile('.onnx'));
app.get('/shard/:slot.onnx.data', slotFile('.onnx.data'));

app.get('/status', (_, res) => res.json({
  ready: flock.ready(), missing: flock.missing(),
  coord_layers: `0-${CUT - 1}`, coord_n_layers: CUT,
  birds: flock.birds.map(b => b.info()),
  n_total: META.n_total, hidden: META.hidden, kv_cache: META.kv_cache,
  wire: 'f16', runtime: 'node', model: META.model,
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

const server = createServer(app);
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
  for (const a of localAddresses()) console.log(`  http://${a}:${PORT}`);
  console.log(`flock listening on port ${PORT} — chat at / , birds join at /flock`);
});
