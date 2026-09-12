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

console.log(`loading coordinator (layers 0-${CUT - 1}) ...`);
const coord = await Coordinator.load();
const flock = new Flock(RANGES);
console.log(`coordinator holds layers 0-${CUT - 1}; birds hold ` +
            RANGES.map(([s, e]) => `${s}-${e}`).join(', '));
console.log('kv cache: ON  |  wire: f16 binary  |  links: webrtc');

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
            kv_cache: META.kv_cache, n_total: META.n_total, model: META.model});
});

// Birds have no readable console, so they POST failures here.
app.post('/diag', (req, res) => {
  const {stage, detail, ua} = req.body || {};
  const dev = /iPad/.test(ua) ? 'iPad' : /iPhone/.test(ua) ? 'iPhone'
            : /Macintosh/.test(ua) ? 'Mac/iPad' : 'device';
  console.log(`[diag ${dev}] ${stage}: ` +
              (typeof detail === 'object' ? JSON.stringify(detail) : detail));
  res.json({ok: true});
});

app.get('/', (_, res) => res.sendFile(path.join(ROOT, 'web/chat.html')));
app.get('/flock', (_, res) => res.sendFile(path.join(ROOT, 'web/bird.html')));
app.get('/shard/:slot.onnx', (req, res) =>
  res.sendFile(path.join(ROOT, `web/shard${req.params.slot}.onnx`)));
app.get('/shard/:slot.onnx.data', (req, res) =>
  res.sendFile(path.join(ROOT, `web/shard${req.params.slot}.onnx.data`)));

app.get('/status', (_, res) => res.json({
  ready: flock.ready(), missing: flock.missing(),
  coord_layers: `0-${CUT - 1}`, birds: flock.birds.map(b => b.info()),
  n_total: META.n_total, hidden: META.hidden, kv_cache: META.kv_cache,
  wire: 'f16', runtime: 'node',
}));

app.post('/chat', async (req, res) => {
  const {prompt, max_tokens = 24} = req.body;
  res.set({'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache',
           'X-Accel-Buffering': 'no'});
  const sse = o => res.write(`data: ${JSON.stringify(o)}\n\n`);

  if (!flock.ready()) {
    sse({type: 'error', text: 'waiting for devices to cover layers ' +
         flock.missing().join(', ') + ' — open /flock on each phone and tap join'});
    return res.end();
  }

  const ids = coord.encode(prompt);
  coord.reset();
  flock.reset();
  const out = [];
  let stepIds = ids, offset = 0;

  for (let step = 0; step < max_tokens; step++) {
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
      sse({type: 'error', text: e.message});
      return res.end();
    }
    const stats = flock.birds.map(b => ({
      range: `${b.start}-${b.end}`, ms: b.lastMs,
      label: b.label, transport: b.transport}));
    const netMs = +(performance.now() - t1).toFixed(1);

    const nxt = await coord.project(flat, stepIds.length);
    const nFloats = stepIds.length * META.hidden;
    sse({type: 'hop', coord_ms: coordMs, birds: stats, roundtrip_ms: netMs,
         kb: +(nFloats * 2 / 1024).toFixed(1),
         kb_json: +(nFloats * 21 / 1024).toFixed(1)});

    if (nxt === coord.meta.eos) break;
    out.push(nxt);
    sse({type: 'token', text: coord.decode([nxt])});
    offset += stepIds.length;
    stepIds = [nxt];
  }
  sse({type: 'done', text: coord.decode(out)});
  res.end();
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
      else if (m.t === 'pull') bird.lastSeen = Date.now();
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

server.listen(8000, '0.0.0.0', () =>
  console.log('flock listening on http://0.0.0.0:8000'));
