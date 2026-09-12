// Stand-in bird against the Node coordinator (websocket path).
//
// Runs the same shards a phone would, so the whole chain can be exercised on one
// machine with no browser. Not a fossil: it is the only way to test the
// coordinator end to end in CI or over ssh, where there is no WebGPU.
//
//   node sim_bird.mjs            claim one free slot
//   node sim_bird.mjs solo       claim EVERY slot, so the flock is covered alone
//   FLOCK_URL=http://host:port   point at a coordinator elsewhere
//
// Each simulated bird keeps its own K/V cache, exactly as a real bird does — the
// point is to exercise the sharded-cache path, not to shortcut it.
import ort from 'onnxruntime-node';
import WebSocket from 'ws';
import path from 'path';
import {fileURLToPath} from 'url';
import {pack, unpack} from '../web/js/wire.mjs';

// Resolve the shards relative to THIS file, not the cwd: a hardcoded absolute
// path made the sim load another checkout's weights when run from a worktree.
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const BASE = process.env.FLOCK_URL || 'http://127.0.0.1:8000';
const solo = process.argv[2] === 'solo';
const label = solo ? 'sim' : (process.argv[2] || 'sim');

/** Claim a slot, load its shard, and serve frames until the socket closes. */
async function bird(tag) {
  const j = await (await fetch(`${BASE}/join`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({label: tag})})).json();
  if (j.error) throw new Error(j.error);
  const s = await ort.InferenceSession.create(
    path.join(ROOT, `web/shard${j.slot}.onnx`));
  console.log(`bird ${j.slot}: layers ${j.start}-${j.end}`);

  let past = null;
  const empty = () => new ort.Tensor('float32', new Float32Array(0),
                                     [1, j.kv_heads, 0, j.head_dim]);
  const ws = new WebSocket(BASE.replace(/^http/, 'ws') + '/ws');
  let beat = null;
  ws.on('open', () => {
    ws.send(JSON.stringify({peer_id: j.peer_id, label: tag}));
    // Heartbeat, same as a real bird: liveness is judged on lastSeen, so an idle
    // sim silently ages out of the flock after 40s without this.
    ws.send(JSON.stringify({t: 'ping'}));
    beat = setInterval(() => ws.send(JSON.stringify({t: 'ping'})), 4000);
  });
  ws.on('close', () => clearInterval(beat));
  ws.on('error', e => console.error(`bird ${j.slot} socket: ${e.message}`));
  ws.on('message', async (data, isBinary) => {
    if (!isBinary) return;
    const ab = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    const {data: h, meta: m} = unpack(ab);
    if (m.reset) past = null;
    const feed = {
      hidden: new ort.Tensor('float32', h, [1, m.seq, j.hidden]),
      position_ids: new ort.Tensor('int64',
        BigInt64Array.from({length: m.seq}, (_, i) => BigInt(m.offset + i)), [1, m.seq]),
    };
    for (let i = 0; i < j.n_layers; i++) {
      feed[`past_k${i}`] = past ? past[`new_k${i}`] : empty();
      feed[`past_v${i}`] = past ? past[`new_v${i}`] : empty();
    }
    const t0 = performance.now();
    const out = await s.run(feed);
    const ms = +(performance.now() - t0).toFixed(1);
    past = {};
    for (let i = 0; i < j.n_layers; i++) {
      past[`new_k${i}`] = out[`new_k${i}`];
      past[`new_v${i}`] = out[`new_v${i}`];
    }
    ws.send(Buffer.from(pack(out.output.data,
      {seq: m.seq, hidden: j.hidden, offset: m.offset})));
    ws.send(JSON.stringify({t: 'stats', ms}));
  });
  return j;
}

if (!solo) {
  await bird(label);
} else {
  // Claim slots until /join says the flock is full. Sequential on purpose: the
  // coordinator hands out the lowest free slot, so racing would be ambiguous.
  const {birds} = await (await fetch(`${BASE}/status`)).json();
  for (let i = 0; i < birds.length; i++) await bird(`sim${i}`);
  console.log(`solo: holding all ${birds.length} slots`);
}
