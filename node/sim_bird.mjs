// Stand-in bird against the coordinator (websocket path).
//
// Runs the same layers a phone would, with the same WGSL kernels, so the whole
// chain can be exercised on one machine with no browser. Not a fossil: it is the
// only way to test the coordinator end to end over ssh or in a script, where
// there is no browser to open /flock in.
//
// RUN IT WITH DENO, NOT NODE -- it needs a GPU for the same reason the
// coordinator does, and Node has no WebGPU:
//
//   deno run --unstable-webgpu --allow-all sim_bird.mjs          one free slot
//   deno run --unstable-webgpu --allow-all sim_bird.mjs solo     every slot
//   FLOCK_URL=http://host:port                                   a coordinator elsewhere
//
// Each simulated bird keeps its own K/V cache, exactly as a real bird does — the
// point is to exercise the sharded-cache path, not to shortcut it.
//
// WEIGHTS COME FROM THE SAME PLACE A REAL BIRD'S DO: the GGUF file, by byte
// range, over the network. There is no exported shard to load any more. What
// differs from bird.html is only HOW they reach the GPU -- this reads the
// cached whole-model file through real_weights.ts, while a phone streams its
// range so the bytes never sit in the JS heap (see web/js/gguf-stream.mjs).
// Both end up calling the same Layer with the same numbers.
import WebSocket from 'ws';
import {pack, unpack} from '../web/js/wire.mjs';
import {getDevice} from '../kernels/lib.ts';
import {Layer, QWEN3_06B} from '../kernels/layer.ts';
import {realModel} from '../kernels/real_weights.ts';

const BASE = process.env.FLOCK_URL || 'http://127.0.0.1:8000';
const solo = process.argv[2] === 'solo';
const label = solo ? 'sim' : (process.argv[2] || 'sim');

const dev = await getDevice();
console.log('reading weights ...');
const weights = await realModel();

/** Claim a slot, build its layers on the GPU, and serve frames until the socket
 *  closes. */
async function bird(tag) {
  const j = await (await fetch(`${BASE}/join`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({label: tag})})).json();
  if (j.error) throw new Error(j.error);

  const layers = [];
  for (let i = j.start; i <= j.end; i++) {
    layers.push(await Layer.create(dev, weights.layers[i], QWEN3_06B));
  }
  console.log(`bird ${j.slot}: layers ${j.start}-${j.end} on the GPU`);

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
    // A reset means the conversation restarted, so OUR shard of the K/V cache is
    // stale too. Each layer owns its own, so each one clears it.
    if (m.reset) for (const L of layers) L.reset();

    const t0 = performance.now();
    // Chain the layers on the GPU, the way kernels/model.ts chains its 28: layer i
    // reads what layer i-1 wrote via a device-to-device copy, everything shares
    // ONE command buffer, and only the shard's final output comes back to the
    // host. A readback per layer instead would cost ~24 ms each -- measured 20.8x
    // for this exact difference in bench_layer.ts.
    const enc = dev.createCommandEncoder();
    for (let i = 0; i < layers.length; i++) {
      const L = layers[i];
      if (i === 0) {
        // Only the first layer takes the hidden state off the wire.
        if (m.seq === 1) L.encode(h, enc);
        else L.encodePrefill(m.seq, h, enc);
      } else {
        enc.copyBufferToBuffer(layers[i - 1].outputBuffer(), 0,
                               L.inputBuffer(), 0, m.seq * j.hidden * 4);
        // No hidden argument: consume what the copy just put in our own buffer.
        if (m.seq === 1) L.encode(undefined, enc);
        else L.encodePrefill(m.seq, undefined, enc);
      }
    }
    dev.queue.submit([enc.finish()]);
    const flat = await layers[layers.length - 1].readOutput(m.seq);
    const ms = +(performance.now() - t0).toFixed(1);

    ws.send(pack(flat, {seq: m.seq, hidden: j.hidden, offset: m.offset}));
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
