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
//   deno run --unstable-webgpu --allow-all sim_bird.mjs            one device
//   deno run --unstable-webgpu --allow-all sim_bird.mjs solo       one device, all layers
//   deno run --unstable-webgpu --allow-all sim_bird.mjs solo 3     three devices
//   FLOCK_URL=http://host:port                                     a coordinator elsewhere
//
// WHY IT CAN LIE ABOUT ITSELF. The allocator weights layers by each device's real
// limits and measured speed, and there is no way to test that without devices that
// differ -- which would mean owning a drawer of phones. So a simulated bird can
// report ARBITRARY capability and speed:
//
//   --bind 64MB        maxStorageBufferBindingSize to report at /join
//   --budget 300MB     how many weight bytes it claims it will hold
//   --slow 40ms        extra delay per layer per token, to look like a phone
//   --label iPhone     what /status calls it
//   --fake             do not touch the GPU at all: echo the hidden state back
//
// --fake is what makes the allocation tests runnable anywhere: a fake bird exercises
// /join, the chain, the timings and the rebalancer without a device and without
// 67MB of weights, so "a deliberately slow bird ends up with fewer layers" is a
// test rather than an anecdote. A REAL simulated bird (no --fake) still computes
// real layers with real kernels, which is what proves the text stays correct.
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
import {existsSync} from 'node:fs';
import {pack, unpack} from '../web/js/wire.mjs';

// The coordinator serves https when .certs/ exists (Chrome hides WebGPU outside a
// secure context), so default to it and fall back to http for a plain run. A
// self-signed cert also means Deno must be told not to verify -- see the
// DENO_TLS_CA_STORE note in the header comment.
const BASE = process.env.FLOCK_URL ||
  (existsSync(new URL('../.certs/cert.pem', import.meta.url))
    ? 'https://127.0.0.1:8000' : 'http://127.0.0.1:8000');

// --- arguments -------------------------------------------------------------
const argv = process.argv.slice(2);
const has = f => argv.includes(f);
const arg = (f, d) => {
  const i = argv.indexOf(f);
  return i >= 0 && argv[i + 1] != null ? argv[i + 1] : d;
};
/** "64MB" / "1.5GB" / "134217728" -> bytes. Units because a limit written in bytes
 *  is a number nobody reads correctly on the command line. */
function bytes(s) {
  if (s == null) return null;
  const m = /^([\d.]+)\s*(k|m|g)?b?$/i.exec(String(s).trim());
  if (!m) throw new Error(`cannot read a size from "${s}"`);
  const mult = {k: 1e3, m: 1e6, g: 1e9}[(m[2] || '').toLowerCase()] || 1;
  return Math.round(+m[1] * mult);
}
/** "40ms" / "40" -> ms. */
const ms = s => s == null ? 0 : +String(s).replace(/ms$/i, '');

const FAKE = has('--fake');
const SOLO = argv[0] === 'solo';
const N = SOLO ? Math.max(1, +(argv[1] && /^\d+$/.test(argv[1]) ? argv[1] : 1)) : 1;
const SLOW = ms(arg('--slow', 0));
const BIND = bytes(arg('--bind', null));
const BUDGET = bytes(arg('--budget', null));
const LABEL = arg('--label', null) ||
  (SOLO ? 'sim' : (argv[0] && !argv[0].startsWith('--') ? argv[0] : 'sim'));

// The GPU and the weights are only loaded when they are actually needed. A fake
// bird must run where there is no device at all, which is the whole point of it.
let dev = null, weights = null, Layer = null, QWEN3_06B = null;
if (!FAKE) {
  const lib = await import('../kernels/lib.ts');
  const layerMod = await import('../kernels/layer.ts');
  const real = await import('../kernels/real_weights.ts');
  Layer = layerMod.Layer; QWEN3_06B = layerMod.QWEN3_06B;
  dev = await lib.getDevice();
  console.log('reading weights ...');
  weights = await real.realModel();
}

/** What this device claims about itself. A real bird measures these with
 *  /js/probe.mjs; a simulated one is told them, which is how a 64MB binding limit
 *  gets tested without a device that has one. */
function caps() {
  const c = {gpu: true, computeOk: true, cores: 8, vendor: FAKE ? 'sim' : 'deno'};
  if (BIND != null) c.maxStorageBufferBindingSize = BIND;
  if (BUDGET != null) c.budget = BUDGET;
  return c;
}

/** Sleep, to look slower than we are. Per LAYER, not per token: a slow device is
 *  slow in proportion to the work it holds, which is exactly the relationship the
 *  allocator is built around, and a flat per-token delay would not test it. */
const stall = n => n > 0 ? new Promise(r => setTimeout(r, n)) : null;

/**
 * Claim a place, build the layers we were given, and serve frames until the socket
 * closes -- re-building them if the coordinator reassigns us.
 */
async function bird(tag) {
  let j = null, pid = null;
  // The coordinator answers `wait` when a token is in flight: layers are assigned at
  // the next token boundary, which is milliseconds away. The peer id from that answer
  // MUST be sent back -- retrying without it is a second /join, and the coordinator
  // admits a second member with the same label that it then waits forever on.
  for (let attempt = 0; attempt < 40; attempt++) {
    j = await (await fetch(`${BASE}/join`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({peer_id: pid, label: tag, caps: caps()})})).json();
    pid = j.peer_id || pid;
    if (!j.wait) break;
    if (attempt === 0) console.log(`${tag}: ${j.reason || 'waiting for layers'}`);
    await new Promise(r => setTimeout(r, j.retry_ms || 500));
  }
  if (j.error) throw new Error(j.error);
  if (j.wait) throw new Error('never got a layer assignment');

  let meta = j;
  let layers = [];

  /** Build (or rebuild) the layers for the range we currently hold.
   *
   *  Built into a local array and published in ONE statement. Pushing into `layers`
   *  directly is a real hazard rather than a style point: every await yields to the
   *  event loop, so a frame arriving mid-build would find `layers` holding 1 of N
   *  layers, pass the non-empty check in the message handler, and compute over a
   *  partial chain -- the wrong layers at the right positions. */
  async function build() {
    layers = [];
    if (meta.start == null) return;
    const n = meta.end - meta.start + 1;
    if (FAKE) { layers = new Array(n).fill(null); return; }
    const next = [];
    for (let i = meta.start; i <= meta.end; i++) {
      next.push(await Layer.create(dev, weights.layers[i], QWEN3_06B));
    }
    layers = next;
  }
  await build();
  console.log(`${tag}: layers ${meta.start}-${meta.end}` +
              (FAKE ? ' (fake, no GPU)' : ' on the GPU') +
              (SLOW ? ` +${SLOW}ms/layer` : '') +
              (j.why ? `\n  ${j.why}` : ''));

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
  ws.on('error', e => console.error(`${tag} socket: ${e.message}`));
  ws.on('message', async (data, isBinary) => {
    if (!isBinary) {
      // A chain message can carry a DIFFERENT range: the coordinator re-splits when
      // a device joins or leaves, or when the timings say this one should hold more
      // or fewer. Rebuild rather than keep computing the layers we happen to have,
      // which would put the wrong layers at the right positions.
      const m = JSON.parse(data.toString());
      if (m.t === 'chain' && (m.start !== meta.start || m.end !== meta.end)) {
        const was = meta.start == null ? 'nothing' : `${meta.start}-${meta.end}`;
        meta = {...meta, start: m.start, end: m.end, slot: m.slot};
        await build();
        console.log(`${tag}: reassigned ${was} -> ` +
                    (m.start == null ? 'nothing' : `${m.start}-${m.end}`));
      }
      return;
    }
    const ab = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    const {data: h, meta: m} = unpack(ab);
    if (!layers.length) return;      // reassigned out of the chain mid-flight

    const t0 = performance.now();
    let flat;
    if (FAKE) {
      // Echo the hidden state back unchanged. The TEXT will be wrong, and that is
      // fine and explicit: a fake bird tests the allocation, the chain and the
      // rebalancer, never the arithmetic. Correctness is what a real simulated bird
      // and node/test/bird_ui.test.mjs are for.
      await stall(SLOW * layers.length);
      flat = h;
    } else {
      // A reset means the conversation restarted, so OUR shard of the K/V cache is
      // stale too. Each layer owns its own, so each one clears it.
      if (m.reset) for (const L of layers) L.reset();
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
                                 L.inputBuffer(), 0, m.seq * meta.hidden * 4);
          // No hidden argument: consume what the copy just put in our own buffer.
          if (m.seq === 1) L.encode(undefined, enc);
          else L.encodePrefill(m.seq, undefined, enc);
        }
      }
      dev.queue.submit([enc.finish()]);
      flat = await layers[layers.length - 1].readOutput(m.seq);
      await stall(SLOW * layers.length);
    }
    const took = +(performance.now() - t0).toFixed(1);

    ws.send(pack(flat, {seq: m.seq, hidden: meta.hidden, offset: m.offset}));
    ws.send(JSON.stringify({t: 'stats', ms: took}));
  });
  return j;
}

if (!SOLO) {
  await bird(LABEL);
} else {
  // N devices in one process. Sequential on purpose: the allocator keeps join order,
  // so racing would make the assignment depend on which fetch won.
  for (let i = 0; i < N; i++) await bird(`${LABEL}${i}`);
  console.log(`solo: ${N} device${N === 1 ? '' : 's'} covering the birds' layers`);
}
