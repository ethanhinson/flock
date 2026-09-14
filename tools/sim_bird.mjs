// A bird without a browser.
//
// Runs the same layers a phone would, with the same WGSL kernels, and speaks the
// same protocol -- join, hello, chain, ready, frames, nack, stats, session, pause
// -- so the whole chain can be exercised on one machine with no browser, over
// ssh, or in a test.
//
// RUN IT WITH DENO, NOT NODE -- it needs a GPU for the same reason the
// coordinator does, and Node has no WebGPU:
//
//   npm run bird                          one device
//   npm run solo                          one device, all the bird layers
//   npm run solo -- 3                     three devices in one process
//   npm run fake -- --restream 3000       no GPU; pretend a re-stream takes 3s
//   FLOCK_URL=https://host:port ...       a coordinator elsewhere
//
// WHY IT CAN LIE ABOUT ITSELF. The allocator weights layers by each device's real
// limits and measured speed, and the readiness handshake exists because a moved
// device holds nothing while it re-streams. None of that can be tested without
// devices that differ and re-streams that take time -- which would mean owning a
// drawer of phones. So a simulated bird can report arbitrary capability, speed and
// re-stream time:
//
//   --bind 64MB            maxStorageBufferBindingSize to report at /join
//   --budget 300MB         how many weight bytes it claims it will hold
//   --slow 40ms            extra delay per layer per token, to look like a phone
//   --restream 3000        ms it holds NO layers after every (re)assignment,
//                          the way a phone does while its weights stream
//   --label iPhone         what /status calls it
//   --fake                 do not touch the GPU: echo the hidden state back
//   --no-rtc               do not answer the coordinator's WebRTC offer
//   --leave-on-unready     the behaviour bird.html had before the readiness
//                          handshake: a frame arriving while it holds no layers
//                          makes it LEAVE the flock. Kept only so the churn test
//                          can demonstrate the cascade that handshake prevents.
//   --session-file f.json  where to keep the session the coordinator issued, so a
//                          sim that is killed and started again RE-ATTACHES the
//                          way a refreshed page does (bird.html keeps it in
//                          sessionStorage; a process has to keep it in a file)
//
// IT TAKES ORDERS ON STDIN, one per line, because a test has to make a device do
// what a phone does without a phone:
//
//   pause      what a tab does on visibilitychange -> hidden: tell the
//              coordinator, then go quiet (the socket is dropped, the way iOS
//              suspends it). The layers stay built, as a phone's may or may not.
//   resume     what a tab does when visible again: re-attach with the session,
//              reconnect, rebuild if the range changed, confirm.
//   leave      POST /leave and exit.
//
// --fake is what makes the membership tests runnable anywhere: a fake bird exercises
// /join, the chain, the timings, readiness and the rebalancer without a device and
// without 67MB of weights. A REAL simulated bird (no --fake) still computes real
// layers with real kernels, which is what proves the text stays correct.
//
// WHAT IT PRINTS IS PART OF ITS INTERFACE. The e2e tests assert on what a bird
// actually holds and does, not on what /status says -- the cascade the churn test
// guards against was invisible to every /status check -- so each state change is
// one line:
//
//   <tag> joined 4-15            <tag> loading 4-9         <tag> ready 4-9
//   <tag> reassigned 4-15 -> 4-9 <tag> nack seq=3 (not-ready: still streaming)
//   <tag> link webrtc            <tag> left: <reason>
//   <tag> session <token>        <tag> reattached 4-9      <tag> reconnected
//   <tag> replay 64@0            <tag> paused              <tag> resumed 4-9
//
// WEIGHTS COME FROM THE SAME PLACE A REAL BIRD'S DO: the GGUF file, by byte
// range, over the network. What differs from bird.html is only HOW they reach the
// GPU -- this reads the cached whole-model file through real_weights.ts, while a
// phone streams its range so the bytes never sit in the JS heap. Both end up
// calling the same Layer with the same numbers.
import WebSocket from 'ws';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {Buffer} from 'node:buffer';
import {createInterface} from 'node:readline';
import {pack, unpack} from '../web/js/wire.mjs';
import {discoverFlock} from './mdns-discover.mjs';

// Where the coordinator is. Three sources, in priority order:
//
//   1. FLOCK_URL env var — explicit, used by tests and when mDNS is not wanted.
//   2. mDNS discovery — query _flock._tcp.local on the LAN and use whatever
//      coordinator answers. This is the default for interactive use: run
//      `npm start` on one machine, `npm run bird` on another, and the bird
//      finds it without being told an address.
//   3. Fallback to localhost — for a solo run on one machine.
//
// The coordinator serves https when .certs/ exists (Chrome hides WebGPU outside
// a secure context), so the fallback checks for the cert. The cert is self-signed,
// so verification is off for this client only.
let BASE;
if (process.env.FLOCK_URL) {
  BASE = process.env.FLOCK_URL;
} else {
  const discovered = await discoverFlock(3000);
  if (discovered) {
    BASE = discovered;
    console.log('(discovered coordinator via mDNS: ' + discovered + ')');
  } else {
    BASE = existsSync(new URL('../.certs/cert.pem', import.meta.url))
      ? 'https://127.0.0.1:8000' : 'http://127.0.0.1:8000';
    console.log('(no mDNS response; falling back to ' + BASE + ')');
  }
}
const WS_URL = BASE.replace(/^http/, 'ws') + '/ws';
// Only pass options when there is one to pass: under Deno's node compatibility
// layer, any options object makes `ws` print a createConnection warning.
const wsOpts = BASE.startsWith('https') ? {rejectUnauthorized: false} : undefined;


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
const RESTREAM = ms(arg('--restream', 0));
const BIND = bytes(arg('--bind', null));
const BUDGET = bytes(arg('--budget', null));
const NO_RTC = has('--no-rtc');
const LEAVE_ON_UNREADY = has('--leave-on-unready');
const SESSION_FILE = arg('--session-file', null);
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
// node-datachannel is the same package the coordinator answers with, loaded
// lazily so a --no-rtc or --fake run in a stripped environment does not need it.
let ndc = null;
if (!NO_RTC) {
  try { ndc = (await import('node-datachannel')).default; }
  catch (e) { console.log(`(no node-datachannel: ${e.message}; websocket only)`); }
}

/** What this device claims about itself. A real bird measures these with
 *  /js/probe.mjs; a simulated one is told them, which is how a 64MB binding limit
 *  gets tested without a device that has one. The same numbers every time, on
 *  purpose: the session is bound to them. */
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
const range = m => m.start == null ? 'nothing' : `${m.start}-${m.end}`;
const postJson = (p, body) => fetch(`${BASE}/${p}`, {method: 'POST',
  headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});

/** Where this bird keeps its session between runs, if anywhere. */
const sessionFile = i => SESSION_FILE ? (SOLO && N > 1 ? `${SESSION_FILE}.${i}` : SESSION_FILE) : null;
const readSession = f => {
  try { return f && existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')) : null; } catch { return null; }
};

const birds = [];

/**
 * Claim a place, build the layers we were given, confirm them, and serve frames
 * until told to leave -- re-building and re-confirming when reassigned, and
 * re-attaching with the session when the socket is lost or the tab "comes back".
 */
async function bird(tag, file = null) {
  const say = s => console.log(`${tag} ${s}`);
  let pid = null, session = readSession(file)?.session || null;
  let meta = null;
  let layers = [];
  let loading = false, pendingChain = null;
  let ws = null, pc = null, chan = null, beat = null;
  let paused = false, leaving = false, everOpened = false;

  /**
   * Ask for a place. With a session, the coordinator RE-ATTACHES us to the
   * membership we had; without one (or with one it refuses) it is a fresh join.
   * The coordinator answers `wait` when a token is in flight: layers are assigned at
   * the end of the turn. The peer id from that answer MUST be sent back --
   * retrying without it is a second /join, and the coordinator admits a second
   * member with the same label that it then waits forever on.
   */
  async function join() {
    let j = null;
    for (let attempt = 0; attempt < 300; attempt++) {
      j = await (await postJson('join', {peer_id: pid, session, label: tag, caps: caps()})).json();
      pid = j.peer_id || pid;
      if (j.session && j.session !== session) {
        session = j.session;
        say(`session ${session}`);
      }
      if (file && session) writeFileSync(file, JSON.stringify({peer_id: pid, session}));
      if (!j.wait) break;
      if (attempt === 0) say(`waiting: ${j.reason || 'no layers yet'}`);
      await new Promise(r => setTimeout(r, j.retry_ms || 500));
    }
    if (j.error) throw new Error(j.error);
    if (j.wait) throw new Error('never got a layer assignment');
    return j;
  }

  const send = o => { try { ws?.send(JSON.stringify(o)); } catch {} };

  /** Tell the coordinator we hold what we were assigned. Nothing may be sent to
   *  us until this arrives; that is the whole handshake. */
  function ready() {
    if (meta.start == null || !layers.length) return;
    send({t: 'ready', start: meta.start, end: meta.end});
    say(`ready ${range(meta)}`);
  }

  /** Build (or rebuild) the layers for the range we currently hold.
   *
   *  Built into a local array and published in ONE statement. Pushing into `layers`
   *  directly is a real hazard rather than a style point: every await yields to the
   *  event loop, so a frame arriving mid-build would find `layers` holding 1 of N
   *  layers, pass the non-empty check in the message handler, and compute over a
   *  partial chain -- the wrong layers at the right positions.
   *
   *  A chain message arriving mid-build is parked and applied afterwards, always
   *  the latest, since ranges supersede. */
  async function build() {
    loading = true;
    layers = [];
    try {
      if (meta.start != null) {
        say(`loading ${range(meta)}`);
        // A phone streams its weights here, and holds nothing meanwhile.
        // --restream models that window; without it a fake bird is ready in
        // microseconds and the cascade this protocol prevents cannot be
        // reproduced.
        await stall(RESTREAM);
        const n = meta.end - meta.start + 1;
        if (FAKE) {
          layers = new Array(n).fill(null);
        } else {
          const next = [];
          for (let i = meta.start; i <= meta.end; i++) {
            next.push(await Layer.create(dev, weights.layers[i], QWEN3_06B));
          }
          layers = next;
        }
      }
    } finally {
      loading = false;
    }
    if (pendingChain) {
      const p = pendingChain; pendingChain = null;
      if (p.start !== meta.start || p.end !== meta.end) return reassign(p);
    }
    ready();
  }

  async function reassign(m) {
    if (loading) { pendingChain = m; return; }
    say(`reassigned ${range(meta)} -> ${range(m)}`);
    meta = {...meta, start: m.start, end: m.end, slot: m.slot};
    await build();
  }

  /** Refuse a frame, saying why, so the lap waiting on it is told now rather
   *  than at its timeout. Never a reason to leave. */
  function nack(m, reason, detail) {
    send({t: 'nack', reason, detail, seq: m?.seq, offset: m?.offset,
          start: meta.start, end: meta.end});
    say(`nack seq=${m?.seq} (${reason}${detail ? ': ' + detail : ''})`);
  }

  async function leave(why) {
    leaving = true;
    say(`left: ${why}`);
    try { await postJson('leave', {peer_id: pid, why}); } catch {}
    try { ws?.close(); } catch {}
  }

  /** One frame in, one frame out on the same link it arrived on. */
  async function onFrame(buf, reply) {
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const {data: h, meta: m} = unpack(ab);
    if (loading || !layers.length) {
      if (LEAVE_ON_UNREADY) {
        // What bird.html did before the handshake: treat a frame it could not
        // serve as a fatal problem and hand its layers back. That leave is a
        // membership change, which reallocates, which catches the next device
        // mid-stream -- the cascade.
        return leave(`frame arrived while holding no layers (seq=${m.seq})`);
      }
      return nack(m, 'not-ready', loading ? 'still streaming' : 'holding no layers');
    }
    if (m.replay) say(`replay ${m.seq}@${m.offset}${m.reset ? ' (reset)' : ''}`);
    const t0 = performance.now();
    let flat;
    try {
      if (FAKE) {
        // Echo the hidden state back unchanged. The TEXT will be wrong, and that is
        // fine and explicit: a fake bird tests the allocation, the chain, readiness
        // and the rebalancer, never the arithmetic. Correctness is what a real
        // simulated bird and test/e2e/bird_ui.test.mjs are for.
        await stall(SLOW * layers.length);
        flat = h;
      } else {
        // A reset means the conversation restarted (or is being replayed from the
        // start), so OUR shard of the K/V cache is stale too. Each layer owns its
        // own, so each one clears it.
        if (m.reset) for (const L of layers) L.reset();
        // The frame's position must be where our cache ends. If it is not, the
        // coordinator and this device disagree about the conversation, and
        // computing anyway would append keys at the wrong positions -- text that
        // is wrong without looking wrong. Refuse it; the coordinator rewinds and
        // replays.
        if (layers[0].nKeys !== m.offset) {
          return nack(m, 'desync', `cache holds ${layers[0].nKeys} positions, frame is at ${m.offset}`);
        }
        // Chain the layers on the GPU, the way kernels/model.ts chains its 28: layer i
        // reads what layer i-1 wrote via a device-to-device copy, everything shares
        // ONE command buffer, and only the shard's final output comes back to the
        // host. A readback per layer instead would cost ~24 ms each -- measured 20.8x
        // for this exact difference in bench_layer.ts.
        const enc = dev.createCommandEncoder();
        for (let i = 0; i < layers.length; i++) {
          const L = layers[i];
          if (i === 0) {
            if (m.seq === 1) L.encode(h, enc);
            else L.encodePrefill(m.seq, h, enc);
          } else {
            enc.copyBufferToBuffer(layers[i - 1].outputBuffer(), 0,
                                   L.inputBuffer(), 0, m.seq * meta.hidden * 4);
            if (m.seq === 1) L.encode(undefined, enc);
            else L.encodePrefill(m.seq, undefined, enc);
          }
        }
        dev.queue.submit([enc.finish()]);
        flat = await layers[layers.length - 1].readOutput(m.seq);
        await stall(SLOW * layers.length);
      }
    } catch (e) {
      return nack(m, 'failed', String(e.message || e));
    }
    const took = +(performance.now() - t0).toFixed(1);
    // The flags travel with the frame: a successor's cache has to see the same
    // reset this one did.
    reply(pack(flat, {seq: m.seq, hidden: meta.hidden, offset: m.offset,
                      reset: m.reset, replay: m.replay}));
    // A replay frame is journaled history, not conversation output: its timing
    // would describe a 64-token prefill, not a token, and would skew the rate the
    // allocator plans on.
    if (!m.replay) send({t: 'stats', ms: took});
  }

  // --- WebRTC: answer the coordinator's offer, so frames skip the socket -----
  function onSignal(d) {
    if (!ndc) return;
    if (d.kind === 'offer') {
      try { pc?.close(); } catch {}
      pc = new ndc.PeerConnection(tag, {iceServers: []});
      pc.onLocalDescription((sdp, type) =>
        send({t: 'signal', data: {kind: type, sdp: {sdp, type}}}));
      pc.onLocalCandidate((candidate, mid) =>
        send({t: 'signal', data: {kind: 'ice', candidate, mid}}));
      pc.onDataChannel(dc => {
        chan = dc;
        say('link webrtc');
        dc.onMessage(msg => {
          if (typeof msg === 'string') return;
          const buf = Buffer.isBuffer(msg) ? msg : Buffer.from(msg);
          onFrame(buf, frame => dc.sendMessageBinary(Buffer.from(frame)));
        });
        dc.onClosed(() => { if (chan === dc) chan = null; });
      });
      pc.setRemoteDescription(d.sdp.sdp, d.sdp.type);
    } else if (d.kind === 'ice' && pc) {
      try { pc.addRemoteCandidate(d.candidate, d.mid || '0'); } catch {}
    }
  }

  // --- the websocket: hello, chain, signaling, and frames when there is no channel
  //
  // Connected BEFORE building, as bird.html does: a device that joins during the
  // build reassigns us, and with no socket open that notice would be lost. It also
  // means the coordinator can see us as a live, loading member -- which is what
  // lets a turn WAIT for us instead of reporting our layers uncovered.
  //
  // Reconnects when the socket drops -- a coordinator restart, a network blip --
  // unless we paused or left on purpose. The hello carries the session, which the
  // coordinator now requires on the socket: a peer id alone is public in /status.
  function connect() {
    if (leaving || paused) return;
    const sock = new WebSocket(WS_URL, wsOpts);
    ws = sock;
    sock.on('open', () => {
      if (everOpened) say('reconnected');
      everOpened = true;
      send({peer_id: pid, session, label: tag});
      // Heartbeat, same as a real bird: liveness is judged on lastSeen, so an idle
      // sim silently ages out of the flock after 40s without this.
      send({t: 'ping'});
      clearInterval(beat);
      beat = setInterval(() => send({t: 'ping'}), 4000);
      // A build that finished before the socket opened (no --restream) confirmed
      // into the void; say it again now that someone is listening.
      if (!loading && layers.length) ready();
    });
    sock.on('close', () => {
      clearInterval(beat);
      try { pc?.close(); } catch {}
      pc = null; chan = null;
      if (ws === sock && !leaving && !paused) setTimeout(connect, 1000);
    });
    sock.on('error', e => { if (!/ECONNREFUSED/.test(e.message)) console.error(`${tag} socket: ${e.message}`); });
    sock.on('message', async (data, isBinary) => {
      if (!isBinary) {
        const m = JSON.parse(data.toString());
        if (m.error) {
          say(`coordinator: ${m.error}`);
          // The coordinator does not know us (it restarted without our session, or
          // dropped us). Claim a place again -- with the session, in case it does
          // recognise that -- and rebuild if the range changed.
          if (/unknown peer/i.test(m.error) && !leaving) {
            try { sock.close(); } catch {}
            await rejoin('the coordinator forgot us');
          }
          return;
        }
        if (m.t === 'signal' && !m.from) return onSignal(m.data);
        // A chain message can carry a DIFFERENT range: the coordinator re-splits when
        // a device joins or leaves, or when the timings say this one should hold more
        // or fewer. Rebuild rather than keep computing the layers we happen to have,
        // which would put the wrong layers at the right positions.
        if (m.t === 'chain' && (m.start !== meta.start || m.end !== meta.end)) {
          await reassign(m);
        }
        return;
      }
      onFrame(data, frame => sock.send(Buffer.from(frame)));
    });
  }

  /** /join again, keeping our built layers if the range is unchanged. */
  async function rejoin(why) {
    const j = await join();
    const same = meta && j.start === meta.start && j.end === meta.end;
    meta = {...meta, ...j};
    say(`${j.reattached ? 'reattached' : 'joined'} ${range(meta)}  (${why})`);
    if (!same || !layers.length) await build();
    // The socket is what carries the confirmation; connect (or reconnect) now.
    if (!ws || ws.readyState > 1) connect();
    else ready();
    return j;
  }

  // --- orders from stdin -------------------------------------------------------
  async function pause() {
    if (paused) return;
    paused = true;
    // Say so on the socket AND with the beacon a page sends from pagehide: a
    // backgrounded phone's socket may already be stalling.
    send({t: 'pause', why: 'simulated background'});
    try { await postJson('pause', {peer_id: pid, session, why: 'simulated background'}); } catch {}
    clearInterval(beat);
    try { ws?.close(); } catch {}
    ws = null;
    say('paused');
  }
  async function resume() {
    if (!paused) return;
    paused = false;
    const j = await join();
    const same = j.start === meta.start && j.end === meta.end;
    meta = {...meta, ...j};
    say(`resumed ${range(meta)}${j.reattached ? '' : '  (fresh join)'}`);
    // Our layers may or may not have survived the background; a phone's often do
    // not. Rebuild if the range changed, otherwise keep them and just confirm.
    if (!same || !layers.length) await build();
    connect();
  }
  const b = {tag, pause, resume, leave: () => leave('told to on stdin'), get paused() { return paused; }};
  birds.push(b);

  // --- first join ----------------------------------------------------------------
  const j = await join();
  meta = j;
  say(`${j.reattached ? 'reattached' : 'joined'} ${range(meta)}` + (j.why ? `  (${j.why})` : ''));
  connect();
  await build();
  return j;
}

// Orders on stdin apply to every bird in this process.
const rl = createInterface({input: process.stdin});
rl.on('line', line => {
  const cmd = line.trim();
  if (!cmd) return;
  for (const b of birds) {
    if (cmd === 'pause') b.pause();
    else if (cmd === 'resume') b.resume();
    else if (cmd === 'leave') b.leave().then(() => process.exit(0));
    else console.log(`${b.tag} ?? ${cmd}`);
  }
});

if (!SOLO) {
  await bird(LABEL, sessionFile(0));
} else {
  // N devices in one process. Sequential on purpose: the allocator keeps join order,
  // so racing would make the assignment depend on which fetch won.
  for (let i = 0; i < N; i++) await bird(`${LABEL}${i}`, sessionFile(i));
  console.log(`solo: ${N} device${N === 1 ? '' : 's'} covering the birds' layers`);
}
