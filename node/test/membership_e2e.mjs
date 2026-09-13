// Dynamic membership against a LIVE coordinator, with fake birds.
//
// What this adds over node/test/flock.test.mjs: that one drives the Flock object
// directly, this one goes through the real HTTP and websocket surface -- /join with
// caps, the chain message, /leave, /evict, and the token-boundary rebalance inside
// /chat. It is the test that would catch a Flock that is correct and a server that
// wires it up wrong.
//
// THE BIRDS ARE FAKE, and the trade is explicit. A fake bird echoes the hidden state
// back unchanged, so the TEXT is meaningless -- what is under test is the allocation,
// the chain and the membership transitions, and a fake bird can report arbitrary
// capability and speed, which is how "a 64MB binding limit never gets an oversized
// layer" gets tested without owning such a device. Correctness of the TEXT is a
// separate claim, proven by a real simulated bird:
//
//   deno run --unstable-webgpu --allow-all sim_bird.mjs solo 3
//   curl -X POST .../chat -d '{"prompt":"Capital of France?","reset":true}'
//
// Needs a coordinator, and no GPU of its own:
//   PORT=8123 npm start &
//   FLOCK_URL=http://127.0.0.1:8123 node test/membership_e2e.mjs
import WebSocket from 'ws';
import {pack, unpack} from '../../web/js/wire.mjs';

const BASE = process.env.FLOCK_URL || 'http://127.0.0.1:8000';
let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};
const sleep = ms => new Promise(r => setTimeout(r, ms));
const status = async () => (await fetch(`${BASE}/status`)).json();
const post = async (p, body) => (await fetch(`${BASE}/${p}`, {method: 'POST',
  headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})).json();

/**
 * A fake bird: joins, serves frames by echoing them, and can lie about its limits
 * and its speed.
 *
 * `slowPerLayer` is per LAYER rather than per token on purpose -- a slow device is
 * slow in proportion to the work it holds, which is exactly the relationship the
 * allocator is built around, and a flat per-token delay would not test it.
 */
async function fakeBird(label, {caps = null, slowPerLayer = 0} = {}) {
  // The peer id from a `wait` answer has to be sent back: retrying without it is a
  // second /join, which admits a second member with the same label.
  let j = null, pid = null;
  for (let i = 0; i < 60; i++) {
    j = await post('join', {peer_id: pid, label, caps});
    pid = j.peer_id || pid;
    if (!j.wait) break;
    await sleep(j.retry_ms || 300);
  }
  if (j.error) return {error: j.error, label};
  const b = {label, peerId: j.peer_id, start: j.start, end: j.end, hidden: j.hidden,
             frames: 0, reassigns: [], why: j.why, closed: false};
  const ws = new WebSocket(BASE.replace(/^http/, 'ws') + '/ws');
  b.ws = ws;
  await new Promise((res, rej) => { ws.on('open', res); ws.on('error', rej); });
  ws.send(JSON.stringify({peer_id: b.peerId, label}));
  ws.send(JSON.stringify({t: 'ping'}));
  b.beat = setInterval(() => { try { ws.send(JSON.stringify({t: 'ping'})); } catch {} },
                       3000);
  ws.on('close', () => { b.closed = true; clearInterval(b.beat); });
  ws.on('message', async (data, isBinary) => {
    if (!isBinary) {
      const m = JSON.parse(data.toString());
      if (m.t === 'chain' && (m.start !== b.start || m.end !== b.end)) {
        b.reassigns.push(`${b.start}-${b.end} -> ${m.start}-${m.end}`);
        b.start = m.start; b.end = m.end;
      }
      return;
    }
    const ab = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    const {data: h, meta: m} = unpack(ab);
    if (b.start == null) return;                 // reassigned out of the chain
    const n = b.end - b.start + 1;
    const t0 = Date.now();
    if (slowPerLayer) await sleep(slowPerLayer * n);
    b.frames++;
    ws.send(pack(h, {seq: m.seq, hidden: b.hidden, offset: m.offset}));
    ws.send(JSON.stringify({t: 'stats', ms: Date.now() - t0 || 1}));
  });
  return b;
}
/** Close a bird's socket and make sure the coordinator has forgotten it, so one
 *  scenario's devices cannot leak into the next one's allocation. */
async function bye(b) {
  try { clearInterval(b.beat); b.ws?.terminate?.(); b.ws?.close(); } catch {}
  if (b.peerId) { try { await post('evict', {peer_id: b.peerId}); } catch {} }
}
/** Belt and braces between scenarios: evict whatever /status still lists. */
async function clear() {
  for (const b of (await status()).birds) await post('evict', {peer_id: b.peer_id});
  await sleep(300);
}

/** Run one turn and return {text, stop, events}. */
async function turn(prompt, max_tokens = 8, reset = true) {
  const res = await fetch(`${BASE}/chat`, {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt, max_tokens, reset})});
  const events = [];
  const rd = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const {done, value} = await rd.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    const parts = buf.split('\n\n');
    buf = parts.pop();
    for (const p of parts) if (p.startsWith('data: ')) events.push(JSON.parse(p.slice(6)));
  }
  const d = events.find(e => e.type === 'done') || {};
  return {text: d.text, stop: d.stop, events,
          error: events.find(e => e.type === 'error')?.text};
}

const up = await fetch(`${BASE}/status`).catch(() => null);
if (!up?.ok) {
  console.error(`no coordinator at ${BASE} — start one with PORT=8123 npm start`);
  process.exit(2);
}
const s0 = await status();
console.log(`coordinator: ${s0.model}, birds cover ${s0.allocation.bird_layers} ` +
            `(${s0.allocation.layer_mb.join('/')} MB per layer)\n`);
const LAYERS = s0.allocation.layers;

// Whatever was left over from an earlier run.
for (const b of s0.birds) await post('evict', {peer_id: b.peer_id});
await sleep(300);

// =========================================================================
console.log('N devices join over HTTP and the layers are covered, whatever N is:');
{
  const birds = [];
  for (let n = 1; n <= Math.min(LAYERS, 4); n++) {
    birds.push(await fakeBird(`d${n}`));
    await sleep(400);
    const s = await status();
    ok(`${n} device${n === 1 ? '' : 's'}: no refusal, every layer covered`,
       s.ready && s.birds.length === n && s.missing.length === 0,
       s.birds.map(b => `${b.label}:${b.start}-${b.end}`).join(' '));
  }
  // The behaviour that used to be impossible: the (LAYERS+1)th device is admitted as
  // a member even though there is no layer for it, and the failure says so.
  const extra = await fakeBird('one-too-many');
  await sleep(400);
  const s = await status();
  if (birds.length >= LAYERS) {
    // It is turned away with a reason -- and, crucially, WITHOUT taking the working
    // flock down with it. Admitting it first and reporting the failure afterwards is
    // what the earlier version did, and it left the flock permanently unsatisfiable.
    ok('more devices than layers: the joiner is refused with a reason',
       !!extra.error && /\d+ bird layers|hold nothing|cannot be fitted/
         .test(extra.error), (extra.error || 'admitted').slice(0, 100));
    ok('  and the flock it tried to join still works',
       s.ready && s.birds.length === birds.length && !s.allocation.infeasible,
       `ready=${s.ready}, ${s.birds.length} devices`);
  } else {
    ok('an extra device just gets a share', s.ready, s.birds.length + ' devices');
  }
  await bye(extra);
  for (const b of birds) await bye(b);
  await clear();
}

// =========================================================================
console.log('\na device with a small binding limit never gets an oversized layer:');
{
  // The bird layers' largest tensor is ~3MB for Qwen3-0.6B, so a limit BELOW that
  // excludes the device from every layer -- which must be reported at /join, before
  // the device downloads anything.
  const tiny = await fakeBird('tiny-buffer',
    {caps: {maxStorageBufferBindingSize: 1e6, gpu: true, computeOk: true}});
  ok('a device that can hold no layer is refused with a reason, at /join',
     !!tiny.error && /cannot hold/.test(tiny.error),
     (tiny.error || 'joined anyway').slice(0, 130));
  ok('  and the reason names the tensor and both numbers',
     /\.weight/.test(tiny.error || '') && /MB/.test(tiny.error || ''));
  ok('  and points at /check, which is where the numbers came from',
     /\/check/.test(tiny.error || ''));
  const s = await status();
  ok('  it is NOT left a member poisoning the flock', s.birds.length === 0,
     `${s.birds.length} members`);
  // A capable device arrives; the failure must clear without a restart.
  const good = await fakeBird('roomy',
    {caps: {maxStorageBufferBindingSize: 4 * 1024 ** 3, gpu: true, computeOk: true}});
  await sleep(400);
  const s2 = await status();
  ok('a capable device clears the failure with no restart',
     s2.ready && !s2.allocation.infeasible,
     s2.birds.map(b => `${b.label}:${b.start}-${b.end}`).join(' '));
  ok('  and /status reports the limit it was told',
     s2.birds[0].max_binding_mb === 4295, String(s2.birds[0].max_binding_mb));
  ok('  the flock reports the predicted per-token cost',
     s2.allocation.makespan_ms > 0, `${s2.allocation.makespan_ms}ms`);
  await bye(good);
  await clear();
}

// =========================================================================
console.log('\na deliberately slow bird ends up with fewer layers:');
if (LAYERS >= 3) {
  const fast = await fakeBird('fast');
  const slow = await fakeBird('slow', {slowPerLayer: 45});
  await sleep(500);
  const before = await status();
  const b0 = before.birds.find(b => b.label === 'slow');
  console.log(`  start: ${before.birds.map(b => `${b.label} ${b.start}-${b.end}`)
    .join(', ')}`);

  // Run turns until the rebalancer has seen enough and the cooldown has expired. The
  // cooldown is 20s, so this takes a few turns' worth of wall time -- which is the
  // honest cost of a policy that refuses to thrash.
  let moved = false;
  for (let i = 0; i < 8 && !moved; i++) {
    await turn('hello', 6, true);
    const s = await status();
    const sb = s.birds.find(b => b.label === 'slow');
    if (sb && (sb.start !== b0.start || sb.end !== b0.end)) moved = true;
    console.log(`  turn ${i + 1}: ${s.birds.map(b =>
      `${b.label} ${b.start}-${b.end} @${b.rate_mb_per_ms ?? '?'}MB/ms`).join(', ')}` +
      `  [${s.allocation.decision.reason}]`);
    if (!moved) await sleep(3000);
  }
  const after = await status();
  const f = after.birds.find(b => b.label === 'fast');
  const sl = after.birds.find(b => b.label === 'slow');
  ok('the slow bird holds fewer layers than the fast one',
     sl.n_layers < f.n_layers, `fast ${f.n_layers}, slow ${sl.n_layers}`);
  ok('  and /status says why, in bytes and MB/ms', /% of the bird bytes/.test(sl.why),
     sl.why);
  // The rate the DECISION was made on is quoted in `why`; the live counters are
  // deliberately back at zero, because a device whose range just changed is
  // re-learned rather than trusted on samples that described different work.
  ok('  quoting the measured rate the decision was made on',
     /at 0\.\d+MB\/ms/.test(sl.why), sl.why);
  ok('  and its counters are reset, so the new share is re-measured',
     sl.rate_samples < after.allocation.min_samples,
     `${sl.rate_samples} samples since the move`);
  ok('  the weighted split beats the even one it replaced',
     after.allocation.makespan_ms <= after.allocation.even_split_ms,
     `${after.allocation.makespan_ms}ms vs even ${after.allocation.even_split_ms}ms`);
  await bye(fast); await bye(slow);
  await clear();
} else {
  console.log(`  (skipped: only ${LAYERS} bird layers, need 3 to show a shift)`);
}

// =========================================================================
console.log('\njoining MID-GENERATION: the token finishes, then layers move:');
if (LAYERS >= 2) {
  const a = await fakeBird('incumbent', {slowPerLayer: 60});
  await sleep(400);
  ok('one device holds everything', (await status()).ready);

  // Start a long turn, then join a device while it is generating.
  const running = turn('count to twenty', 40, true);
  await sleep(900);
  const s = await status();
  ok('the turn is in flight', s.busy);
  const joiner = await fakeBird('latecomer');
  ok('the joining device is told to wait rather than refused',
     !joiner.error, joiner.error || 'admitted');
  const r = await running;
  ok('the turn ENDED, it did not produce wrong output silently',
     r.stop === 'rebalanced' || r.stop === 'eos' || r.stop === 'length', r.stop);
  if (r.stop === 'rebalanced') {
    ok('  and it said the context was dropped, in words',
       /context was dropped/.test(r.events.find(e => e.type === 'done')?.note || ''),
       r.events.find(e => e.type === 'done')?.note);
    ok('  naming which devices moved',
       (r.events.find(e => e.type === 'rebalanced')?.moved || []).length > 0);
    const s2 = await status();
    ok('  the context really is zero, not stale', s2.cached_tokens === 0,
       String(s2.cached_tokens));
    ok('  and last_rebalance explains it for the UI',
       !!s2.last_rebalance?.dropped_context, JSON.stringify(s2.last_rebalance));
  }
  await sleep(600);
  const s3 = await status();
  ok('both devices now hold layers and the flock is covered',
     s3.ready && s3.birds.length === 2 && s3.missing.length === 0,
     s3.birds.map(b => `${b.label}:${b.start}-${b.end}`).join(' '));
  ok('the incumbent was TOLD its new range, not left guessing',
     (a.reassigns || []).length > 0, (a.reassigns || []).join('; ') || 'never told');

  // And the flock works again straight afterwards.
  const next = await turn('hello', 4, true);
  ok('a turn after the rebalance completes normally', !next.error,
     next.error || next.stop);
  await bye(a); await bye(joiner);
  await clear();
}

// =========================================================================
console.log('\nleaving MID-GENERATION: the turn fails loudly, then recovers:');
if (LAYERS >= 2) {
  const keep = await fakeBird('stays');
  const go = await fakeBird('goes');
  await sleep(500);
  ok('two devices cover the layers', (await status()).ready);

  const running = turn('count to twenty', 40, true);
  await sleep(700);
  // Pull the plug the way a phone does: the socket dies mid-lap.
  clearInterval(go.beat);
  go.ws.terminate();
  const r = await running;
  ok('the turn did NOT quietly return short text', !!r.error || r.stop !== 'eos',
     r.error || r.stop);
  ok('  and the reason names the device and the layers it was holding',
     /dropped its connection|stopped responding|no device|not holding/
       .test(r.error || '') && /26-27/.test(r.error || ''),
     (r.error || '(no error)').slice(0, 100));

  // The sweeper removes it after the liveness grace period; /leave is the fast path.
  await post('leave', {peer_id: go.peerId});
  await sleep(600);
  const s = await status();
  ok('after the departure the survivor covers every layer',
     s.ready && s.missing.length === 0,
     s.birds.map(b => `${b.label}:${b.start}-${b.end}`).join(' '));
  const next = await turn('hello', 4, true);
  ok('and the flock generates again', !next.error, next.error || next.stop);
  ok('  the survivor was told it holds more now', (keep.reassigns || []).length > 0,
     (keep.reassigns || []).join('; ') || 'never told');
  await bye(keep);
  await clear();
}

// =========================================================================
console.log('\nno devices at all -> a message naming the layers, not a crash:');
{
  const s = await status();
  ok('not ready', !s.ready);
  ok('  and it names the uncovered layers', s.missing.length > 0, s.missing.join(','));
  const r = await turn('hello', 4, true);
  ok('  /chat refuses with the layers in the message',
     /waiting for devices|cover layer/.test(r.error || ''),
     (r.error || '(none)').slice(0, 90));
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
