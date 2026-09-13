// THE CHURN TEST: devices join while a conversation is running, and nothing
// cascades.
//
// The failure this reproduces, diagnosed from a live log with real phones:
//
//   1. a device joined; the coordinator reassigned every bird and IMMEDIATELY sent
//      the next frame. A moved bird re-streams its weights for 3-18s and holds NO
//      layers meanwhile, so the frame arrived at a device holding nothing;
//   2. the bird treated that as fatal and LEFT the flock -- a membership change,
//      which reallocated everyone again, which caught the next bird mid-stream;
//   3. every one of those changes dropped the conversation context. The log read
//      joined -> left -> dropped -> joined, in a loop, and chat never completed a
//      turn.
//
// /status was correct throughout: the server's split was always consistent. So
// this test does NOT trust /status. It runs real tools/sim_bird.mjs processes and
// reads what each one says it HOLDS -- `<tag> ready a-b`, `<tag> nack ...`,
// `<tag> left: ...` -- and asserts on that, and on whether a multi-turn chat
// actually completes while devices join.
//
// The birds are fake (no GPU, the hidden state is echoed) with --restream, which
// makes each one hold nothing for a couple of seconds after every assignment,
// exactly like a phone. Correctness of the TEXT is a different claim, made by the
// GPU tests.
//
// Needs a coordinator on a port that is NOT a real flock's; `npm test` starts one.
//   FLOCK_NO_TLS=1 PORT=8123 npm start &
//   FLOCK_URL=http://127.0.0.1:8123 node test/e2e/churn.test.mjs
import {spawn} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BASE = process.env.FLOCK_URL || 'http://127.0.0.1:8000';
if (/:8000(\/|$)/.test(BASE) && !process.env.FLOCK_E2E_FORCE) {
  console.error(`refusing to run against ${BASE}: that is the default port of a real ` +
                `flock and this test evicts every member. Start a coordinator on ` +
                `another port, or set FLOCK_E2E_FORCE=1.`);
  process.exit(2);
}
const RESTREAM = +(process.env.CHURN_RESTREAM_MS || 2500);

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};
const sleep = ms => new Promise(r => setTimeout(r, ms));
const status = async () => (await fetch(`${BASE}/status`)).json();
const post = async (p, body) => (await fetch(`${BASE}/${p}`, {method: 'POST',
  headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})).json();

// --- simulated birds as child processes -----------------------------------
// EVERY sim this test starts is killed when it exits, however it exits. Stand-ins
// left holding layers have blocked real devices from joining four times.
const sims = [];
function killAll() {
  for (const s of sims) { try { process.kill(-s.proc.pid, 'SIGKILL'); } catch {}
                          try { s.proc.kill('SIGKILL'); } catch {} }
}
process.on('exit', killAll);
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => { killAll(); process.exit(130); });
}

/**
 * Start one sim and watch what it prints. `state` is what the bird itself says:
 * held is the last range it confirmed, or null while it holds nothing.
 */
function sim(tag, extra = []) {
  const proc = spawn('deno', ['run', '--unstable-webgpu', '--allow-all',
    path.join(ROOT, 'tools/sim_bird.mjs'), '--fake', '--label', tag,
    '--restream', String(RESTREAM), ...extra],
    {cwd: ROOT, env: {...process.env, FLOCK_URL: BASE}, detached: true,
     stdio: ['ignore', 'pipe', 'pipe']});
  const s = {tag, proc, lines: [], held: null, readyAt: [], nacks: 0, left: null,
             reassigns: 0, loading: 0, exited: null};
  let buf = '';
  proc.stdout.on('data', d => {
    buf += d.toString();
    const parts = buf.split('\n'); buf = parts.pop();
    for (const line of parts) {
      s.lines.push(line);
      if (process.env.CHURN_VERBOSE) console.log(`    [${tag}] ${line}`);
      const m = line.match(/^(\S+) (\w[\w-]*) ?(.*)$/);
      if (!m || m[1] !== tag) continue;
      const [, , ev, rest] = m;
      if (ev === 'ready') { s.held = rest.trim(); s.readyAt.push(Date.now()); }
      else if (ev === 'loading') { s.held = null; s.loading++; }
      else if (ev === 'reassigned') { s.held = null; s.reassigns++; }
      else if (ev === 'nack') s.nacks++;
      else if (ev === 'left') s.left = rest;
    }
  });
  proc.stderr.on('data', d => { if (process.env.CHURN_VERBOSE) process.stderr.write(d); });
  proc.on('exit', code => { s.exited = code; });
  sims.push(s);
  return s;
}
const until = async (pred, ms = 30000, step = 100) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await pred()) return true; await sleep(step); }
  return false;
};
const ranges = s => s.birds.map(b => `${b.label}:${b.start}-${b.end}${b.ready ? '' : '?'}`)
                     .join(' ');

/** Run one turn and return what the stream said. */
async function turn(prompt, max_tokens = 8, reset = false) {
  const t0 = Date.now();
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
  return {stop: d.stop, cached: d.cached, tokens: d.tokens, events,
          waits: events.filter(e => e.type === 'waiting'),
          error: events.find(e => e.type === 'error')?.text, ms: Date.now() - t0};
}

const up = await fetch(`${BASE}/status`).catch(() => null);
if (!up?.ok) {
  console.error(`no coordinator at ${BASE} -- start one with FLOCK_NO_TLS=1 PORT=8123 npm start`);
  process.exit(2);
}
const s0 = await status();
for (const b of s0.birds) await post('evict', {peer_id: b.peer_id});
await sleep(300);
const LAYERS = s0.allocation.layers;
console.log(`coordinator: ${s0.model}, birds cover ${s0.allocation.bird_layers} ` +
            `(${LAYERS} layers); simulated re-stream ${RESTREAM}ms\n`);
if (LAYERS < 3) {
  console.log(`  (needs at least 3 bird layers to place 3 devices; have ${LAYERS})`);
  process.exit(0);
}

try {
  // =======================================================================
  console.log('a device joins between turns: the next turn WAITS, nobody leaves:');
  const A = sim('A', ['--slow', '10']);
  ok('the first bird streams, builds, and confirms its range',
     await until(() => A.held != null, 20000), `A holds ${A.held}`);
  ok('  and /status shows the flock ready only once it has',
     (await status()).ready, ranges(await status()));

  const t1 = await turn('Say hello.', 6, true);
  ok('turn 1 completes on one device', !t1.error && t1.stop !== 'rebalanced',
     t1.error || `${t1.stop}, ${t1.tokens} tokens`);

  const B = sim('B', ['--slow', '10']);
  await until(async () => (await status()).birds.length === 2, 10000);
  const sJ = await status();
  ok('the join reassigned A (the donor) and placed B; both are now loading',
     !sJ.ready && sJ.pending.length === 2 && sJ.missing.length === 0 ||
     (sJ.birds.length === 2 && sJ.birds.every(b => b.ready === false)), ranges(sJ));

  // Send the next message IMMEDIATELY -- this is the frame that used to arrive at
  // a device holding nothing.
  const t2 = await turn('And again.', 6, false);
  ok('turn 2 completes even though both devices were mid-stream when it was sent',
     !t2.error && t2.stop !== 'rebalanced', t2.error || `${t2.stop}, ${t2.tokens} tokens`);
  ok('  the coordinator said what it was waiting for, instead of hanging',
     t2.waits.length > 0 && t2.waits[0].pending.length > 0,
     t2.waits[0] ? t2.waits[0].pending.map(p => `${p.label}:${p.range}`).join(' ')
                 : 'no waiting events');
  ok('  the turn took at least one re-stream', t2.ms >= RESTREAM * 0.8, `${t2.ms}ms`);
  ok('  no bird was sent a frame while it held nothing', A.nacks + B.nacks === 0,
     `nacks A=${A.nacks} B=${B.nacks}`);
  ok('  nobody left', !A.left && !B.left, [A.left, B.left].filter(Boolean).join('; '));

  // =======================================================================
  console.log('\na device joins MID-TURN: the answer completes, then it is placed:');
  const t3p = turn('Count slowly.', 14, false);
  await sleep(700);
  ok('turn 3 is in flight', (await status()).busy);
  const C = sim('C', ['--slow', '10']);
  const t3 = await t3p;
  ok('turn 3 completed in full: the join waited for the turn, not the token',
     !t3.error && (t3.stop === 'eos' || t3.stop === 'length'), t3.error || t3.stop);
  ok('  so no bird was told to move while the answer was being generated',
     A.held != null && B.held != null && (A.reassigns + B.reassigns) === 1,
     `reassigns so far A=${A.reassigns} B=${B.reassigns}`);
  await until(async () => (await status()).birds.length === 3, 10000);
  await sleep(300);
  ok('after the turn C is placed by taking a run off ONE incumbent',
     (A.reassigns + B.reassigns) === 2,
     `reassigns A=${A.reassigns} B=${B.reassigns} (one of them moved for C)`);

  const t4 = await turn('Once more.', 6, false);
  ok('turn 4 completes with three devices, after waiting for the movers',
     !t4.error && t4.stop !== 'rebalanced', t4.error || `${t4.stop}, waited ${t4.waits.length}x`);
  const t5 = await turn('And once more.', 6, false);
  ok('turn 5 continues the conversation with its context intact (no join in between)',
     !t5.error && t5.cached > t4.cached, `context ${t4.cached} -> ${t5.cached}`);

  // What each bird HOLDS versus what the coordinator believes -- the check that
  // /status alone cannot make.
  const sF = await status();
  const byLabel = new Map(sF.birds.map(b => [b.label, b]));
  const agree = [A, B, C].every(s => {
    const b = byLabel.get(s.tag);
    return b && s.held === `${b.start}-${b.end}` && b.ready;
  });
  ok('every bird holds exactly the range the coordinator thinks it holds',
     agree, `${[A, B, C].map(s => `${s.tag}=${s.held}`).join(' ')} | server ${ranges(sF)}`);
  const held = [A, B, C].map(s => (s.held || '-1--1').split('-').map(Number))
                        .sort((x, y) => x[0] - y[0]);
  const [lo, hi] = sF.allocation.bird_layers.split('-').map(Number);
  const contiguous = held[0][0] === lo && held[held.length - 1][1] === hi &&
    held.every((r, i) => i === 0 || r[0] === held[i - 1][1] + 1);
  ok('  and together they hold every bird layer exactly once', contiguous,
     held.map(r => r.join('-')).join(' '));
  ok('  no bird ever received a frame it could not serve',
     A.nacks + B.nacks + C.nacks === 0, `nacks ${A.nacks}/${B.nacks}/${C.nacks}`);
  ok('  no bird left and none was dropped', ![A, B, C].some(s => s.left) &&
     sF.birds.length === 3, [A, B, C].map(s => s.left).filter(Boolean).join('; '));
  ok('  the whole thing took a bounded number of re-streams: one per join, plus the joiner',
     A.loading + B.loading + C.loading <= 3 + 2,
     `loads A=${A.loading} B=${B.loading} C=${C.loading}`);

  // =======================================================================
  console.log('\nbirds with the OLD behaviour (leave on an unready frame) still cannot cascade:');
  // The coordinator-side gate alone is enough: a bird that would leave if it were
  // sent an early frame is never sent one.
  for (const s of [A, B, C]) { try { process.kill(-s.proc.pid, 'SIGKILL'); } catch {} }
  for (const b of (await status()).birds) await post('evict', {peer_id: b.peer_id});
  await sleep(300);
  const L1 = sim('L1', ['--leave-on-unready']);
  await until(() => L1.held != null, 20000);
  const L2 = sim('L2', ['--leave-on-unready']);
  await until(async () => (await status()).birds.length === 2, 10000);
  const tl = await turn('Hello there.', 6, true);
  ok('a turn sent the moment a legacy bird joined still completes',
     !tl.error && tl.stop !== 'rebalanced', tl.error || tl.stop);
  await until(() => L1.held != null && L2.held != null, 20000);
  ok('  neither legacy bird was ever given a reason to leave', !L1.left && !L2.left,
     [L1.left, L2.left].filter(Boolean).join('; ') || `L1=${L1.held} L2=${L2.held}`);
  ok('  and both are still members', (await status()).birds.length === 2);
} finally {
  killAll();
  await sleep(200);
  for (const b of (await status().catch(() => ({birds: []}))).birds) {
    await post('evict', {peer_id: b.peer_id}).catch(() => {});
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
