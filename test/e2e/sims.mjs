// Stand-in birds for the e2e tests: real tools/sim_bird.mjs processes, driven
// over stdin and read over stdout.
//
// Shared by the session and restart tests, which both need to kill a stand-in,
// start it again with its session, pause it, and read what it SAYS it holds --
// the same principle churn.test.mjs established: /status is the coordinator's
// belief, and every failure this family of tests guards against was invisible
// there. What each sim prints is part of its interface (see sim_bird.mjs).
//
// EVERY sim started through here is killed when the test exits, however it
// exits. Stand-ins left holding layers have blocked real devices from joining
// more than once.
import {spawn} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
export const DENO = ['deno', 'run', '--unstable-webgpu', '--allow-all'];
export const sleep = ms => new Promise(r => setTimeout(r, ms));

const sims = new Set();
export function killAll() {
  for (const s of sims) {
    try { process.kill(-s.proc.pid, 'SIGKILL'); } catch {}
    try { s.proc.kill('SIGKILL'); } catch {}
  }
  sims.clear();
}
process.on('exit', killAll);
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => { killAll(); process.exit(130); });
}

/**
 * Start one sim and watch what it prints.
 *
 * `s.held` is the range it last confirmed (null while it holds nothing), and
 * the counters are what its own log said: reassigns, replays, nacks, whether it
 * left, whether it re-attached. `extra` are sim_bird.mjs arguments; a real
 * (GPU) bird unless `--fake` is among them.
 */
export function sim(tag, base, extra = [], {verbose = process.env.SIM_VERBOSE} = {}) {
  const proc = spawn(DENO[0], [...DENO.slice(1), path.join(ROOT, 'tools/sim_bird.mjs'),
    '--label', tag, ...extra],
    {cwd: ROOT, env: {...process.env, FLOCK_URL: base}, detached: true,
     stdio: ['pipe', 'pipe', 'pipe']});
  const s = {tag, proc, lines: [], held: null, nacks: 0, left: null, reassigns: 0,
             loading: 0, replays: 0, replayTokens: 0, reattached: null, paused: 0,
             resumed: 0, reconnects: 0, session: null, exited: null};
  let buf = '';
  proc.stdout.on('data', d => {
    buf += d.toString();
    const parts = buf.split('\n'); buf = parts.pop();
    for (const line of parts) {
      s.lines.push(line);
      if (verbose) console.log(`    [${tag}] ${line}`);
      const m = line.match(/^(\S+) (\w[\w-]*) ?(.*)$/);
      if (!m || m[1] !== tag) continue;
      const [, , ev, rest] = m;
      if (ev === 'ready') s.held = rest.trim();
      else if (ev === 'loading') { s.held = null; s.loading++; }
      else if (ev === 'reassigned') { s.held = null; s.reassigns++; }
      else if (ev === 'nack') s.nacks++;
      else if (ev === 'left') s.left = rest;
      else if (ev === 'replay') { s.replays++; s.replayTokens += +(rest.match(/(\d+)@/)?.[1] || 0); }
      else if (ev === 'reattached') s.reattached = rest.trim();
      else if (ev === 'paused') s.paused++;
      else if (ev === 'resumed') s.resumed++;
      else if (ev === 'reconnected') s.reconnects++;
      else if (ev === 'session') s.session = rest.trim();
    }
  });
  proc.stderr.on('data', d => { if (verbose) process.stderr.write(d); });
  proc.on('exit', code => { s.exited = code; sims.delete(s); });
  sims.add(s);
  return s;
}

/** Send one command line to a sim's stdin (`pause`, `resume`, `leave`). */
export const tell = (s, cmd) => { try { s.proc.stdin.write(cmd + '\n'); } catch {} };

/** Kill one sim the hard way, the way a refresh or a crash does. */
export function kill(s) {
  try { process.kill(-s.proc.pid, 'SIGKILL'); } catch {}
  try { s.proc.kill('SIGKILL'); } catch {}
  sims.delete(s);
}

export const until = async (pred, ms = 30000, step = 100) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await pred()) return true; await sleep(step); }
  return false;
};

export const status = base => fetch(`${base}/status`).then(r => r.json());
export const post = (base, p, body) => fetch(`${base}/${p}`, {method: 'POST',
  headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}).then(r => r.json());
export const ranges = s => s.birds.map(b => `${b.label}:${b.start}-${b.end}` +
                                          `${b.ready ? '' : '?'}${b.paused ? '(paused)' : ''}`)
                            .join(' ');

/** Run one turn and return what the stream said. */
export async function turn(base, prompt, max_tokens = 24, reset = false) {
  const t0 = Date.now();
  const res = await fetch(`${base}/chat`, {method: 'POST',
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
  return {text: d.text, stop: d.stop, cached: d.cached, tokens: d.tokens, turn: d.turn,
          events, waits: events.filter(e => e.type === 'waiting'),
          replays: events.filter(e => e.type === 'replay'),
          error: events.find(e => e.type === 'error')?.text, ms: Date.now() - t0};
}

/** Refuse to run against port 8000: that is where a real flock lives, and these
 *  tests evict every member they find. */
export function guardPort(base) {
  if (/:8000(\/|$)/.test(base) && !process.env.FLOCK_E2E_FORCE) {
    console.error(`refusing to run against ${base}: that is the default port of a real ` +
                  `flock and this test evicts every member. Start a coordinator on ` +
                  `another port, or set FLOCK_E2E_FORCE=1.`);
    process.exit(2);
  }
}

/** A pass/fail recorder in the house style. */
export function checker() {
  const c = {pass: 0, fail: 0};
  c.ok = (name, cond, extra = '') => {
    if (cond) { c.pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
    else { c.fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
    return !!cond;
  };
  c.done = () => {
    console.log(`\n${c.pass} passed, ${c.fail} failed`);
    return c.fail ? 1 : 0;
  };
  return c;
}
