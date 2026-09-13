// The one test command. `npm test` runs everything; `npm test -- unit e2e` picks.
//
//   check    node --check over every module and the inline page scripts
//   unit     test/unit/*    node, no GPU, no coordinator (two need the network)
//   e2e      test/e2e/*     against a coordinator THIS RUNNER STARTS on a spare
//                           port, with simulated birds it starts and kills
//   kernels  kernels/test_* deno, needs a GPU; the WGSL engine's own suite
//   gpu      test/gpu/*     deno, needs a GPU and the network
//
// Why the runner starts the coordinator itself: a test that needs a live flock
// and is pointed at whatever is on port 8000 will evict the real devices there,
// and stand-ins left behind by a test have blocked real phones from joining more
// than once. So the e2e coordinator gets a random free port, plain http, and every
// child process is in its own process group and killed on the way out -- on
// success, on failure, and on Ctrl-C.
import {spawn} from 'node:child_process';
import {createServer} from 'node:net';
import {readdirSync, mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const want = new Set(process.argv.slice(2));
const all = want.size === 0;
const pick = g => all || want.has(g);
const DENO = ['deno', 'run', '--unstable-webgpu', '--allow-all'];

const results = [];
const children = new Set();
function killAll() {
  for (const c of children) {
    try { process.kill(-c.pid, 'SIGKILL'); } catch {}
    try { c.kill('SIGKILL'); } catch {}
  }
  children.clear();
}
process.on('exit', killAll);
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => { killAll(); process.exit(130); });
}

/** Run one test process to completion; record pass/fail and the summary line. */
function run(name, cmd, args, {env = {}, cwd = ROOT} = {}) {
  return new Promise(resolve => {
    const t0 = Date.now();
    console.log(`\n=== ${name} ===`);
    const p = spawn(cmd, args, {cwd, env: {...process.env, ...env}, detached: true,
                                stdio: ['ignore', 'pipe', 'pipe']});
    children.add(p);
    let out = '';
    const tee = d => { const s = d.toString(); out += s; process.stdout.write(s); };
    p.stdout.on('data', tee);
    p.stderr.on('data', tee);
    p.on('exit', code => {
      children.delete(p);
      const secs = ((Date.now() - t0) / 1000).toFixed(1);
      // Each suite prints its own count; keep the last such line for the table.
      const lines = out.trim().split('\n');
      const summary = [...lines].reverse().find(l =>
        /\d+ passed|passed, \d+ failed|CHECKS PASSED|CHECKS FAILED|all checks passed|skip/i
          .test(l)) || lines[lines.length - 1] || '';
      results.push({name, ok: code === 0, secs, summary: summary.trim().slice(0, 70)});
      resolve(code === 0);
    });
  });
}

/** A port nobody is listening on. Never 8000: that is where a real flock lives. */
const freePort = () => new Promise(res => {
  const srv = createServer();
  srv.listen(0, '127.0.0.1', () => { const {port} = srv.address(); srv.close(() => res(port)); });
});

/** Start a background process (coordinator, sim bird) and return it. */
function background(args, env = {}) {
  const p = spawn(args[0], args.slice(1), {cwd: ROOT, env: {...process.env, ...env},
    detached: true, stdio: ['ignore', 'pipe', 'pipe']});
  children.add(p);
  p.log = '';
  p.stdout.on('data', d => { p.log += d; if (process.env.TEST_VERBOSE) process.stdout.write(d); });
  p.stderr.on('data', d => { p.log += d; if (process.env.TEST_VERBOSE) process.stderr.write(d); });
  return p;
}
const stop = p => { try { process.kill(-p.pid, 'SIGKILL'); } catch {} children.delete(p); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function waitFor(pred, ms, what) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await pred().catch(() => false)) return true; await sleep(300); }
  throw new Error(`timed out waiting for ${what}`);
}

// --- check -----------------------------------------------------------------
if (pick('check')) {
  await run('check: inline page scripts', 'node',
            ['tools/check_html.mjs', ...readdirSync('web').filter(f => f.endsWith('.html'))
              .map(f => `web/${f}`)]);
  const mods = [];
  for (const dir of ['server', 'tools', 'test/unit', 'test/e2e', 'web/js', 'test']) {
    for (const f of readdirSync(path.join(ROOT, dir))) {
      if (/\.(m?js)$/.test(f)) mods.push(`${dir}/${f}`);
    }
  }
  await run('check: node --check', 'node', ['--check', ...mods]);
}

// --- unit ------------------------------------------------------------------
if (pick('unit')) {
  for (const f of readdirSync(path.join(ROOT, 'test/unit')).filter(f => f.endsWith('.test.mjs')).sort()) {
    await run(`unit: ${f}`, 'node', [`test/unit/${f}`]);
  }
}

// --- e2e -------------------------------------------------------------------
if (pick('e2e')) {
  const port = await freePort();
  const BASE = `http://127.0.0.1:${port}`;
  console.log(`\n=== starting a coordinator on ${BASE} (plain http, 8 bird layers) ===`);
  // 8 bird layers rather than the default 24: enough for every scenario (three
  // devices need three), and a real bird page joining streams 67MB instead of
  // 200MB, which is the difference between a test and a download.
  // Its own state directory, thrown away afterwards: the coordinator persists
  // the conversation journal and the sessions it issued, and a test run must
  // neither read a real flock's nor leave its own behind.
  const stateDir = mkdtempSync(path.join(tmpdir(), 'flock-test-state-'));
  const coord = background([...DENO, 'server/server.js'],
    {PORT: String(port), FLOCK_NO_TLS: '1', FLOCK_BIRD_LAYERS: '8', FLOCK_STATE_DIR: stateDir});
  let coordUp = false;
  try {
    await waitFor(async () => (await fetch(`${BASE}/status`)).ok, 240000,
                  'the coordinator to load its layers');
    coordUp = true;
  } catch (e) {
    results.push({name: 'e2e: coordinator', ok: false, secs: '-',
                  summary: e.message + ': ' + coord.log.trim().split('\n').pop()});
  }
  if (coordUp) {
    const env = {FLOCK_URL: BASE};
    await run('e2e: membership.test.mjs', 'node', ['test/e2e/membership.test.mjs'], {env});
    await run('e2e: churn.test.mjs', 'node', ['test/e2e/churn.test.mjs'], {env});
    // Session durability needs REAL birds (it asserts on text); it starts and
    // kills its own.
    await run('e2e: session.test.mjs', 'node', ['test/e2e/session.test.mjs'], {env});

    // The two page-driving tests need the layers covered by REAL birds (they
    // assert on generated text), so a GPU sim holds them all first.
    console.log('\n=== starting a simulated bird (real GPU) to cover the layers ===');
    const sim = background([...DENO, 'tools/sim_bird.mjs', 'solo', '--label', 'sim'], env);
    try {
      await waitFor(async () => (await (await fetch(`${BASE}/status`)).json()).ready, 120000,
                    'the simulated bird to hold every layer');
      await run('e2e: chat_ui.test.mjs', 'node', ['test/e2e/chat_ui.test.mjs'], {env});
      await run('e2e: bird_ui.test.mjs', 'deno',
                [...DENO.slice(1), 'test/e2e/bird_ui.test.mjs'], {env});
    } catch (e) {
      results.push({name: 'e2e: simulated bird', ok: false, secs: '-',
                    summary: e.message + ': ' + sim.log.trim().split('\n').pop()});
    }
    stop(sim);
  }
  stop(coord);
  rmSync(stateDir, {recursive: true, force: true});
  // The restart test needs a coordinator it can kill, so it starts its own on
  // another free port with its own state directory.
  await run('e2e: restart.test.mjs', 'node', ['test/e2e/restart.test.mjs']);
}

// --- kernels ---------------------------------------------------------------
if (pick('kernels')) {
  const suites = readdirSync(path.join(ROOT, 'kernels')).filter(f => /^test_.*\.ts$/.test(f)).sort();
  for (const f of suites) await run(`kernels: ${f}`, DENO[0], [...DENO.slice(1), `kernels/${f}`]);
}

// --- gpu -------------------------------------------------------------------
if (pick('gpu')) {
  await run('gpu: gguf-stream.test.ts', DENO[0], [...DENO.slice(1), 'test/gpu/gguf-stream.test.ts']);
}

// --- summary ---------------------------------------------------------------
killAll();
console.log('\n' + '='.repeat(78));
const w = Math.max(...results.map(r => r.name.length), 10);
for (const r of results) {
  console.log(`${r.ok ? ' ok ' : 'FAIL'}  ${r.name.padEnd(w)}  ${String(r.secs).padStart(6)}s  ${r.summary}`);
}
const failed = results.filter(r => !r.ok).length;
console.log('='.repeat(78));
console.log(`${results.length - failed} of ${results.length} suites passed` +
            (failed ? `, ${failed} FAILED` : ''));
process.exit(failed ? 1 : 0);
