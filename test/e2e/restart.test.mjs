// COORDINATOR RESTART: the process dies mid-conversation and the conversation
// survives it.
//
// The coordinator journals every token it feeds (and the activation leaving its
// own layers) to disk, and persists the sessions it issued. On start it reloads
// both: the conversation's length and turn count come back, its own K/V cache is
// recomputed by prefilling its layers from the journaled ids, the birds
// re-attach with the peer ids they had, and the next turn replays the journal
// through them before generating. So turn 3 after a restart is exactly turn 3 of
// an uninterrupted run -- asserted here against a reference run on the same
// devices.
//
// This test starts ITS OWN coordinator (on a free port, with its own state
// directory) because it has to kill it. `npm test` runs it; by hand:
//   node test/e2e/restart.test.mjs
import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {createServer} from 'node:net';
import {spawn} from 'node:child_process';
import path from 'node:path';
import {ROOT, DENO, sim, kill, killAll, until, status, ranges, turn, checker, sleep}
  from './sims.mjs';

const c = checker();
const {ok} = c;
const STATE = mkdtempSync(path.join(tmpdir(), 'flock-restart-'));
const sess = tag => path.join(STATE, `${tag}.json`);
const freePort = () => new Promise(res => {
  const srv = createServer();
  srv.listen(0, '127.0.0.1', () => { const {port} = srv.address(); srv.close(() => res(port)); });
});
const PORT = await freePort();
const BASE = `http://127.0.0.1:${PORT}`;

/** Start a coordinator on our port, with our state directory. */
function coordinator() {
  const p = spawn(DENO[0], [...DENO.slice(1), 'server/server.js'], {cwd: ROOT,
    env: {...process.env, PORT: String(PORT), FLOCK_NO_TLS: '1', FLOCK_BIRD_LAYERS: '8',
          FLOCK_STATE_DIR: path.join(STATE, 'coord')},
    detached: true, stdio: ['ignore', 'pipe', 'pipe']});
  p.log = '';
  p.stdout.on('data', d => { p.log += d; if (process.env.SIM_VERBOSE) process.stdout.write(d); });
  p.stderr.on('data', d => { p.log += d; if (process.env.SIM_VERBOSE) process.stderr.write(d); });
  return p;
}
const stopCoord = p => { try { process.kill(-p.pid, 'SIGKILL'); } catch {} try { p.kill('SIGKILL'); } catch {} };
process.on('exit', () => { if (coord) stopCoord(coord); });
const coordUp = () => until(async () => (await fetch(`${BASE}/status`).catch(() => null))?.ok, 240000, 300);
const readyWith = n => async () => {
  const s = await status(BASE);
  return s.ready && s.birds.length === n;
};

const P1 = 'My name is Ethan and I live in Denver. Reply with just the word OK.';
const P2 = 'What is my name? Answer in one short sentence.';
const P3 = 'Which city do I live in? Answer in one short sentence.';
const fine = t => !t.error && (t.stop === 'eos' || t.stop === 'length');
const brief = t => t.error || `${t.stop}, turn ${t.turn}, cached ${t.cached}: ${JSON.stringify(t.text)}`;

let coord = coordinator();
let A = null, B = null;
try {
  console.log(`starting a coordinator on ${BASE} (state in ${STATE})`);
  ok('the coordinator comes up', await coordUp(), coord.log.trim().split('\n').pop());
  A = sim('A', BASE, ['--session-file', sess('A')]);
  ok('the first stand-in holds every bird layer', await until(() => A.held != null, 90000), A.held);
  B = sim('B', BASE, ['--session-file', sess('B')]);
  ok('the second joins and both are ready', await until(readyWith(2), 90000), ranges(await status(BASE)));

  console.log('\nreference: three turns, no restart:');
  const R1 = await turn(BASE, P1, 24, true);
  const R2 = await turn(BASE, P2);
  const R3 = await turn(BASE, P3);
  ok('the reference conversation completes', fine(R1) && fine(R2) && fine(R3),
     [R1, R2, R3].map(brief).join(' | '));

  console.log('\nthe coordinator is killed after turn 2 and started again:');
  const t1 = await turn(BASE, P1, 24, true);
  const t2 = await turn(BASE, P2);
  ok('two turns done', fine(t1) && fine(t2) && t2.turn === 2, brief(t2));
  const before = await status(BASE);
  const peers = before.birds.map(b => b.peer_id).sort().join(',');
  const shape = ranges(before);
  const loads = `A=${A.loading} B=${B.loading}`;
  stopCoord(coord);
  await sleep(500);
  coord = coordinator();
  ok('the restarted coordinator comes up', await coordUp(), coord.log.trim().split('\n').pop());
  const s1 = await status(BASE);
  ok('it restored the conversation from its journal',
     s1.cached_tokens === t2.cached && s1.turns === 2,
     `cached ${s1.cached_tokens} (was ${t2.cached}), turns ${s1.turns}; ` +
     (coord.log.match(/\[state\][^\n]*/) || ['no [state] line'])[0]);
  ok('  and the sessions it had issued', s1.birds.length === 2 &&
     s1.birds.map(b => b.peer_id).sort().join(',') === peers, ranges(s1));
  ok('both stand-ins reconnected and confirmed the layers they still hold',
     await until(readyWith(2), 60000) && A.reconnects > 0 && B.reconnects > 0,
     `${ranges(await status(BASE))}; reconnects A=${A.reconnects} B=${B.reconnects}`);
  ok('  with the same peer ids and the same split', ranges(await status(BASE)) === shape,
     `${shape} -> ${ranges(await status(BASE))}`);
  ok('  nobody re-streamed for it', `A=${A.loading} B=${B.loading}` === loads,
     `loads before ${loads}, after A=${A.loading} B=${B.loading}`);
  const t3 = await turn(BASE, P3);
  ok('turn 3 continues the conversation', fine(t3) && t3.turn === 3 && t3.cached > t2.cached, brief(t3));
  ok('  after the journal was replayed through both birds',
     A.replays > 0 && B.replays > 0 && t3.replays.length > 0,
     `replay frames A=${A.replays} B=${B.replays}; ${JSON.stringify(t3.replays[0] || null)}`);
  ok('  the answer is EXACTLY the reference answer', t3.text === R3.text,
     `${JSON.stringify(t3.text)} vs reference ${JSON.stringify(R3.text)}`);
  ok('  and it reflects the earlier turns', /Denver/.test(t3.text || ''), JSON.stringify(t3.text));
  ok('  no frame was refused, nobody left', A.nacks + B.nacks === 0 && !A.left && !B.left,
     `nacks A=${A.nacks} B=${B.nacks}`);

  console.log('\na bird refreshing after the restart is also replayed:');
  kill(B);
  B = sim('B', BASE, ['--session-file', sess('B')]);
  ok('it re-attached to the restarted coordinator with its session',
     await until(() => B.reattached != null, 60000) && await until(readyWith(2), 90000),
     `${B.reattached}; ${ranges(await status(BASE))}`);
  const t4 = await turn(BASE, P2);
  ok('and the next turn still has the whole conversation',
     fine(t4) && t4.turn === 4 && t4.cached > t3.cached && /Ethan/.test(t4.text || ''), brief(t4));
} finally {
  killAll();
  stopCoord(coord);
  await sleep(200);
  rmSync(STATE, {recursive: true, force: true});
}

process.exit(c.done());
