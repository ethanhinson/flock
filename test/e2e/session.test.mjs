// SESSION DURABILITY: a refresh, a join and a pause no longer cost the
// conversation its context.
//
// Before this, every one of those was a reassignment, and a reassignment dropped
// the conversation: the K/V cache is sharded by layer, a moved layer left its
// keys behind, and re-encoding the history was not an option because the chat
// template scaffolds the current turn differently from past ones. Seen live: a
// second phone joining reset the chat to `turn: 1, cached: 29`.
//
// Now the coordinator journals the exact token ids it fed and the activations
// that left its own layers, and REPLAYS that journal through the birds whenever
// the topology changes -- so the rebuilt caches are bit-for-bit what an
// uninterrupted flock would hold. This test proves that the strong way: it runs
// the same three-turn conversation once uninterrupted as a REFERENCE, then again
// with a refresh in the middle, and asserts the answer after the refresh is
// exactly the reference's. Greedy decode plus bit-identical replay means
// "exactly", not "similar".
//
// The birds are REAL (GPU) stand-ins, because the claim is about text, and it
// asserts on what each stand-in says it did -- re-attached, replayed, held --
// never on /status alone.
//
// Needs a coordinator on a port that is NOT a real flock's; `npm test` starts one.
//   FLOCK_NO_TLS=1 PORT=8123 FLOCK_BIRD_LAYERS=8 npm start &
//   FLOCK_URL=http://127.0.0.1:8123 node test/e2e/session.test.mjs
import {mkdtempSync, readFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {sim, tell, kill, killAll, until, status, post, ranges, turn, guardPort,
        checker, sleep} from './sims.mjs';

const BASE = process.env.FLOCK_URL || 'http://127.0.0.1:8000';
guardPort(BASE);
const c = checker();
const {ok} = c;

const STATE = mkdtempSync(path.join(tmpdir(), 'flock-session-'));
const sess = tag => path.join(STATE, `${tag}.json`);
const evictAll = async () => {
  for (const b of (await status(BASE)).birds) await post(BASE, 'evict', {peer_id: b.peer_id});
  await sleep(300);
};
const readyWith = n => async () => {
  const s = await status(BASE);
  return s.ready && s.birds.length === n && s.birds.every(b => !b.paused);
};

// A conversation whose later turns depend on the first. Short answers, so a
// stand-in reaches EOS well inside the token cap.
const P1 = 'My name is Ethan and I live in Denver. Reply with just the word OK.';
const P2 = 'What is my name? Answer in one short sentence.';
const P3 = 'Which city do I live in? Answer in one short sentence.';
const P4 = 'Say my name and my city in one short sentence.';
const fine = t => !t.error && (t.stop === 'eos' || t.stop === 'length');
const brief = t => t.error || `${t.stop}, turn ${t.turn}, cached ${t.cached}: ${JSON.stringify(t.text)}`;

const up = await fetch(`${BASE}/status`).catch(() => null);
if (!up?.ok) {
  console.error(`no coordinator at ${BASE} -- start one with FLOCK_NO_TLS=1 PORT=8123 npm start`);
  process.exit(2);
}
await evictAll();
const s0 = await status(BASE);
console.log(`coordinator: ${s0.model}, birds cover ${s0.allocation.bird_layers}\n`);

let A = null, B = null;
try {
  // =======================================================================
  console.log('reference: two devices, three turns, nothing interrupted:');
  A = sim('A', BASE, ['--session-file', sess('A')]);
  ok('the first stand-in holds every bird layer', await until(() => A.held != null, 90000), A.held);
  B = sim('B', BASE, ['--session-file', sess('B')]);
  ok('the second joins and both are ready', await until(readyWith(2), 90000), ranges(await status(BASE)));
  const R1 = await turn(BASE, P1, 24, true);
  const R2 = await turn(BASE, P2);
  const R3 = await turn(BASE, P3);
  ok('the reference conversation completes', fine(R1) && fine(R2) && fine(R3),
     [R1, R2, R3].map(brief).join(' | '));
  ok('  and recalls what it was told', /Ethan/.test(R2.text || '') && /Denver/.test(R3.text || ''),
     `${JSON.stringify(R2.text)} ${JSON.stringify(R3.text)}`);

  // =======================================================================
  console.log('\nrefresh mid-conversation keeps context:');
  const t1 = await turn(BASE, P1, 24, true);
  const t2 = await turn(BASE, P2);
  ok('two turns done', fine(t1) && fine(t2) && t2.turn === 2, brief(t2));
  const sBefore = await status(BASE);
  const peersBefore = sBefore.birds.map(b => b.peer_id).sort().join(',');
  const rangesBefore = ranges(sBefore);
  const aReassigns = A.reassigns, aLoads = A.loading;
  // A refresh: the process dies without a word and a new one comes back with
  // the session the old one saved. Its GPU state -- layers and cache -- is gone.
  kill(B);
  B = sim('B', BASE, ['--session-file', sess('B')]);
  ok('the restarted stand-in re-attached with its session',
     await until(() => B.reattached != null, 60000), B.reattached || 'never re-attached');
  ok('  and streamed its layers again', await until(() => B.held != null, 90000), B.held);
  await sleep(300);
  const sAfter = await status(BASE);
  ok('  same peer id, same layers, same membership',
     sAfter.birds.map(b => b.peer_id).sort().join(',') === peersBefore &&
     ranges(sAfter) === rangesBefore, `${rangesBefore} -> ${ranges(sAfter)}`);
  ok('  no other bird moved or re-streamed', A.reassigns === aReassigns && A.loading === aLoads,
     `A reassigns ${aReassigns}->${A.reassigns}, loads ${aLoads}->${A.loading}`);
  ok('  nobody left and nothing was dropped', !A.left && !B.left && sAfter.birds.length === 2 &&
     sAfter.cached_tokens === t2.cached, `cached ${sAfter.cached_tokens} (turn 2 ended at ${t2.cached})`);
  const t3 = await turn(BASE, P3);
  ok('turn 3 continues the conversation', fine(t3) && t3.turn === 3 && t3.cached > t2.cached,
     brief(t3));
  ok('  the refreshed bird was replayed the whole journal first',
     B.replays > 0 && B.replayTokens === t2.cached && t3.replays.length > 0,
     `B replayed ${B.replayTokens} tokens in ${B.replays} frame(s); journal held ${t2.cached}`);
  ok('  the answer is EXACTLY what the uninterrupted flock said', t3.text === R3.text,
     `${JSON.stringify(t3.text)} vs reference ${JSON.stringify(R3.text)}`);
  ok('  and it reflects the earlier turns', /Denver/.test(t3.text || ''), JSON.stringify(t3.text));
  ok('  no frame was refused', A.nacks + B.nacks === 0, `nacks A=${A.nacks} B=${B.nacks}`);

  // =======================================================================
  console.log('\na join no longer drops context (the "turn: 1, cached: 29" reset):');
  kill(A); kill(B);
  await evictAll();
  A = sim('A', BASE, ['--session-file', sess('A2')]);
  ok('one device holds everything', await until(() => A.held != null, 90000), A.held);
  const j1 = await turn(BASE, P1, 24, true);
  ok('turn 1 on one device', fine(j1) && j1.turn === 1, brief(j1));
  B = sim('B', BASE, ['--session-file', sess('B2')]);
  ok('a second device joins and both are ready', await until(readyWith(2), 90000),
     ranges(await status(BASE)));
  const sJ = await status(BASE);
  ok('  the join did NOT drop the context', sJ.cached_tokens === j1.cached && sJ.turns === 1 &&
     sJ.last_rebalance && sJ.last_rebalance.dropped_context === false,
     `cached ${sJ.cached_tokens} (was ${j1.cached}), turns ${sJ.turns}, ` +
     `last_rebalance ${JSON.stringify(sJ.last_rebalance)}`);
  const j2 = await turn(BASE, P2);
  ok('turn 2 continues the SAME conversation', fine(j2) && j2.turn === 2 && j2.cached > j1.cached,
     brief(j2));
  ok('  the newcomer and the donor were both replayed', A.replays > 0 && B.replays > 0,
     `replay frames A=${A.replays} B=${B.replays}`);
  if (j1.text === R1.text) {
    ok('  the answer is EXACTLY the two-device reference answer', j2.text === R2.text,
       `${JSON.stringify(j2.text)} vs ${JSON.stringify(R2.text)}`);
  } else {
    ok('  the answer recalls turn 1 (turn 1 differed from the reference, so no exact match)',
       /Ethan/.test(j2.text || ''), JSON.stringify(j2.text));
  }
  ok('  and it recalls the name', /Ethan/.test(j2.text || ''), JSON.stringify(j2.text));

  // =======================================================================
  console.log('\npause/resume keeps context:');
  const replaysA = A.replays, replaysB = B.replays;
  tell(B, 'pause');
  ok('the stand-in paused', await until(() => B.paused > 0, 10000));
  ok('  the coordinator reallocated around it promptly, not after the 40s grace',
     await until(async () => {
       const s = await status(BASE);
       const b = s.birds.find(x => x.label === 'B');
       return b && b.paused && b.start == null && s.ready && A.held != null;
     }, 45000), ranges(await status(BASE)));
  const sP = await status(BASE);
  ok('  the paused device is still a member, holding nothing',
     sP.birds.length === 2 && sP.birds.find(x => x.label === 'B')?.paused === true, ranges(sP));
  ok('  and the context survived the move', sP.cached_tokens === j2.cached, `cached ${sP.cached_tokens}`);
  const p3 = await turn(BASE, P3);
  ok('the next turn completes on the remaining device with context intact',
     fine(p3) && p3.turn === 3 && p3.cached > j2.cached && /Denver/.test(p3.text || ''), brief(p3));
  ok('  the survivor was replayed before it', A.replays > replaysA, `A replay frames ${replaysA}->${A.replays}`);
  tell(B, 'resume');
  ok('the stand-in resumed and re-attached', await until(() => B.resumed > 0, 20000));
  ok('  it is placed again and ready', await until(readyWith(2), 90000), ranges(await status(BASE)));
  const p4 = await turn(BASE, P4);
  ok('the following turn includes it and context is still intact',
     fine(p4) && p4.turn === 4 && p4.cached > p3.cached &&
     /Ethan/.test(p4.text || '') && /Denver/.test(p4.text || ''), brief(p4));
  ok('  the resumed bird was replayed', B.replays > replaysB, `B replay frames ${replaysB}->${B.replays}`);
  ok('  nobody left', !A.left && !B.left, [A.left, B.left].filter(Boolean).join('; '));
  ok('  pause and resume cost the resumed bird one re-stream', B.loading >= 2, `B loads ${B.loading}`);

  // =======================================================================
  console.log('\nsecurity: a forged or stolen session is a fresh join, never someone else\'s membership:');
  const members = (await status(BASE)).birds.map(b => b.peer_id);
  const caps = {gpu: true, computeOk: true, cores: 8, vendor: 'sim'};
  const forged = await post(BASE, 'join', {session: 'f'.repeat(64), label: 'forger', caps});
  ok('a forged token gets a fresh join', !forged.error && !forged.reattached && forged.peer_id &&
     !members.includes(forged.peer_id), JSON.stringify({peer: forged.peer_id, re: forged.reattached}));
  await post(BASE, 'evict', {peer_id: forged.peer_id});
  const stolen = JSON.parse(readFileSync(sess('A2'), 'utf8')).session;
  const thief = await post(BASE, 'join', {session: stolen, label: 'thief',
    caps: {...caps, maxStorageBufferBindingSize: 64e6, cores: 2, vendor: 'other'}});
  ok('a real token from a device with different limits gets a fresh join',
     !thief.error && !thief.reattached && thief.peer_id && !members.includes(thief.peer_id),
     JSON.stringify({peer: thief.peer_id, re: thief.reattached}));
  await post(BASE, 'evict', {peer_id: thief.peer_id});
  ok('  the real members were untouched', (await until(async () => {
    const s = await status(BASE);
    return s.birds.map(b => b.peer_id).sort().join(',') === [...members].sort().join(',');
  }, 5000)), ranges(await status(BASE)));
  ok('  and the flock still answers with its context', await until(readyWith(2), 90000) &&
     await (async () => { const t = await turn(BASE, P2); return fine(t) && /Ethan/.test(t.text || ''); })());
} finally {
  killAll();
  await sleep(200);
  await evictAll().catch(() => {});
  rmSync(STATE, {recursive: true, force: true});
}

process.exit(c.done());
