// Session tokens: bound to a peer, an address and a device's limits; expire;
// persist. Needs no GPU and no network.
//
//   node test/unit/session.test.mjs
import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {Sessions, capsKey} from '../../server/session.js';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};
let t = 1e9;
const now = () => t;
const caps = {bind: 1e9, budget: Infinity, cores: 8, gpu: 'apple'};

console.log('a token names one membership and checks who presents it:');
{
  const s = new Sessions({ttlMs: 1000, now});
  const tok = s.issue({peerId: 'p1', from: '10.0.0.5', caps, label: 'iPhone'});
  ok('tokens are long and random', /^[0-9a-f]{48}$/.test(tok), tok.length + ' chars');
  ok('the right device from the right address re-attaches',
     s.validate(tok, {from: '10.0.0.5', caps}).ok);
  ok('unlimited limits compare equal whether live or reloaded',
     capsKey(caps) === capsKey({...caps, budget: null}));
  ok('a forged token is refused', s.validate('f'.repeat(48), {from: '10.0.0.5', caps}).reason === 'unknown token');
  ok('the same token from another address is refused',
     /from 10\.0\.0\.9/.test(s.validate(tok, {from: '10.0.0.9', caps}).reason));
  ok('the same token with different device limits is refused',
     s.validate(tok, {from: '10.0.0.5', caps: {...caps, bind: 64e6}}).reason === 'different device limits');
  t += 900;
  ok('activity pushes the expiry out', s.touch('p1') && s.validate(tok, {from: '10.0.0.5', caps}).ok);
  t += 1001;
  const v = s.validate(tok, {from: '10.0.0.5', caps});
  ok('an idle token expires', !v.ok && v.reason === 'expired', v.reason);
  ok('  and is gone afterwards', s.get('p1') === null && s.expired('p1'));
  const t2 = s.issue({peerId: 'p2', from: '10.0.0.6', caps});
  const t3 = s.issue({peerId: 'p2', from: '10.0.0.6', caps});
  ok('re-issuing for a peer revokes its earlier token', t2 !== t3 &&
     !s.validate(t2, {}).ok && s.validate(t3, {}).ok);
  ok('revoke forgets it', s.revoke('p2') && !s.validate(t3, {}).ok && !s.revoke('p2'));
}

console.log('\npersisted with the placement, dropping what has expired:');
{
  const DIR = mkdtempSync(path.join(tmpdir(), 'flock-session-'));
  const FILE = path.join(DIR, 'sessions.json');
  const s = new Sessions({file: FILE, ttlMs: 5000, now});
  const a = s.issue({peerId: 'a', from: '10.0.0.1', caps, label: 'A', onCoordinator: true});
  const b = s.issue({peerId: 'b', from: '10.0.0.2', caps, label: 'B'});
  s.update('a', {start: 4, end: 9});
  s.update('b', {start: 10, end: 27, paused: true, paused_at: t});
  t += 3000;
  s.touch('b'); s.save();
  t += 2500;                     // a has lapsed (5s), b was touched 2.5s ago
  const r = new Sessions({file: FILE, ttlMs: 5000, now});
  const got = r.load();
  ok('only the live session comes back', got.length === 1 && got[0].peer_id === 'b',
     got.map(x => x.peer_id).join(','));
  ok('with its placement and pause state', got[0].start === 10 && got[0].end === 27 && got[0].paused === true);
  ok('and its token still validates from its address', r.validate(b, {from: '10.0.0.2', caps}).ok);
  ok('the lapsed one does not', !r.validate(a, {from: '10.0.0.1', caps}).ok);
  rmSync(DIR, {recursive: true, force: true});
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
