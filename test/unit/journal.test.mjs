// The conversation journal: exact ids, exact activation bits, append-only on
// disk, and honest about a torn tail. Needs no GPU and no network.
//
//   node test/unit/journal.test.mjs
import {mkdtempSync, rmSync, readFileSync, appendFileSync, statSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {Journal} from '../../server/journal.js';
import {f32to16} from '../../web/js/wire.mjs';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};
const H = 8;
const acts = (n, seed) => Float32Array.from({length: n * H}, (_, i) => Math.sin(seed + i) * 0.05);
const DIR = mkdtempSync(path.join(tmpdir(), 'flock-journal-'));
const FILE = path.join(DIR, 'journal.bin');

console.log('in memory: ids in order, activations as the f16 bits the wire carried:');
{
  const j = new Journal({hidden: H});
  j.append([1, 2, 3], 0, acts(3, 1));
  j.append([4], 3, acts(1, 2));
  j.endTurn();
  j.append([5, 6], 4, acts(2, 3));
  ok('ids come back in order', j.ids().join(',') === '1,2,3,4,5,6', j.ids().join(','));
  ok('length and turns are tracked', j.length === 6 && j.turns === 1 && j.turnStart === 4,
     `${j.length}/${j.turns}/${j.turnStart}`);
  const a = j.activations(2, 3);        // positions 2,3,4 across three records
  const want = [...acts(3, 1).subarray(2 * H), ...acts(1, 2), ...acts(2, 3).subarray(0, H)];
  const same = want.every((v, i) => f32to16(v) === f32to16(a[i]));
  ok('activations gather across record boundaries, bit-exact after the f16 round trip', same);
  let threw = null;
  try { j.append([9], 5, acts(1, 9)); } catch (e) { threw = e.message; }
  ok('appending at the wrong offset throws instead of misfiling', /cannot append/.test(threw || ''), threw);
  j.rewindTurn();
  ok('rewinding a turn drops back to where it began', j.length === 4 && j.ids().join(',') === '1,2,3,4',
     j.ids().join(','));
  try { threw = null; j.truncate(2); } catch (e) { threw = e.message; }
  ok('truncating inside a record is refused', /not a record boundary/.test(threw || ''), threw);
  j.clear();
  ok('clear forgets everything', j.length === 0 && j.turns === 0 && j.ids().length === 0);
}

console.log('\non disk: append-only, reloads exactly, cuts off a torn tail:');
{
  const j = new Journal({hidden: H, file: FILE});
  j.clear();
  j.append([10, 11], 0, acts(2, 4));
  j.endTurn();
  j.append([12, 13, 14], 2, acts(3, 5));
  const size1 = statSync(FILE).size;
  j.append([15], 5, acts(1, 6));
  ok('every append grows the file', statSync(FILE).size > size1);
  j.rewindTurn();                        // back to 2: a truncate record, not a rewrite
  ok('a rewind appends a record rather than rewriting', statSync(FILE).size > size1);

  const k = new Journal({hidden: H, file: FILE});
  const r = k.load();
  ok('reload gives the same ids, length and turns', r.ok && k.ids().join(',') === '10,11' &&
     k.length === 2 && k.turns === 1 && k.turnStart === 2, `${JSON.stringify(r)} ids=${k.ids()}`);
  const a0 = k.activations(0, 2), want = acts(2, 4);
  ok('and the same activation bits', want.every((v, i) => f32to16(v) === f32to16(a0[i])));

  // The process dies mid-write: a partial record at the end of the file.
  k.append([16, 17], 2, acts(2, 7));
  const whole = readFileSync(FILE);
  appendFileSync(FILE, whole.subarray(whole.length - 20, whole.length - 3));
  const m = new Journal({hidden: H, file: FILE});
  const r2 = m.load();
  ok('a torn last record is cut off, everything before it kept',
     r2.ok && r2.torn === 17 && m.ids().join(',') === '10,11,16,17', JSON.stringify(r2));
  ok('  and the file was trimmed so the next append is well-formed',
     statSync(FILE).size === whole.length);

  const other = new Journal({hidden: H + 1, file: FILE});
  const r3 = other.load();
  ok('a journal for a different model is ignored, not misread', !r3.ok && other.length === 0,
     JSON.stringify(r3));
}
rmSync(DIR, {recursive: true, force: true});

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
