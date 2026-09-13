// Dynamic membership: devices join and leave, and the assignment follows.
//
// The behaviour under test is the one FLOCK_BIRDS made impossible. A flock used to be
// N fixed slots decided at startup: the third phone to point at a two-slot
// coordinator got "flock full", and N could not change without a restart. So every
// check here is about a number that used to be a constant.
//
// The links are stubbed -- a fake `ws` object with a send() -- because what is under
// test is membership and placement, not the transport. The transport is unchanged and
// is covered by node/test/bird_ui.test.mjs against a live coordinator.
//
// Needs no GPU and no network.
//
//   node test/flock.test.mjs
import {Flock} from '../src/mesh.js';
import {MIN_SAMPLES, COOLDOWN_MS} from '../src/speed.js';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};
const MB = 1e6;

/** Layers 24-27, the split flock has always shipped, at 4.2MB each. */
const LAYERS = (n = 4, first = 24, bytes = 4.2 * MB) =>
  Array.from({length: n}, (_, i) => ({
    layer: first + i, bytes, maxTensor: bytes / 4,
    biggest: `blk.${first + i}.ffn_down.weight`}));

/** A fake websocket that records what the coordinator told this device. */
const fakeWs = () => { const sent = []; return {sent, send: s => sent.push(JSON.parse(s))}; };

/** An injectable clock. It must NOT advance on read: Flock.now() is called several
 *  times per plan(), and a read-advancing clock pushed lastMoveAt into the future and
 *  made the cooldown look like 20,000 years. */
function clock(at = 1e6) {
  const c = () => at;
  c.advance = ms => { at += ms; };
  return c;
}

/** Admit a device and connect it, the way /join then the websocket hello do.
 *
 *  lastSeen is real wall time on purpose: Bird.alive() is judged against the real
 *  clock (a device is or is not answering right now), while the rebalancer's cooldown
 *  runs off Flock.now(), which a test can move. Mixing them would make a test that
 *  fast-forwards the cooldown also declare every device dead. */
function join(flock, id, {label = id, caps = null, ms = null} = {}) {
  const b = flock.claim(id, label, caps);
  b.ws = fakeWs();
  b.lastSeen = Date.now();
  if (ms != null) b.lastMs = ms;
  return b;
}
const ranges = f => f.chain().map(b => `${b.peerId}:${b.start}-${b.end}`).join(' ');

// =========================================================================
console.log('one device holds everything; there is no N to be full at:');
{
  const f = new Flock(LAYERS());
  join(f, 'a');
  f.plan({force: true});
  ok('a single device covers all four layers', ranges(f) === 'a:24-27', ranges(f));
  ok('  and the flock reports itself ready', f.ready());
  ok('  with nothing missing', f.missing().length === 0);
}

// =========================================================================
console.log('\ndevices 2..8 join one at a time and the split follows each time:');
{
  const f = new Flock(LAYERS(28, 0));       // 28 layers, so 8 devices still fit
  for (let n = 1; n <= 8; n++) {
    join(f, `d${n}`);
    f.plan({force: true});
    ok(`${n} device${n === 1 ? '' : 's'}: covered, nobody empty, nobody refused`,
       f.ready() && f.chain().length === n && f.chain().every(b => b.end >= b.start),
       ranges(f));
  }
  ok('no join was ever refused for being "full"', f.members().length === 8);
}

// =========================================================================
console.log('\na device leaves and its layers are handed on:');
{
  const f = new Flock(LAYERS(8, 0));
  join(f, 'a'); join(f, 'b'); join(f, 'c');
  f.plan({force: true});
  const before = ranges(f);
  ok('three devices cover 0-7', f.ready(), before);

  ok('release() reports it was a member', f.release('b') === true);
  f.plan({force: true});
  ok('the remaining two still cover every layer', f.ready() && f.chain().length === 2,
     ranges(f));
  ok('  so no layer is orphaned', f.missing().length === 0, f.missing().join(','));

  ok('releasing a stranger is a no-op, not a throw', f.release('nobody') === false);
}

// =========================================================================
console.log('\nlosing a device without warning leaves the gap VISIBLE:');
{
  const f = new Flock(LAYERS(8, 0));
  const a = join(f, 'a'); join(f, 'b');
  f.plan({force: true});
  // Its socket dies and it stops answering. Until the sweeper removes it the flock
  // must report itself NOT ready and name the layers nobody is running -- reporting
  // ready here is the failure that produces silently wrong text.
  a.ws = null; a.lastSeen = Date.now() - 60000;      // silent for a minute
  ok('the flock is not ready while a holder is silent', !f.ready());
  ok('  and names the layers nobody is answering for',
     f.missing().length === 1 && /^0-/.test(f.missing()[0]), f.missing().join(','));
  const gone = f.sweep();
  ok('the sweeper removes it', gone.length === 1 && gone[0] === 'a', gone.join(','));
  f.plan({force: true});
  ok('  and the survivor takes the whole range', f.ready() && ranges(f) === 'b:0-7',
     ranges(f));
}

// =========================================================================
console.log('\na device that refreshes keeps its layers (a refresh is not a leave):');
{
  const f = new Flock(LAYERS(8, 0));
  const a = join(f, 'a'); join(f, 'b');
  f.plan({force: true});
  const held = `${a.start}-${a.end}`;
  // What the websocket close handler does: tear the link down, but do NOT remove.
  a.teardown();
  f.plan();
  ok('it is still a member during the reconnect window', f.byPeer('a') !== null);
  ok('  and still holds the same layers', `${a.start}-${a.end}` === held, held);
  // It comes back with the same peer id.
  a.ws = fakeWs(); a.lastSeen = f.now();
  ok('  so reconnecting costs no reallocation and no dropped context',
     f.plan().move === false && ranges(f).includes(`a:${held}`), ranges(f));
}

// =========================================================================
console.log('\na device with a small binding limit never gets an oversized layer:');
{
  // Layer 2's largest tensor is 200MB; the phone's limit is WebGPU's 128 MiB default.
  const ls = LAYERS(4, 0, 20 * MB);
  ls[2].maxTensor = 200 * MB;
  ls[2].biggest = 'blk.2.ffn_down.weight';
  const f = new Flock(ls);
  join(f, 'phone', {label: 'iPhone',
    caps: {maxStorageBufferBindingSize: 128 * 1024 * 1024}});
  join(f, 'mac', {label: 'Mac',
    caps: {maxStorageBufferBindingSize: 4 * 1024 ** 3}});
  f.plan({force: true});
  const phone = f.byPeer('phone');
  ok('the phone does not hold layer 2', !(2 >= phone.start && 2 <= phone.end),
     ranges(f));
  ok('  and every layer is still covered', f.ready(), ranges(f));
  ok('  the reported limit is visible in /status', phone.info().max_binding_mb === 134,
     String(phone.info().max_binding_mb));
}

// =========================================================================
console.log('\na flock that cannot hold the model says so and holds NOTHING:');
{
  // Qwen3-14B's output.weight: 638MB in one tensor, against a 128 MiB limit. Two
  // layers, so "the devices cannot hold it" is the only reason it can fail -- one
  // layer and two devices would fail for the unrelated reason that somebody gets
  // nothing.
  const ls = [{layer: 39, bytes: 200 * MB, maxTensor: 20 * MB,
               biggest: 'blk.39.ffn_down.weight'},
              {layer: 40, bytes: 700 * MB, maxTensor: 638 * MB,
               biggest: 'output.weight'}];
  const f = new Flock(ls);
  join(f, 'phone', {label: 'iPhone',
    caps: {maxStorageBufferBindingSize: 128 * 1024 * 1024}});
  f.plan({force: true});
  ok('infeasible is set with the precise reason', !!f.infeasible &&
     /output\.weight/.test(f.infeasible.message), f.infeasible?.message.slice(0, 70));
  ok('  it is NOT reported ready', !f.ready());
  ok('  no device is left holding a range it cannot serve',
     f.birds.every(b => !b.placed()));
  ok('  and the detail is structured for the UI', f.infeasible.detail.kind === 'tensor');

  // A device that CAN hold it turns up. The failure must clear.
  join(f, 'mac', {label: 'Mac', caps: {maxStorageBufferBindingSize: 4 * 1024 ** 3}});
  f.plan({force: true});
  ok('adding a capable device clears the failure', !f.infeasible && f.ready(),
     ranges(f));
}

// =========================================================================
console.log('\na deliberately slow device ends up with fewer layers:');
{
  const t = clock();
  const f = new Flock(LAYERS(12, 0, 5 * MB), {now: t});
  const fast = join(f, 'fast', {label: 'Mac'});
  const slow = join(f, 'slow', {label: 'iPhone'});
  f.plan({force: true});
  const even = [fast.start, fast.end, slow.start, slow.end].join(',');
  ok('with no timings yet, both devices are assumed equal',
     fast.end - fast.start === slow.end - slow.start, even);

  // Run tokens. The slow device is 5x slower per byte, exactly the measured ratio.
  // observe() is called at a token boundary, as /chat does it. The clock jumps past
  // the cooldown between tokens so the cooldown is not what is being tested here.
  for (let i = 0; i < 40; i++) {
    fast.lastMs = fast.bytes / 5e6;
    slow.lastMs = slow.bytes / 1e6;
    t.advance(COOLDOWN_MS + 1000);
    f.observe();
  }
  ok('the slow device now holds fewer layers than the fast one',
     slow.end - slow.start < fast.end - fast.start,
     `fast ${fast.start}-${fast.end}, slow ${slow.start}-${slow.end}`);
  ok('  and the reason is on the record, in bytes and MB/ms',
     /% of the bird bytes/.test(slow.why) && /MB\/ms/.test(slow.why), slow.why);
  ok('  /status carries the measured rate and the sample count',
     slow.info().rate_trusted && slow.info().rate_samples >= MIN_SAMPLES,
     `${slow.info().rate_mb_per_ms}MB/ms over ${slow.info().rate_samples}`);
  ok('  the flock reports what the split cost and what an even one would have',
     f.allocation().makespan_ms < f.allocation().even_split_ms,
     `${f.allocation().makespan_ms}ms vs even ${f.allocation().even_split_ms}ms`);
}

// =========================================================================
console.log('\none slow sample does NOT move any layers:');
{
  const t = clock();
  const f = new Flock(LAYERS(12, 0, 5 * MB), {now: t});
  const a = join(f, 'a'), b = join(f, 'b');
  f.plan({force: true});
  // Settle both at the same rate.
  for (let i = 0; i < 10; i++) {
    a.lastMs = a.bytes / 2e6; b.lastMs = b.bytes / 2e6;
    t.advance(COOLDOWN_MS + 1000);
    f.observe();
  }
  const before = ranges(f);
  // One catastrophic token on b: a GC pause, the user switching apps. The cooldown is
  // stepped past on purpose, so it is the GAIN GATE and the EWMA being tested and not
  // the cooldown hiding the result.
  b.lastMs = b.bytes / 2e6 * 20;
  t.advance(COOLDOWN_MS + 1000);
  const d = f.observe();
  ok('nothing moved on one 20x-slow token', ranges(f) === before && !d.move,
     `${before}  |  ${d.reason}`);
  ok('  and the refusal says why, in the numbers', /gain too small|already optimal/
     .test(d.reason), d.reason);
}

// =========================================================================
console.log('\na mid-token change is deferred to the token boundary:');
{
  const f = new Flock(LAYERS(8, 0));
  join(f, 'a');
  f.plan({force: true});
  ok('one device holds 0-7', ranges(f) === 'a:0-7', ranges(f));

  // A token is in flight, so the server calls defer() instead of plan().
  const b = f.claim('b', 'phone');
  f.defer();
  ok('the joining device is a member', f.members().length === 2);
  ok('  but holds no layers yet', !b.placed());
  ok('  and the device that IS serving the token keeps its whole range',
     ranges(f) === 'a:0-7', ranges(f));
  ok('  a change is recorded as pending', f.pending);

  // The token finishes.
  b.ws = fakeWs(); b.lastSeen = f.now();
  const d = f.applyPending();
  ok('at the boundary it lands', d.move && b.placed(), ranges(f));
  ok('  and it is reported as a MOVE, so the caller knows the cache is stale',
     d.moved.length > 0, d.moved.join(','));
  ok('  pending is cleared', !f.pending);
  ok('  a second applyPending is a no-op', f.applyPending() === null);
}

// =========================================================================
console.log('\nevery bird is told its new range, not left to guess:');
{
  const f = new Flock(LAYERS(8, 0));
  const a = join(f, 'a');
  f.plan({force: true});
  a.ws.sent.length = 0;
  const b = join(f, 'b');
  f.plan({force: true});
  f.announce();
  const toA = a.ws.sent.filter(m => m.t === 'chain').pop();
  const toB = b.ws.sent.filter(m => m.t === 'chain').pop();
  ok('a is told its new (shorter) range', toA && toA.start === a.start &&
     toA.end === a.end, JSON.stringify(toA));
  ok('  and that b is now its successor', toA.next_peer === 'b', toA.next_peer);
  ok('b is told its range and that it answers the coordinator',
     toB && toB.start === b.start && toB.next_peer === null, JSON.stringify(toB));

  // A member holding nothing is told so explicitly, rather than left silent.
  const c = f.claim('c', 'c'); c.ws = fakeWs(); c.lastSeen = f.now();
  const chain = f.chainFor('c');
  ok('an unplaced member is told it holds nothing', chain.start === null &&
     chain.slot === -1, JSON.stringify(chain));
}

// =========================================================================
console.log('\nmore devices than layers is a reported failure, not silent empties:');
{
  const f = new Flock(LAYERS(2, 0));
  join(f, 'a'); join(f, 'b'); join(f, 'c');
  f.plan({force: true});
  ok('infeasible names the shortfall', !!f.infeasible &&
     /only 2 layer/.test(f.infeasible.message), f.infeasible?.message.slice(0, 80));
  ok('  and nothing is placed', f.birds.every(b => !b.placed()));
  ok('  so it cannot report itself ready', !f.ready());
  f.release('c');
  f.plan({force: true});
  ok('dropping one device fixes it', !f.infeasible && f.ready(), ranges(f));
}

// =========================================================================
console.log('\nthe lap goes through the placed birds in LAYER order:');
{
  const f = new Flock(LAYERS(6, 0));
  join(f, 'a'); join(f, 'b'); join(f, 'c');
  f.plan({force: true});
  const order = f.chain().map(b => b.start);
  ok('the chain is sorted by start layer',
     order.every((s, i) => i === 0 || s > order[i - 1]), order.join('<'));
  ok('slot is the chain position', f.chain().every((b, i) => b.slot === i),
     f.chain().map(b => b.slot).join(','));
  // An unplaced member must not appear in the chain: it would forward frames it does
  // no work on, which looks like a working flock that is inexplicably slow.
  f.claim('d', 'd');
  ok('an unplaced member is not in the lap', f.chain().length === 3);
}

// =========================================================================
console.log('\na sweep re-plans, so the freed layers are not left orphaned:');
{
  const f = new Flock(LAYERS(8, 0));
  const a = join(f, 'a'); join(f, 'b');
  f.plan({force: true});
  a.ws = null; a.lastSeen = Date.now() - 60000;
  const gone = f.sweep();
  ok('the silent device is removed', gone.length === 1, gone.join(','));
  // The bug this pins: sweep() removing a device and NOT re-planning leaves its layers
  // assigned to nobody, so the flock reports itself uncovered until some caller
  // remembers to rebalance. Correct only for as long as every caller remembers.
  ok('  and its layers are already re-assigned, with no caller doing anything',
     f.ready() && ranges(f) === 'b:0-7', ranges(f));
  ok('a sweep that removes nothing does not touch the plan',
     f.sweep().length === 0 && ranges(f) === 'b:0-7');
}

// =========================================================================
console.log('\na REJOIN reporting worse limits cannot poison a working flock:');
{
  // The commonest path into /join is a reload: bird.html keeps its peer id in
  // localStorage and re-probes on every load, so "already a member" is normal, not an
  // edge case. If the new caps make the flock infeasible, the earlier version skipped
  // its rollback because the id was KNOWN -- and then every device was unplaced and
  // nothing recovered until 40s of silence let the sweeper run.
  const ls = LAYERS(4, 0, 20 * MB);
  const f = new Flock(ls);
  const a = join(f, 'a', {caps: {maxStorageBufferBindingSize: 4 * 1024 ** 3}});
  const b = join(f, 'b', {caps: {maxStorageBufferBindingSize: 4 * 1024 ** 3}});
  f.plan({force: true});
  const before = ranges(f);
  ok('two healthy devices cover the layers', f.ready(), before);

  // `a` reloads and now reports a limit below the smallest layer's largest tensor.
  const prevCaps = {...a.caps};
  f.claim('a', 'a', {maxStorageBufferBindingSize: 1});
  f.plan({force: true});
  ok('the new caps do make it infeasible', !!f.infeasible,
     f.infeasible?.message.slice(0, 60));
  ok('  and everyone is unplaced, which is why it must be undone',
     f.birds.every(x => !x.placed()));

  // The server's rollback for a KNOWN device: put its old caps back, re-plan.
  f.setCaps(a, prevCaps);
  f.plan({force: true});
  ok('restoring its previous caps restores the whole flock',
     f.ready() && !f.infeasible && ranges(f) === before, ranges(f));
  ok('  and it is still a member on the range it already loaded',
     f.byPeer('a') !== null && a.placed(), `a holds ${a.start}-${a.end}`);
}

// =========================================================================
console.log('\n200 rounds of churn never break the invariants:');
{
  // The properties that, if they ever fail, produce WRONG TEXT rather than an error:
  // a gap in the chain skips layers, an overlap runs one twice, and a duplicate slot
  // makes two devices fight over one DOM node and one position in the lap. None of
  // them throws, so none of them would be noticed without a check like this.
  const f = new Flock(LAYERS(8, 0));
  let bad = [];
  for (let round = 0; round < 200; round++) {
    const n = 1 + (round % 8);
    for (const b of [...f.birds]) f.release(b.peerId);
    for (let i = 0; i < n; i++) join(f, `d${i}_${round}`);
    f.plan({force: true});
    const chain = f.chain();
    const slots = chain.map(b => b.slot);
    if (new Set(slots).size !== slots.length) bad.push(`round ${round}: duplicate slots`);
    if (!slots.every((s, i) => s === i)) bad.push(`round ${round}: slot != chain index`);
    let want = 0;
    for (const b of chain) {
      if (b.start !== want) { bad.push(`round ${round}: gap/overlap at ${b.start}`); break; }
      want = b.end + 1;
    }
    if (want !== 8 && !f.infeasible) bad.push(`round ${round}: only covered 0-${want - 1}`);
    for (const b of f.birds) {
      if (b.placed() && (b.start < 0 || b.end > 7)) bad.push(`round ${round}: out of plan`);
    }
  }
  ok('no gaps, no overlaps, no duplicate slots, nothing outside the layer plan',
     bad.length === 0, bad.slice(0, 3).join('; ') || '200 rounds, 1-8 devices each');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
