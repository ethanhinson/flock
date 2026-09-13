// The allocator, the speed tracker and the rebalance policy — as arithmetic.
//
// EVERY CLAIM IN THIS FILE IS TESTABLE WITHOUT A PHONE, and that is deliberate:
// "a device with a 64MB binding limit never gets an oversized layer" and "a slow
// device ends up with fewer layers" are the two things that cannot be checked by
// hand without owning a drawer of devices, so the allocator is pure and they are
// checked here. What a real device adds on top is covered in the report; this file
// covers the decisions.
//
// Needs no GPU and no network, which is why it belongs in `npm test`.
//
//   node test/allocate.test.mjs
import {allocate, device, layerPlan, Infeasible, explain, DEFAULT_RATE}
  from '../src/allocate.js';
import {Rate, shouldRebalance, currentMakespan, MIN_SAMPLES, MIN_GAIN, COOLDOWN_MS,
        MIN_MS} from '../src/speed.js';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};
const MB = 1e6;
const mb = n => `${(n / MB).toFixed(0)}MB`;

/** A run of layers, all the same size unless `sizes` says otherwise. */
function layers(first, n, {bytes = 20 * MB, maxTensor = 6 * MB, sizes = null} = {}) {
  return Array.from({length: n}, (_, i) => ({
    layer: first + i,
    bytes: sizes ? sizes[i] : bytes,
    maxTensor: Array.isArray(maxTensor) ? maxTensor[i] : maxTensor,
    biggest: `blk.${first + i}.ffn_down.weight`,
  }));
}
/** Do the assigned ranges cover `layers` exactly once, contiguously, in order? */
function covers(assign, ls) {
  let want = ls[0].layer;
  for (const a of assign) {
    if (a.start !== want) return false;
    if (a.end < a.start) return false;
    want = a.end + 1;
  }
  return want === ls[ls.length - 1].layer + 1;
}

// =========================================================================
console.log('1/2/3/5/8 devices all cover the layers, whatever N is:');
// The point of removing FLOCK_BIRDS. 28 bird-side layers so every N divides
// awkwardly except 1 -- an even split would leave a remainder at 3, 5 and 8.
{
  const ls = layers(0, 28);
  for (const n of [1, 2, 3, 5, 8]) {
    const devs = Array.from({length: n}, (_, i) => device({id: `d${i}`, label: `d${i}`}));
    const r = allocate(ls, devs);
    ok(`${n} device${n === 1 ? '' : 's'} cover layers 0-27 contiguously`,
       covers(r.assign, ls) && r.assign.length === n,
       r.assign.map(a => `${a.start}-${a.end}`).join(' '));
    ok(`  ${n}: nobody holds zero layers`, r.assign.every(a => a.layers >= 1),
       r.assign.map(a => a.layers).join('+'));
  }
}

// =========================================================================
console.log('\nweighted by BYTES, not layer count (the 13% Qwen3-14B spread):');
{
  // Real numbers: Qwen3-14B Q4_K_M layers range 185.8-210.2 MB. Four layers at the
  // extremes, two devices of identical speed. An even split by COUNT gives each two
  // layers; a byte-weighted one must give the big pair to nobody twice -- with these
  // four the counts happen to match, so the check is that the MAKESPAN reflects the
  // bytes rather than the count.
  const sizes = [210.2 * MB, 185.8 * MB, 185.8 * MB, 210.2 * MB];
  const ls = layers(24, 4, {sizes});
  const devs = [device({id: 'a', label: 'a'}), device({id: 'b', label: 'b'})];
  const r = allocate(ls, devs);
  ok('both devices get the same bytes, not just the same count',
     Math.abs(r.assign[0].bytes - r.assign[1].bytes) < 1 * MB,
     `${mb(r.assign[0].bytes)} vs ${mb(r.assign[1].bytes)}`);

  // The case where count and bytes disagree: six layers, the first three big.
  const lop = layers(0, 6, {sizes: [300 * MB, 300 * MB, 300 * MB,
                                    100 * MB, 100 * MB, 100 * MB]});
  const r2 = allocate(lop, [device({id: 'a', label: 'a'}), device({id: 'b', label: 'b'})]);
  ok('a lopsided model gives the big-layer device FEWER layers',
     r2.assign[0].layers < r2.assign[1].layers,
     r2.assign.map(a => `${a.layers} layers/${mb(a.bytes)}`).join(' | '));
  ok('  and the byte shares are closer than the layer counts',
     Math.abs(r2.assign[0].share - r2.assign[1].share) < 0.25,
     r2.assign.map(a => `${(a.share * 100).toFixed(0)}%`).join('/'));
  ok('  an even split of the same layers would have been slower',
     r2.evenMs > r2.makespanMs, `even ${r2.evenMs}ms vs ${r2.makespanMs}ms`);
}

// =========================================================================
console.log('\na small binding limit never gets an oversized layer:');
{
  // Layer 4's biggest tensor is 200MB; every other layer's is 6MB. One device has
  // WebGPU's DEFAULT 128 MiB binding limit, the other has 4 GiB. Layer 4 can only go
  // on the roomy one, at any split.
  const maxTensor = [6 * MB, 6 * MB, 6 * MB, 6 * MB, 200 * MB, 6 * MB, 6 * MB, 6 * MB];
  const ls = layers(0, 8, {maxTensor});
  const small = device({id: 'phone', label: 'phone', bind: 128 * 1024 * 1024});
  const big = device({id: 'mac', label: 'mac', bind: 4 * 1024 ** 3});
  const r = allocate(ls, [small, big]);
  const phone = r.assign.find(a => a.id === 'phone');
  ok('the layer with the 200MB tensor did NOT go to the 128MiB device',
     !(4 >= phone.start && 4 <= phone.end),
     `phone holds ${phone.start}-${phone.end}`);
  ok('  and every layer is still covered', covers(r.assign, ls),
     r.assign.map(a => `${a.id}:${a.start}-${a.end}`).join(' '));

  // Reversed order: the constrained device is now FIRST in the chain, so the only
  // contiguous answer is a shorter prefix for it.
  const r2 = allocate(ls, [big, small]);
  const phone2 = r2.assign.find(a => a.id === 'phone');
  ok('the same holds with the constrained device later in the chain',
     !(4 >= phone2.start && 4 <= phone2.end) && covers(r2.assign, ls),
     `phone holds ${phone2.start}-${phone2.end}`);

  // Every layer oversized for the small device, so it can hold nothing at all.
  const allBig = layers(0, 4, {maxTensor: 200 * MB});
  let threw = null;
  try { allocate(allBig, [small, big]); } catch (e) { threw = e; }
  ok('a device that can hold NOTHING is a reported failure, not an empty range',
     threw instanceof Infeasible, threw ? threw.message.slice(0, 60) : 'no throw');
}

// =========================================================================
console.log('\nrespects a total memory budget:');
{
  const ls = layers(0, 8, {bytes: 50 * MB});     // 400MB of layers
  const tiny = device({id: 'tiny', label: 'tiny', budget: 120 * MB});
  const roomy = device({id: 'roomy', label: 'roomy', budget: 2000 * MB});
  const r = allocate(ls, [tiny, roomy]);
  const t = r.assign.find(a => a.id === 'tiny');
  ok('a 120MB budget gets at most 2 x 50MB layers', t.bytes <= 120 * MB,
     `${mb(t.bytes)} in ${t.layers} layers`);
  ok('  and the rest is covered', covers(r.assign, ls));

  // Not enough budget anywhere.
  let threw = null;
  try {
    allocate(ls, [device({id: 'a', label: 'a', budget: 100 * MB}),
                  device({id: 'b', label: 'b', budget: 100 * MB})]);
  } catch (e) { threw = e; }
  ok('too little memory in total fails with the shortfall in MB',
     threw instanceof Infeasible && /short/.test(threw.message) &&
     threw.detail.kind === 'budget',
     threw ? threw.message.slice(0, 90) : 'no throw');
}

// =========================================================================
console.log('\na slow device ends up with fewer layers (the measured 5x):');
{
  // The measurement from the README: a phone did 4 Qwen3-0.6B layers in 15.4ms
  // (16.8MB / 15.4ms) while this Mac did 24 layers in 19.6ms (~100MB / 19.6ms).
  const ls = layers(0, 12, {bytes: 4.2 * MB});
  const phone = device({id: 'phone', label: 'iPhone', rate: 16.8e6 / 15.4});
  const mac = device({id: 'mac', label: 'Mac', rate: 100.8e6 / 19.6});
  const r = allocate(ls, [phone, mac]);
  const p = r.assign.find(a => a.id === 'phone'), m = r.assign.find(a => a.id === 'mac');
  ok('the phone gets strictly fewer layers than the Mac', p.layers < m.layers,
     `phone ${p.layers}, mac ${m.layers}`);
  ok('  the ratio tracks the ~4.7x speed ratio', m.layers / p.layers > 3,
     `${(m.layers / p.layers).toFixed(1)}x`);
  ok('  the two predicted times are close: that IS the objective',
     Math.abs(p.ms - m.ms) / Math.max(p.ms, m.ms) < 0.25,
     `phone ${p.ms}ms, mac ${m.ms}ms`);
  ok('  and it beats the even split by a lot', r.evenMs / r.makespanMs > 1.5,
     `even ${r.evenMs}ms vs weighted ${r.makespanMs}ms`);
  console.log(`       ${explain(p)}`);
  console.log(`       ${explain(m)}`);
}

// =========================================================================
console.log('\na model that cannot fit fails precisely, before anything downloads:');
{
  // The real case: Qwen3-14B Q4_K_M's output.weight is 638MB as ONE tensor and
  // tie_word_embeddings is false, so it cannot be dropped. WebGPU's default
  // maxStorageBufferBindingSize is 128 MiB.
  const ls = [{layer: 40, bytes: 700 * MB, maxTensor: 638 * MB,
               biggest: 'output.weight'}];
  let e = null;
  try {
    allocate(ls, [device({id: 'phone', label: 'iPhone', bind: 128 * 1024 * 1024})]);
  } catch (err) { e = err; }
  ok('it throws Infeasible', e instanceof Infeasible);
  ok('  names the tensor', /output\.weight/.test(e.message), e.message.slice(0, 70));
  ok('  names the layer', /layer 40/.test(e.message));
  ok('  gives the number it needed AND the number it had',
     /638\.0MB/.test(e.message) && /134\.2MB/.test(e.message));
  ok('  says adding more devices will NOT help',
     /adding more devices will not help/i.test(e.message));
  ok('  and carries it structurally too, for the UI',
     e.detail.kind === 'tensor' && e.detail.needs === 638 * MB &&
     e.detail.tensor === 'output.weight');
  console.log(`       ${e.message}`);

  // More devices than layers: the other way a flock is unsatisfiable, and it has a
  // different fix, so it gets a different message.
  let e2 = null;
  try { allocate(layers(0, 2), [device({id: 'a'}), device({id: 'b'}), device({id: 'c'})]); }
  catch (err) { e2 = err; }
  ok('more devices than layers is reported separately',
     e2 instanceof Infeasible && e2.detail.kind === 'too-many-devices',
     e2 ? e2.message.slice(0, 80) : 'no throw');
}

// =========================================================================
console.log('\nthe objective really is the makespan (the DP is exact):');
{
  // Brute-force every contiguous partition of 9 layers across 3 devices and check
  // the DP found the best one. This is the check that the allocator is optimal and
  // not merely plausible -- a greedy version passes every test above.
  const ls = layers(0, 9, {sizes: [10, 30, 12, 8, 40, 15, 9, 25, 11].map(x => x * MB)});
  const devs = [device({id: 'a', label: 'a', rate: 1e6}),
                device({id: 'b', label: 'b', rate: 3e6}),
                device({id: 'c', label: 'c', rate: 2e6})];
  const r = allocate(ls, devs);
  let bestBrute = Infinity, at = null;
  for (let i = 1; i < 9; i++) for (let j = i + 1; j < 9; j++) {
    const runs = [ls.slice(0, i), ls.slice(i, j), ls.slice(j)];
    const span = Math.max(...runs.map((run, k) =>
      run.reduce((a, l) => a + l.bytes, 0) / devs[k].rate));
    if (span < bestBrute) { bestBrute = span; at = [i, j]; }
  }
  ok('the DP matches an exhaustive search over every contiguous split',
     Math.abs(r.makespanMs - bestBrute) < 0.01,
     `dp ${r.makespanMs}ms, brute ${bestBrute.toFixed(2)}ms at cuts ${at}`);
}

// =========================================================================
console.log('\nRate: a bytes/ms EWMA that one bad token cannot move far:');
{
  const r = new Rate();
  ok('untrusted before MIN_SAMPLES samples', !r.trusted());
  ok('  and falls back to the default rate', r.rate(DEFAULT_RATE) === DEFAULT_RATE);
  for (let i = 0; i < MIN_SAMPLES; i++) r.observe(20 * MB, 10);
  ok(`trusted after ${MIN_SAMPLES} samples`, r.trusted(), `${r.samples} samples`);
  ok('  and reads about 2MB/ms', Math.abs(r.value - 2e6) / 2e6 < 0.01,
     `${(r.value / 1e6).toFixed(3)}MB/ms`);

  const before = r.value;
  r.observe(20 * MB, 200);                 // one token 20x slower: a GC pause
  ok('one 20x-slow token moves the estimate by less than 30%',
     (before - r.value) / before < 0.3,
     `${(before / 1e6).toFixed(2)} -> ${(r.value / 1e6).toFixed(2)}MB/ms`);

  // A device that is REALLY slower converges there, it just takes several samples.
  for (let i = 0; i < 30; i++) r.observe(20 * MB, 100);
  ok('a sustained 10x slowdown does converge', Math.abs(r.value - 0.2e6) / 0.2e6 < 0.05,
     `${(r.value / 1e6).toFixed(3)}MB/ms`);

  // A rate is size-invariant, which is the property that stops the controller from
  // chasing its own last decision.
  const a = new Rate(), b = new Rate();
  for (let i = 0; i < 10; i++) { a.observe(40 * MB, 20); b.observe(80 * MB, 40); }
  ok('halving a device\'s layers does not change its measured RATE',
     Math.abs(a.value - b.value) < 1, `${a.value} vs ${b.value}`);

  // A REPORTED ZERO IS "FASTER THAN THE CLOCK", NOT "NO MEASUREMENT". This had the
  // inverted failure that makes it worth a test of its own: a bird reports its time
  // rounded to 0.1ms, so a device fast enough to round to 0.0 accumulated no samples,
  // never became trusted, and therefore blocked every rebalance -- the faster the
  // device, the less measurable it was. Observed live: a Mac stuck at 1 sample while a
  // phone beside it reached 54, and "waiting for timings: 1/2 devices" was the standing
  // reason for nine turns running.
  {
    const z = new Rate();
    for (let i = 0; i < MIN_SAMPLES; i++) z.observe(20 * MB, 0);
    ok('a reported 0ms still counts as a sample', z.samples === MIN_SAMPLES,
       `${z.samples} samples`);
    ok('  so a device too fast to measure becomes trusted', z.trusted());
    ok('  at the reporting resolution, not at infinity',
       Number.isFinite(z.value) && z.value === 20 * MB / MIN_MS,
       `${(z.value / 1e6).toFixed(0)}MB/ms`);
    const slow = new Rate();
    for (let i = 0; i < MIN_SAMPLES; i++) slow.observe(20 * MB, 40);
    ok('  and still reads as far faster than a measurable device',
       z.value > slow.value * 100, `${z.value / slow.value | 0}x`);
  }
  ok('a garbage time is still ignored: no sample, no NaN',
     new Rate().observe(1 * MB, NaN).samples === 0 &&
     new Rate().observe(1 * MB, -5).samples === 0 &&
     new Rate().observe(1 * MB, null).samples === 0);
  ok('zero bytes is ignored: a device holding nothing measures nothing',
     new Rate().observe(0, 10).samples === 0);
}

// =========================================================================
console.log('\nthe rebalance gates: converge, do not thrash:');
{
  const cur = [{id: 'a', start: 0, end: 5, bytes: 60 * MB, rate: 2e6},
               {id: 'b', start: 6, end: 11, bytes: 60 * MB, rate: 2e6}];
  const same = {assign: [{id: 'a', start: 0, end: 5}, {id: 'b', start: 6, end: 11}],
                makespanMs: 30};
  const better = {assign: [{id: 'a', start: 0, end: 2}, {id: 'b', start: 3, end: 11}],
                  makespanMs: 10};
  const barely = {assign: [{id: 'a', start: 0, end: 4}, {id: 'b', start: 5, end: 11}],
                  makespanMs: 29};

  let d = shouldRebalance({current: cur, proposed: same, now: 1e6,
                           trustedCount: 2, deviceCount: 2});
  ok('an unchanged plan is a no-op', !d.move, d.reason);

  d = shouldRebalance({current: cur, proposed: barely, now: 1e6,
                       trustedCount: 2, deviceCount: 2});
  ok(`a ${((30 - 29) / 30 * 100).toFixed(0)}% gain is refused (needs ` +
     `${(MIN_GAIN * 100).toFixed(0)}%)`, !d.move, d.reason);

  d = shouldRebalance({current: cur, proposed: better, now: 1e6,
                       trustedCount: 2, deviceCount: 2});
  ok('a 67% gain is taken', d.move, d.reason);

  d = shouldRebalance({current: cur, proposed: better, now: 1e6,
                       trustedCount: 1, deviceCount: 2});
  ok('nothing moves until every device has enough samples', !d.move, d.reason);
  ok('  and the reason says how many are missing',
     /1\/2 devices/.test(d.reason), d.reason);

  d = shouldRebalance({current: cur, proposed: better, now: 1e6,
                       lastMoveAt: 1e6 - 1000, trustedCount: 2, deviceCount: 2});
  ok('a move inside the cooldown is refused', !d.move, d.reason);
  d = shouldRebalance({current: cur, proposed: better, now: 1e6,
                       lastMoveAt: 1e6 - COOLDOWN_MS - 1,
                       trustedCount: 2, deviceCount: 2});
  ok('  and allowed once the cooldown has passed', d.move, d.reason);

  // Membership changes bypass every gate: the alternative is layers nobody covers.
  d = shouldRebalance({current: cur,
    proposed: {assign: [{id: 'a', start: 0, end: 3}, {id: 'b', start: 4, end: 7},
                        {id: 'c', start: 8, end: 11}], makespanMs: 29.9},
    now: 1e6, lastMoveAt: 1e6 - 10, trustedCount: 0, deviceCount: 3});
  ok('a membership change ignores the cooldown, the samples and the gain gate',
     d.move && /membership/.test(d.reason), d.reason);
}

// =========================================================================
console.log('\nthe closed loop converges instead of oscillating:');
{
  // The whole failure mode in one simulation. Two devices, one truly 5x slower.
  // Each round: allocate, "run" a token (the ms is bytes/true-rate plus noise), feed
  // the timings back, decide. A controller that chases noise never stops moving; this
  // one must stop, and must land near the true optimum.
  const ls = layers(0, 12, {bytes: 5 * MB});
  const truth = {a: 5e6, b: 1e6};              // bytes/ms, the real hardware
  const rates = {a: new Rate(), b: new Rate()};
  let assign = null, lastMoveAt = 0, moves = 0, t = 0;
  // Deterministic +-8% noise, so the run is reproducible and a failure is a real
  // failure rather than an unlucky seed.
  let seed = 12345;
  const noise = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return 1 + ((seed % 1000) / 1000 - 0.5) * 0.16;
  };
  for (let round = 0; round < 200; round++) {
    t += 1000;
    const devs = ['a', 'b'].map(id => device({
      id, label: id, rate: rates[id].rate(DEFAULT_RATE)}));
    const proposed = allocate(ls, devs);
    const d = shouldRebalance({
      current: assign, proposed, now: t, lastMoveAt,
      trustedCount: Object.values(rates).filter(r => r.trusted()).length,
      deviceCount: 2});
    if (d.move) {
      const changed = !assign || proposed.assign.some((a, i) =>
        a.start !== assign[i].start || a.end !== assign[i].end);
      assign = proposed.assign.map(a => ({...a}));
      if (changed) { moves++; lastMoveAt = t; }
    }
    for (const a of assign) {
      rates[a.id].observe(a.bytes, (a.bytes / truth[a.id]) * noise());
      a.rate = rates[a.id].rate(DEFAULT_RATE);
    }
  }
  const fa = assign.find(a => a.id === 'a'), fb = assign.find(a => a.id === 'b');
  ok('it settles: at most 4 moves over 200 tokens', moves <= 4, `${moves} moves`);
  ok('  and settles on the RIGHT split (fast device holds ~5x the bytes)',
     fa.layers > fb.layers && Math.abs(fa.layers / fb.layers - 5) < 2.5,
     `a:${fa.layers} b:${fb.layers} (${(fa.layers / fb.layers).toFixed(1)}x)`);
  ok('  the stages end up balanced, which is the objective',
     Math.abs(fa.bytes / truth.a - fb.bytes / truth.b) < 3,
     `a ${(fa.bytes / truth.a).toFixed(1)}ms vs b ${(fb.bytes / truth.b).toFixed(1)}ms`);

  // THE CONTROL, and it needs the conditions oscillation actually happens in: two
  // devices of SIMILAR speed and enough layers that the optimal cut sits between two
  // choices. Then noise alone flips it, and an ungated controller follows every flip
  // -- each one a re-download of the moved layers and a dropped K/V cache. With the
  // gates, the same noise moves nothing. That contrast is the evidence the gates do
  // the work; without it, "it settled" could just mean the noise was too small to
  // matter.
  const fine = layers(0, 48, {bytes: 1 * MB});
  const close = {a: 1.05e6, b: 1e6};
  const run = (gated) => {
    const rs = {a: new Rate(), b: new Rate()};
    let cur = null, moves = 0, t = 0, lastAt = 0;
    seed = 999;
    for (let round = 0; round < 300; round++) {
      t += 1000;
      const proposed = allocate(fine, ['a', 'b'].map(id =>
        device({id, label: id, rate: rs[id].rate(DEFAULT_RATE)})));
      let take = true;
      if (gated) {
        const d = shouldRebalance({current: cur, proposed, now: t, lastMoveAt: lastAt,
          trustedCount: Object.values(rs).filter(r => r.trusted()).length,
          deviceCount: 2});
        take = d.move;
      }
      if (take) {
        const changed = !cur || proposed.assign.some((a, i) =>
          a.start !== cur[i].start || a.end !== cur[i].end);
        if (changed) { moves++; lastAt = t; }
        cur = proposed.assign.map(a => ({...a}));
      }
      for (const a of cur) {
        rs[a.id].observe(a.bytes, (a.bytes / close[a.id]) * noise());
        a.rate = rs[a.id].rate(DEFAULT_RATE);
      }
    }
    return moves;
  };
  const ungated = run(false), gated = run(true);
  ok('two similar-speed devices: ungated, noise alone keeps moving layers',
     ungated > 10, `${ungated} moves in 300 tokens`);
  ok('  gated, the same noise moves almost nothing', gated <= 2,
     `${gated} moves in 300 tokens`);
  ok('  which is what "converges rather than oscillates" means',
     ungated > gated * 5, `${ungated} ungated vs ${gated} gated`);
}

// =========================================================================
console.log('\nlayerPlan reads bytes and the largest tensor out of a header:');
{
  const fake = {tensors: [
    {name: 'blk.0.attn_q.weight', bytes: 3 * MB},
    {name: 'blk.0.ffn_down.weight', bytes: 9 * MB},
    {name: 'blk.1.attn_q.weight', bytes: 3 * MB},
    {name: 'blk.1.ffn_down.weight', bytes: 7 * MB},
    {name: 'output.weight', bytes: 638 * MB},
    {name: 'token_embd.weight', bytes: 100 * MB},
  ]};
  const p = layerPlan(fake, 0, 1);
  ok('sums each layer\'s bytes', p[0].bytes === 12 * MB && p[1].bytes === 10 * MB);
  ok('finds each layer\'s largest tensor and names it',
     p[0].maxTensor === 9 * MB && p[0].biggest === 'blk.0.ffn_down.weight');
  ok('ignores tensors that are not in a block (output.weight, token_embd)',
     p.length === 2);
  let e = null;
  try { layerPlan(fake, 0, 5); } catch (err) { e = err; }
  ok('a missing layer is an error, not an empty plan', /no tensors for layer 2/.test(
     e?.message || ''), e?.message);
}

// =========================================================================
console.log('\ncurrentMakespan is the slowest stage, not the sum:');
ok('two 10ms stages cost 10ms, not 20',
   currentMakespan([{bytes: 10 * MB, rate: 1e6}, {bytes: 10 * MB, rate: 1e6}]) === 10);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
