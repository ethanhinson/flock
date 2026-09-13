// flock/allocate — decide which layers each device holds.
//
// WHY THIS IS NOT AN EVEN SPLIT. Three measurements make an even split the wrong
// answer, and each one is handled here:
//
//   1. Devices differ by ~5x. An iPhone did 4 layers in 15.4ms; this Mac did 24
//      layers in 19.6ms -- 3.85ms/layer versus 0.82ms/layer. A pipeline runs at
//      the pace of its SLOWEST stage, so giving both devices the same number of
//      layers wastes the fast one and the whole flock waits on the phone.
//
//   2. Layers are not the same size as each other. Qwen3-14B Q4_K_M layers range
//      185.8-210.2 MB, a 13% spread. "Half the layers" is therefore not "half the
//      bytes", so every budget and every weight here is in BYTES, never in layer
//      count.
//
//   3. Some tensors cannot be divided at all and are huge. That model's
//      output.weight is 638MB as a single tensor, and WebGPU's default
//      maxStorageBufferBindingSize is 128 MiB. A device whose binding limit is
//      below a layer's LARGEST TENSOR cannot hold that layer at any split, and the
//      only honest thing to do is say so before gigabytes are downloaded.
//      (Splitting such a tensor column-wise is another agent's work; this module
//      is written so a split tensor arrives as a smaller `maxTensor` and needs no
//      change here.)
//
// WHAT THE ALLOCATOR OPTIMISES. A pipeline's cost per token is its slowest stage,
// so the objective is MAKESPAN: minimise max over devices of (bytes assigned /
// that device's measured bytes-per-ms). Layers must stay CONTIGUOUS per device --
// the hidden state makes one lap in layer order, so a device holding 24 and 26
// would force two hops through it -- which makes the search a contiguous partition
// problem, solved exactly here by dynamic programming rather than approximated.
//
// Everything in this file is pure: no network, no GPU, no clock. That is what makes
// "a device with a 64MB binding limit never gets an oversized layer" testable
// without owning the device.

/** Each layer's total bytes and its single largest tensor, from a GGUF header. */
export function layerPlan(model, first, last) {
  const out = [];
  for (let i = first; i <= last; i++) {
    const ts = model.tensors.filter(t => t.name.startsWith(`blk.${i}.`));
    if (!ts.length) throw new Error(`model has no tensors for layer ${i}`);
    let bytes = 0, maxTensor = 0, biggest = ts[0].name;
    for (const t of ts) {
      bytes += t.bytes;
      if (t.bytes > maxTensor) { maxTensor = t.bytes; biggest = t.name; }
    }
    out.push({layer: i, bytes, maxTensor, biggest});
  }
  return out;
}

// A device that has never run a token has no measured speed, so it starts at this
// rate and its first samples correct it. The value is bytes-per-millisecond taken
// from the SLOWER of the two devices actually measured (a phone: 4 Qwen3-0.6B
// layers, ~16.8MB, in 15.4ms). Optimism here would hand a new phone a share it
// cannot carry and the whole flock would crawl for one token before the first
// rebalance corrected it; pessimism only costs the new device a small first share.
export const DEFAULT_RATE = 16.8e6 / 15.4;

/**
 * A device, as the allocator needs to see it.
 *
 * `bind` is maxStorageBufferBindingSize -- the hard per-tensor wall, and the one
 * limit no partition can work around. `budget` is how many weight bytes this
 * device is willing to hold in total. `rate` is measured bytes-per-millisecond;
 * higher means a bigger share.
 *
 * `reservedMs` is per-token time this device already owes to work the allocator
 * does not assign. The coordinator machine is the case that matters: when its
 * browser joins as a bird, the embedding, output_norm and LM head still run on
 * that same GPU, so its measured rate flatters it -- the allocator would hand the
 * fastest device the biggest layer share while being blind to the fact that it is
 * already busy. Excluding it instead would waste the best hardware on the
 * network, so the honest fix is to charge it for what it already does.
 */
export function device({id, label = '?', bind = Infinity, budget = Infinity,
                        rate = DEFAULT_RATE, reservedMs = 0} = {}) {
  return {id, label, bind: bind || Infinity, budget: budget || Infinity,
          rate: rate > 0 ? rate : DEFAULT_RATE,
          reservedMs: reservedMs > 0 ? reservedMs : 0};
}

/** Thrown when no assignment can satisfy every device's limits. */
export class Infeasible extends Error {
  constructor(message, detail) {
    super(message);
    this.name = 'Infeasible';
    this.detail = detail;
  }
}

const mb = n => n === Infinity ? 'unlimited' : `${(n / 1e6).toFixed(1)}MB`;

/**
 * Split `layers` across `devices`, weighting by measured speed and respecting
 * every device's limits.
 *
 * Devices keep the order they are given -- join order, in the server. Reordering
 * them is free in principle (the chain is just an ordering) but it is a factorial
 * search for a gain that does not exist on real models, where the layers in a
 * bird's range have near-identical largest tensors: if one device's binding limit
 * excludes one of them it excludes them all, and no permutation rescues it.
 * Keeping join order instead makes the assignment STABLE and explainable, which
 * matters more -- a human reading /status sees the order they joined in.
 *
 * Every device gets at least one layer. A device that can hold nothing is a
 * failure to report, not a member to hand an empty range: an empty range leaves it
 * in the chain forwarding frames it does no work on, which looks like a working
 * flock that is inexplicably slow.
 *
 * Returns {assign: [{id, start, end, layers, bytes, ms, share}], makespanMs, ...}.
 */
export function allocate(layers, devices) {
  const D = devices.length, L = layers.length;
  if (!D) throw new Infeasible('no devices in the flock', {kind: 'empty'});
  if (L < D) {
    throw new Infeasible(
      `${D} devices but only ${L} layer(s) to give out: ${D - L} device(s) would ` +
      `hold nothing. Give the birds more layers (FLOCK_BIRD_LAYERS) or use fewer ` +
      `devices.`,
      {kind: 'too-many-devices', devices: D, layers: L});
  }

  // Feasibility BEFORE the search, so the message names a cause rather than "no
  // solution". Neither of these two is fixable by any partition, and finding that
  // out here is the whole point -- it happens before a single weight byte moves.
  const prefix = [0];
  for (const l of layers) prefix.push(prefix[prefix.length - 1] + l.bytes);
  for (const l of layers) {
    if (devices.some(d => l.maxTensor <= d.bind)) continue;
    const best = devices.reduce((a, b) => (b.bind > a.bind ? b : a));
    throw new Infeasible(
      `no device can hold layer ${l.layer}: its largest tensor ${l.biggest} is ` +
      `${mb(l.maxTensor)}, and the roomiest device in the flock (${best.label}) ` +
      `allows ${mb(best.bind)} per tensor. One tensor cannot be split across ` +
      `devices, so adding more devices will not help -- this needs a device with a ` +
      `bigger maxStorageBufferBindingSize, a more quantized model, or column-wise ` +
      `tensor splitting.`,
      {kind: 'tensor', layer: l.layer, tensor: l.biggest,
       needs: l.maxTensor, has: best.bind, device: best.id});
  }
  const totalBytes = prefix[L];
  const totalBudget = devices.reduce((a, d) => a + d.budget, 0);
  if (totalBytes > totalBudget) {
    throw new Infeasible(
      `the flock is ${mb(totalBytes - totalBudget)} short: layers ` +
      `${layers[0].layer}-${layers[L - 1].layer} are ${mb(totalBytes)} of weights ` +
      `but the ${D} device(s) together budget only ${mb(totalBudget)}. Add a ` +
      `device, or use a smaller model.`,
      {kind: 'budget', needs: totalBytes, has: totalBudget});
  }

  // Exact contiguous partition by DP. best[d][i] is the smallest achievable
  // makespan for layers i.. using devices d.., and cut[d][i] remembers where
  // device d's run ended so the assignment can be walked back out.
  //
  // Minimising the MAX stage rather than the total is what makes this the right
  // objective for a pipeline: 10 layers on a phone and 2 on a laptop has a better
  // total than the reverse and is much worse per token.
  const INF = Infinity;
  const best = Array.from({length: D + 1}, () => new Float64Array(L + 1).fill(INF));
  const cut = Array.from({length: D + 1}, () => new Int32Array(L + 1).fill(-1));
  best[D][L] = 0;
  for (let d = D - 1; d >= 0; d--) {
    const dev = devices[d];
    const room = D - 1 - d;          // layers that must be left for later devices
    for (let i = 0; i + room < L; i++) {
      let bytes = 0;
      for (let j = i; j + room < L; j++) {
        const layer = layers[j];
        // Both breaks are sound because a longer run only adds: once one tensor is
        // too big or the running total passes the budget, no longer run can fit.
        if (layer.maxTensor > dev.bind) break;
        bytes += layer.bytes;
        if (bytes > dev.budget) break;
        const tail = best[d + 1][j + 1];
        if (tail === INF) continue;
        // A stage costs everything that device spends on this token, including
        // work it owes elsewhere (see `reservedMs`) -- otherwise the busiest
        // device looks like the cheapest place to add more.
        const span = Math.max(bytes / dev.rate + dev.reservedMs, tail);
        if (span < best[d][i]) { best[d][i] = span; cut[d][i] = j; }
      }
    }
  }

  if (best[0][0] === INF) {
    // Every layer fits SOMEWHERE and the bytes fit in total, so what is left is a
    // CONTIGUITY conflict: the device that could hold a layer is not in a chain
    // position a contiguous run reaches. Name the tightest device -- the one to
    // drop or grow.
    const tight = devices.reduce((a, b) => (b.bind < a.bind ? b : a));
    const blocked = layers.filter(l => l.maxTensor > tight.bind).map(l => l.layer);
    throw new Infeasible(
      `no contiguous assignment fits: each device's layers must stay in order, and ` +
      `${tight.label} (${mb(tight.bind)} per tensor` +
      `${tight.budget === Infinity ? '' : `, ${mb(tight.budget)} budget`}) cannot ` +
      `take ${blocked.length ? `layer(s) ${blocked.join(', ')}` : 'enough layers'} ` +
      `from the position it holds in the chain. Drop that device or raise its budget.`,
      {kind: 'contiguity', device: tight.id, blocked});
  }

  const assign = [];
  let i = 0;
  for (let d = 0; d < D; d++) {
    const end = cut[d][i];
    const run = layers.slice(i, end + 1);
    const bytes = run.reduce((a, l) => a + l.bytes, 0);
    assign.push({
      id: devices[d].id, label: devices[d].label,
      start: run[0].layer, end: run[run.length - 1].layer,
      layers: run.length, bytes, rate: devices[d].rate,
      ms: +(bytes / devices[d].rate).toFixed(2),
      share: +(bytes / totalBytes).toFixed(4),
    });
    i = end + 1;
  }
  // The even-by-layer-count split's makespan, for comparison. /status reports both
  // so a human can see what the weighting actually bought rather than trusting it.
  let evenMs = 0;
  const per = Math.floor(L / D), extra = L % D;
  for (let d = 0, at = 0; d < D; d++) {
    const k = per + (d < extra ? 1 : 0);
    evenMs = Math.max(evenMs, (prefix[at + k] - prefix[at]) / devices[d].rate);
    at += k;
  }
  return {assign, makespanMs: +best[0][0].toFixed(2), totalBytes,
          evenMs: +evenMs.toFixed(2)};
}

/** One line saying why a device got the share it did, for /status and the log. */
export function explain(a) {
  return `${a.layers} layer${a.layers === 1 ? '' : 's'} (${a.start}-${a.end}), ` +
         `${mb(a.bytes)} = ${(a.share * 100).toFixed(0)}% of the bird bytes, at ` +
         `${(a.rate / 1e6).toFixed(2)}MB/ms -> ${a.ms}ms`;
}

export {mb as formatBytes};
