// flock/speed — turn per-token timings into a rate, and decide when to act on it.
//
// The timings already existed (bird.lastMs, reported over the websocket as
// `stats`). What did not exist was any use of them. This module is the policy that
// makes them mean something, and it is separate from the allocator on purpose: the
// allocator answers "given these rates, who holds what", this answers "do we
// believe the rates enough to move layers".
//
// THE FAILURE MODE THIS IS BUILT AGAINST IS OSCILLATION, not slowness. A naive
// loop -- measure, reallocate, measure, reallocate -- thrashes, because moving
// layers CHANGES the measurement that caused the move. A phone that looks slow
// loses layers, then holds fewer layers and looks fast, then gains them back, and
// every swing costs a full re-download of the moved layers' weights plus a dropped
// K/V cache. Four independent brakes, each aimed at a different way that happens:
//
//   1. RATE, NOT TIME. A device is tracked by bytes-per-millisecond, which is
//      invariant to how many layers it holds. A raw ms reading is not: halve a
//      device's layers and its ms halves, so a ms-based controller would read its
//      own last decision as new evidence and chase it forever.
//
//   2. EWMA over samples. One slow token -- a GC pause, a wifi hiccup, the user
//      switching apps -- moves the estimate by ALPHA, not all the way.
//
//   3. MIN_SAMPLES before a device's measured rate is trusted at all. Until then
//      it keeps the default rate, so the first token after a join cannot trigger a
//      reshuffle.
//
//   4. A GAIN GATE plus a COOLDOWN. A reallocation has to promise to cut the
//      predicted makespan by at least MIN_GAIN, and cannot happen more often than
//      COOLDOWN_MS. The gain gate is what makes this converge rather than settle
//      into a limit cycle: near the optimum every candidate move is a small gain,
//      every small gain is refused, and the assignment stops moving.
//
// Deterministic and clock-injectable, so the convergence claim is a test rather
// than a hope.

/** How much of a new sample moves the estimate. Low, because a phone's slow token
 *  is much more often noise than a lasting change. */
export const ALPHA = 0.25;

/** Samples before a device's own rate is used instead of the default. Three is
 *  enough to outvote one outlier and small enough to adapt within a short reply. */
export const MIN_SAMPLES = 3;

/** Predicted-makespan improvement a move must promise. 15% is above the run-to-run
 *  noise measured on these timings (a phone's per-token ms varies by a few
 *  percent) and well below the ~5x spread between a phone and a laptop, so real
 *  imbalance is acted on and noise is not. */
export const MIN_GAIN = 0.15;

/** Minimum wall time between reallocations. A move costs the moved layers' weights
 *  a re-download and the whole flock its K/V cache, so this is deliberately long
 *  relative to a token. */
export const COOLDOWN_MS = 20000;

/** Tracks one device's throughput in bytes per millisecond. */
export class Rate {
  constructor({alpha = ALPHA, minSamples = MIN_SAMPLES} = {}) {
    this.alpha = alpha;
    this.minSamples = minSamples;
    this.samples = 0;
    this.value = null;      // null until measured at all
    this.lastMs = null;
    this.lastBytes = null;
  }

  /**
   * Record one token: this device did `bytes` of weights in `ms`.
   *
   * `bytes` is passed in rather than derived from a layer count because layers are
   * not the same size -- the 13% spread across Qwen3-14B layers would otherwise
   * show up as a 13% speed difference that is really a size difference.
   */
  observe(bytes, ms) {
    if (!(ms > 0) || !(bytes > 0)) return this;     // a zero is a lost frame
    const r = bytes / ms;
    this.value = this.value == null ? r : this.value + this.alpha * (r - this.value);
    this.samples++;
    this.lastMs = ms;
    this.lastBytes = bytes;
    return this;
  }

  /** The rate to plan with: measured once there is enough of it, else `fallback`. */
  rate(fallback) {
    return this.trusted() ? this.value : fallback;
  }

  trusted() { return this.samples >= this.minSamples && this.value > 0; }

  /** Forget the measurement but not that the device exists. Called when a device's
   *  range changes: the old samples described a different amount of work, and while
   *  a RATE is supposed to be size-invariant, a device whose thermal or memory
   *  behaviour changed with the new share should be re-learned rather than trusted. */
  reset() { this.samples = 0; this.value = null; return this; }
}

/**
 * Should we move layers right now?
 *
 * Pure: takes the current assignment, the proposed one and the clock, and returns
 * a decision with the REASON, which /status shows verbatim. Refusing is the common
 * case and the reason for refusing is the interesting part -- without it the only
 * observable behaviour of the whole mechanism is "nothing happened".
 */
export function shouldRebalance({current, proposed, now, lastMoveAt = 0,
                                 trustedCount = 0, deviceCount = 0,
                                 minGain = MIN_GAIN, cooldownMs = COOLDOWN_MS} = {}) {
  const no = (reason, extra = {}) => ({move: false, reason, ...extra});
  if (!proposed) return no('nothing proposed');
  // Membership changes are NOT gated: a device that joined or left must be given or
  // relieved of layers immediately or the flock does not cover them at all. The
  // gates below are only about moving layers between a stable set of devices.
  const sameSet = current &&
    current.length === proposed.assign.length &&
    current.every((c, i) => c.id === proposed.assign[i].id);
  if (!sameSet) return {move: true, reason: 'membership changed', gain: null};
  const same = current.every((c, i) => c.start === proposed.assign[i].start &&
                                       c.end === proposed.assign[i].end);
  if (same) return no('already optimal for the rates we have');
  if (trustedCount < deviceCount) {
    return no(`waiting for timings: ${trustedCount}/${deviceCount} devices have ` +
              `${MIN_SAMPLES}+ samples`);
  }
  const since = now - lastMoveAt;
  if (lastMoveAt && since < cooldownMs) {
    return no(`cooling down: ${((cooldownMs - since) / 1000).toFixed(0)}s to go`);
  }
  const nowMs = currentMakespan(current);
  const gain = nowMs > 0 ? (nowMs - proposed.makespanMs) / nowMs : 0;
  if (gain < minGain) {
    return no(`gain too small: ${(gain * 100).toFixed(1)}% predicted, need ` +
              `${(minGain * 100).toFixed(0)}%`, {gain});
  }
  return {move: true, gain,
          reason: `predicted ${nowMs.toFixed(1)}ms -> ${proposed.makespanMs}ms ` +
                  `(${(gain * 100).toFixed(0)}% faster)`};
}

/** The slowest stage of an assignment: what a token actually costs. */
export function currentMakespan(assign) {
  return assign.reduce((m, a) => Math.max(m, a.bytes / a.rate), 0);
}
