// flock mesh — the flock of birds, and the links between them.
//
// Every link is a WebRTC data channel: coordinator <=> bird, and bird <=> bird.
// The websocket exists ONLY to carry offers/answers/ICE. Once a channel opens,
// no activation touches it again.
//
// MEMBERSHIP IS DYNAMIC. This used to be a fixed array of N slots built from
// FLOCK_BIRDS at startup: a third phone pointed at a two-slot coordinator got
// "flock full" and the only way to change N was a restart. Now the flock is a LIST
// that grows and shrinks, and the layer assignment is recomputed from the devices
// actually present -- by the allocator in ./allocate.js, weighted by each device's
// measured speed and constrained by its real GPU limits.
//
// `slot` survives as the chain POSITION rather than a reservation: it is an index
// into the current ordering, reassigned on every membership change, and the bird
// page and chat page both key their DOM off it. Ordering is join order, which the
// allocator relies on being stable.
import nodeDataChannel from 'node-datachannel';
// Buffer is an implicit global under Node but not under Deno, and the server runs
// under Deno because the coordinator needs a GPU. Imported rather than replaced:
// node-datachannel's sendMessageBinary and ws both want a real Buffer, so the
// transport keeps using exactly what it used before.
import {Buffer} from 'node:buffer';
import {pack, unpack} from '../web/js/wire.mjs';
import {allocate, adjust, device, Infeasible, explain, DEFAULT_RATE} from './allocate.js';
import {Rate, shouldRebalance, currentMakespan, MIN_SAMPLES} from './speed.js';

const ICE = [];  // LAN-only: host ICE candidates are directly reachable, no STUN server needed

/** What a device reported about itself, normalised: missing means unlimited.
 *  Exported because the session token is bound to these numbers, so the server
 *  has to normalise a re-attaching device's report the same way. */
export function normCaps(caps) {
  const num = v => (typeof v === 'number' && v > 0 && Number.isFinite(v)) ? v : null;
  const bind = num(caps?.maxStorageBufferBindingSize ?? caps?.bind);
  const budget = num(caps?.budget ?? caps?.maxBufferSize);
  return {
    bind: bind ?? Infinity,
    // A device's whole-weights budget is not a limit WebGPU reports, so it comes
    // from maxBufferSize when nothing better is offered. That is generous (it is
    // one buffer's ceiling, not the device's memory) but it is the only number a
    // browser will tell us, and the alternative -- guessing from the user agent --
    // was what produced the iPad OOM this design is trying to stop repeating.
    budget: budget ?? Infinity,
    cores: num(caps?.cores) ?? null,
    gpu: caps?.vendor || caps?.gpu || null,
    secure: caps?.secureContext ?? null,
    compute_ok: caps?.computeOk ?? null,
  };
}

export class Bird {
  constructor(peerId, label = '?') {
    this.peerId = peerId; this.label = label;
    // No range until the allocator gives one. A bird with start === null is a
    // member that has been admitted but not yet placed, which is a real state now
    // that /join no longer hands out a pre-decided slot.
    this.start = null; this.end = null; this.slot = -1;
    this.lastSeen = 0; this.lastMs = null; this.claimedAt = 0;
    this.frames = 0;
    this.transport = 'none';
    this.forwardsDirectly = false;   // set when the bird reports a live peer link
    // PAUSED: the device said it is going away (a phone going to the background,
    // where iOS suspends JS, sockets and WebGPU alike) and will be back. It stays
    // a member but holds no layers, so the others take its share at once rather
    // than after the liveness grace; when it comes back it re-attaches with its
    // session and is placed again. See Flock.pause().
    this.paused = false; this.pausedAt = 0;
    this.ws = null; this.pc = null; this.chan = null;
    this._waiter = null;
    // THE READINESS HANDSHAKE. Being assigned a range and HOLDING it are different
    // states separated by a weight stream of 3-18 seconds on a phone, and the gap
    // used to be invisible: the coordinator assigned, announced, and sent the next
    // frame at once. `confirmed` is the range this device has said it built, sent
    // as {t:'ready', start, end} once its layers are on the GPU; a lap does not
    // start until every placed bird's confirmation matches its assignment.
    this.confirmed = null;
    // Consecutive frames this device failed to compute (a NACK with reason
    // 'failed'). A device that cannot compute is a device that cannot serve, and
    // that -- not a transient frame during a transition -- is what removes it.
    this.failures = 0;

    // What this device said it can do, from the /check probe it runs before
    // joining. Unknown means unconstrained: a device that did not report is
    // trusted, because refusing it would be worse than the OOM it might hit, and
    // the OOM is reported through /diag either way.
    this.caps = {bind: Infinity, budget: Infinity, cores: null, gpu: null};
    // True when this bird's browser is on the coordinator machine, so its GPU is
    // shared with the embedding and the LM head. Set from the join request's
    // source address -- loopback means same machine.
    this.onCoordinator = false;
    // Its measured throughput, and the bytes it is currently responsible for --
    // the pair is what makes a rate rather than a raw time. See ./speed.js.
    this.rate = new Rate();
    this.bytes = 0;
    // Why it got the share it has, in one line, for /status.
    this.why = 'not placed yet';
  }

  alive(grace = 40000) {
    const linked = !!(this.chanOpen() || this.ws);
    // Latch the fact that this device was once really connected. claimed() needs it
    // to tell "still arriving over HTTP" (hold it, its layers are on the way) from
    // "arrived and then went quiet" (let it go, its layers are uncovered).
    if (linked) this.everLinked = true;
    return linked && (Date.now() - this.lastSeen) < grace;
  }

  /** Is this device still a member? Alive, or admitted over HTTP and still
   *  connecting for the first time.
   *
   *  /join happens over plain HTTP, so a just-admitted device has no websocket yet
   *  and so is not yet `alive()`. It still has to be counted as a member, or the
   *  allocator would give its layers away and take them back a second later, once
   *  per joining phone. (In the fixed-slot design the same window stopped several
   *  simultaneous joins from all being handed slot 0.)
   *
   *  The window applies ONLY before the device has ever been heard from. A device
   *  that connected and then went quiet is not "still connecting", and holding it as
   *  a member on the strength of its /join would keep its layers uncovered for an
   *  extra 15s on top of the liveness grace period. */
  claimed(hold = 15000) {
    if (this.alive()) return true;
    if (this.everLinked) return false;      // it connected once; this is not that window
    return !!this.peerId && Date.now() - this.claimedAt < hold;
  }

  chanOpen() {
    try { return !!(this.chan && this.chan.isOpen()); } catch { return false; }
  }

  /** Is it assigned layers right now? A member between reallocations may not be. */
  placed() { return this.start != null && this.end != null; }

  /** Has it confirmed that it holds the range it is assigned? */
  isReady() {
    return this.placed() && !!this.confirmed &&
           this.confirmed.start === this.start && this.confirmed.end === this.end;
  }

  /** The device says its layers for [start, end] are built and on the GPU. A
   *  confirmation for a range it is no longer assigned is kept -- it is true, just
   *  stale -- and simply does not count until the assignment matches again. */
  confirm(start, end) {
    if (start == null || end == null) { this.confirmed = null; return false; }
    this.confirmed = {start, end};
    this.failures = 0;
    return this.isReady();
  }

  /** The device refused a frame. `reason` is 'not-ready' (it is between
   *  assignments, or still streaming) or 'failed' (it tried and could not). Wakes
   *  the lap waiting on it, so the coordinator learns now rather than at the 120s
   *  timeout. */
  nack(m) {
    this.lastSeen = Date.now();
    if (m.reason === 'failed') this.failures++;
    if (this._waiter) this._waiter({nack: m});
  }

  info() {
    return {slot: this.slot, start: this.start, end: this.end,
            n_layers: this.placed() ? this.end - this.start + 1 : 0,
            alive: this.alive(), last_ms: this.lastMs,
            paused: this.paused,
            // Assigned is not holding: a bird re-streams for seconds after a
            // reassignment, and this is where /status says which ones still are.
            ready: this.isReady(),
            confirmed: this.confirmed ? `${this.confirmed.start}-${this.confirmed.end}` : null,
            failures: this.failures,
            label: this.label, transport: this.transport,
            frames: this.frames, peer_id: this.peerId,
            // How long since we last heard anything: the UI can say "12s ago"
            // rather than a bare alive/dead flag that hides a stalling device.
            last_seen_ms: this.lastSeen ? Date.now() - this.lastSeen : null,
            forwards_directly: this.forwardsDirectly,
            // Why this device got the share it did. The whole point of a
            // speed-weighted split is that a human can tell an unfair-looking
            // assignment from a correctly-measured one, and that needs numbers.
            bytes: this.bytes,
            mb: +(this.bytes / 1e6).toFixed(1),
            rate_mb_per_ms: this.rate.value == null
              ? null : +(this.rate.value / 1e6).toFixed(3),
            rate_samples: this.rate.samples,
            rate_trusted: this.rate.trusted(),
            max_binding_mb: this.caps.bind === Infinity
              ? null : +(this.caps.bind / 1e6).toFixed(0),
            budget_mb: this.caps.budget === Infinity
              ? null : +(this.caps.budget / 1e6).toFixed(0),
            why: this.why};
  }

  /** Push into this bird but wait for `awaitOn` to answer (the chain's tail).
   *  When this bird IS the tail, `awaitOn` is itself — so this one method covers
   *  both the chained and the single-bird case.
   *
   *  `flags` (reset, replay) are the lap's, not this bird's: every head of a
   *  direct run gets them, and a bird that forwards directly passes them on. A
   *  per-bird "reset pending" flag used to live here, and it only ever reached
   *  the birds the coordinator sent to itself -- a bird fed by its predecessor's
   *  data channel never saw the reset, and its cache kept growing. */
  sendChained(floats, seq, hidden, offset, awaitOn, flags = {}, timeout = 120000) {
    const frame = pack(floats, {seq, hidden, offset, reset: !!flags.reset, replay: !!flags.replay});
    const viaRTC = this.chanOpen();
    const link = viaRTC ? this.chan : this.ws;
    if (!link) return Promise.reject(new Error(
      `no device holding layers ${this.start}-${this.end}`));
    this.transport = viaRTC ? 'webrtc' : 'ws';
    return new Promise((res, rej) => {
      const timer = setTimeout(() => {
        awaitOn._waiter = null;
        rej(new Error(`the flock stopped responding — is every device awake ` +
                      `with its tab in front?`));
      }, timeout);
      awaitOn._waiter = v => { clearTimeout(timer); awaitOn._waiter = null; res(v); };
      if (viaRTC) this.chan.sendMessageBinary(Buffer.from(frame));
      else this.ws.send(Buffer.from(frame));
    });
  }

  deliver(buf) {
    this.lastSeen = Date.now();
    this.frames++;
    this.failures = 0;
    const {data, meta} = unpack(
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    if (this._waiter) this._waiter({data, meta});
  }

  /** Drop every link to this bird. Called when its websocket goes away, so a
   *  refresh starts from a clean slate instead of leaking a PeerConnection. */
  teardown() {
    try { this.chan?.close(); } catch {}
    try { this.pc?.close(); } catch {}
    this.chan = null; this.pc = null; this.ws = null;
    this.transport = 'none';
    if (this._waiter) { const w = this._waiter; this._waiter = null;
      try { w(null); } catch {} }
  }

  /** Offer a direct data channel to this bird, signalled over its websocket. */
  openRTC(onOpen) {
    // A reloaded bird signals again; drop the previous connection or stale
    // PeerConnections accumulate and the wrong one wins.
    try { this.chan?.close(); } catch {}
    try { this.pc?.close(); } catch {}
    this.chan = null; this.pc = null;
    this.pc = new nodeDataChannel.PeerConnection(`coord-${this.slot}`, {iceServers: ICE});
    this.pc.onLocalDescription((sdp, type) =>
      this.ws?.send(JSON.stringify({t: 'signal', data: {kind: type, sdp: {sdp, type}}})));
    this.pc.onLocalCandidate((candidate, mid) =>
      this.ws?.send(JSON.stringify({t: 'signal', data: {kind: 'ice', candidate, mid}})));
    this.chan = this.pc.createDataChannel('flock');
    this.chan.onOpen(() => {
      this.transport = 'webrtc';
      onOpen?.(this);
    });
    // node-datachannel delivers both text and binary through onMessage.
    this.chan.onMessage(msg => {
      if (typeof msg === 'string') return;
      this.deliver(Buffer.isBuffer(msg) ? msg : Buffer.from(msg));
    });
    this.chan.onClosed(() => { this.transport = this.ws ? 'ws' : 'none'; this.chan = null; });
  }

  onSignal(d) {
    if (!this.pc) return;
    if (d.kind === 'answer' || d.kind === 'offer') {
      this.pc.setRemoteDescription(d.sdp.sdp, d.sdp.type);
    } else if (d.kind === 'ice') {
      try { this.pc.addRemoteCandidate(d.candidate, d.mid || '0'); } catch {}
    }
  }
}


/**
 * The flock: a dynamic set of devices, and the layer assignment over them.
 *
 * Construct it with the LAYER PLAN (from allocate.layerPlan), not with ranges.
 * Ranges are an output now -- recomputed from whoever is present -- which is the
 * whole difference from the fixed-slot version: there is no N to be full at.
 */
export class Flock {
  /**
   * @param {Array} layers  layer descriptors covering the birds' share of the model
   * @param {object} opts   `now` for tests, `onPlan` called after every successful
   *                        reallocation with the new assignment
   */
  constructor(layers, {now = () => Date.now(), onPlan = null,
                       pauseGraceMs = 2500, pauseMaxMs = 10 * 60 * 1000} = {}) {
    this.layers = layers;
    this.birds = [];            // chain order; index === slot
    this.now = now;
    this.onPlan = onPlan;
    // A pause takes effect after `pauseGraceMs`: a page REFRESH also fires the
    // events that send a pause, and the same device is back with its session
    // within a second or two. Reallocating in that window would move layers for
    // a device that is about to hold exactly what it held. A device that is
    // really gone to the background stays paused past the grace, and its share
    // is handed on then -- seconds, not the 40s liveness grace.
    // A paused device is kept as a member for `pauseMaxMs` (the session's
    // lifetime); one that never comes back is dropped after that.
    this.pauseGraceMs = pauseGraceMs;
    this.pauseMaxMs = pauseMaxMs;
    // The last reason a reallocation was made or refused, verbatim in /status. A
    // speed-weighted split whose decisions are invisible is indistinguishable from
    // a broken one, so this is part of the feature and not debug output.
    this.decision = {at: 0, moved: false, reason: 'no devices yet', gain: null};
    this.lastMoveAt = 0;
    this.makespanMs = null;
    this.evenMs = null;
    // Set when the devices present cannot hold the model. Kept rather than thrown
    // so /status and /chat can both report the same precise reason instead of the
    // server dying on a phone's join.
    this.infeasible = null;
    // Membership changes that arrived mid-token, applied at the next token
    // boundary. See applyPending().
    this.deferred = false;
  }

  /**
   * Devices that count for allocation.
   *
   * The set is deliberately WIDER than "answering right now": it is every device the
   * sweeper has not given up on. A refresh drops the websocket and comes back within a
   * second or two with the same peer id, so excluding it the moment its socket closes
   * would reallocate -- and therefore drop the conversation -- for what is about to be
   * the same device holding the same range. So membership survives a brief silence,
   * and only sweep() ends it.
   *
   * The cost of the wider set is that a genuinely dead device keeps its layers for up
   * to `grace`, during which ready() is false and /status names the layers nobody is
   * answering for. Reporting the gap is the right trade: the alternative is thrashing
   * the whole flock on every phone that locks its screen for a second.
   */
  members(grace = 40000) {
    return this.birds.filter(b => !this.pauseInEffect(b) &&
                                  (b.claimed() || (Date.now() - b.lastSeen) < grace));
  }

  /** Has this device's pause outlasted the refresh window, so its layers should
   *  go to the others? Before that a paused device is still allocated to. */
  pauseInEffect(b) {
    return b.paused && (Date.now() - b.pausedAt) >= this.pauseGraceMs;
  }

  /** A paused device that has been away longer than a session lasts is not
   *  coming back on this membership. */
  pauseLapsed(b) {
    return b.paused && (Date.now() - b.pausedAt) >= this.pauseMaxMs;
  }

  /**
   * The device is going away and says so. It keeps its membership and, for the
   * refresh window, its layers; after that the allocator treats it as absent and
   * hands its share to its neighbours. Returns false if it was not a member.
   */
  pause(peerId) {
    const b = this.byPeer(peerId);
    if (!b) return false;
    if (!b.paused) { b.paused = true; b.pausedAt = Date.now(); }
    return true;
  }

  /** The device is back. It is a member holding whatever it was left with (nothing,
   *  once the pause took effect); the caller re-plans to place it again. */
  resume(peerId) {
    const b = this.byPeer(peerId);
    if (!b) return false;
    const was = b.paused;
    b.paused = false; b.pausedAt = 0;
    b.lastSeen = Date.now();
    return was;
  }

  /**
   * Admit a device, or re-admit one that already has this peer id.
   *
   * Never refuses. The fixed-slot version answered "flock full" once N devices had
   * joined, which is exactly the behaviour being removed: a device that turns up is
   * capacity, and what it changes is the assignment, not whether it is allowed in.
   *
   * `caps` is what the device reported from the /check probe: bind is
   * maxStorageBufferBindingSize, budget is how many weight bytes it will hold.
   */
  claim(peerId, label, caps = null, opts = {}) {
    let b = this.byPeer(peerId);
    if (!b) {
      b = new Bird(peerId, label);
      this.birds.push(b);
    } else if (label) {
      b.label = label;
    }
    if (caps) this.setCaps(b, caps);
    if (opts.onCoordinator !== undefined) b.onCoordinator = !!opts.onCoordinator;
    // Date.now(), NOT this.now(). The two are the same clock in production, but they
    // are different CLOCKS: `now` is injectable so a test can fast-forward the
    // rebalancer's cooldown, while LIVENESS is judged against real wall time by
    // alive(), claimed() and members(), and by Bird.deliver() when a frame arrives.
    // Writing this field from the injectable one would give it two writers on
    // different time bases, and a test that jumps the cooldown forward would also
    // declare every device either decades stale or impossibly fresh.
    b.lastSeen = Date.now();
    b.claimedAt = Date.now();
    return b;
  }

  /** Record what a device says it can do, normalising missing fields to unlimited. */
  /** Per-token cost of the coordinator's own layers, smoothed like a bird's rate.
   *
   *  Used as `reservedMs` for a bird on this machine. Smoothed because one slow
   *  token should not move an allocation -- the same reason bird rates use an EWMA.
   */
  noteCoordMs(ms) {
    if (!(ms > 0)) return;
    this.coordMs = this.coordMs == null ? ms : this.coordMs * 0.75 + ms * 0.25;
  }

  setCaps(b, caps) {
    b.caps = normCaps(caps);
    return b;
  }

  /** Remove a device by peer id. Returns true if it was a member. */
  release(peerId) {
    const i = this.birds.findIndex(b => b.peerId === peerId);
    if (i < 0) return false;
    try { this.birds[i].teardown(); } catch {}
    this.birds.splice(i, 1);
    return true;
  }

  /** Drop every device we have not heard from inside the grace period.
   *
   *  A fixed slot could just go red and wait for its owner to come back. A member
   *  cannot: it is holding layers nobody is running, so the flock stays uncovered
   *  until it is actually removed and its layers are given to someone else.
   *
   *  The test is LAST HEARD FROM, not "has a link". A refresh drops the websocket and
   *  comes back within a second or two with the same peer id, and removing it on the
   *  close would reallocate -- and so drop the conversation -- for what is about to
   *  be the same device holding the same range. So a device keeps its layers until it
   *  has been silent for `grace`, whether or not its socket is still there. */
  sweep(grace = 40000) {
    // The exact complement of members(), so the two cannot disagree about who is in
    // the flock -- a device the sweeper keeps but the allocator ignores would hold
    // layers nobody ever gives away. The one addition is a PAUSED device: the
    // allocator ignores it (it holds nothing) and the sweeper keeps it, until its
    // session would have lapsed, because it said it was coming back.
    const keep = new Set(this.members(grace));
    for (const b of this.birds) if (b.paused && !this.pauseLapsed(b)) keep.add(b);
    const gone = this.birds.filter(b => !keep.has(b));
    if (!gone.length) return [];
    for (const b of gone) this.release(b.peerId);
    // Re-plan HERE, not only in the caller. Removing a device leaves its layers
    // assigned to nobody, so a sweep that does not re-plan leaves the flock not-ready
    // with a hole in the chain -- correct only for as long as every caller remembers to
    // rebalance afterwards. Announcing is still the caller's job: it owns the
    // conversation state that a move invalidates.
    this.plan({force: true});
    return gone.map(b => b.peerId);
  }

  /**
   * Recompute the assignment for whoever is present.
   *
   * Two kinds of change, and they are handled differently on purpose:
   *
   *   `force`   membership changed. The alternative to moving is not covering
   *             some layers, so the gain gate and cooldown do not apply -- but the
   *             move is a STICKY EDIT (allocate.adjust): a joiner takes a run off
   *             one incumbent, a leaver's run goes to its neighbours, and nobody
   *             else moves. Every device that moves re-streams its weights and
   *             holds nothing meanwhile, so the number of movers is the cost.
   *
   *   not       the timings say the split could be better. This is the exact
   *             search, under the gain gate and the cooldown in ./speed.js. The
   *             caller decides WHEN this is allowed to happen (see the server:
   *             only at the start of a conversation, never during one).
   *
   * Returns the decision: {move, reason, gain, moved: [peerIds]}. Never throws for
   * an unsatisfiable flock; that lands in this.infeasible, where /status and /chat
   * both read it.
   */
  plan({force = false} = {}) {
    const members = this.members();
    // The exact search keeps the order it is given, so give it the CURRENT chain
    // order (placed birds by layer, then the unplaced in join order) rather than
    // join order: a speed rebalance should refine the split, not reshuffle who is
    // next to whom.
    const ordered = [...this.birds].sort((x, y) => {
      if (x.placed() && y.placed()) return x.start - y.start;
      return (x.placed() ? 0 : 1) - (y.placed() ? 0 : 1);
    }).filter(b => members.includes(b));
    const current = this.birds.filter(b => b.placed()).map(b => ({
      id: b.peerId, start: b.start, end: b.end, bytes: b.bytes,
      rate: b.rate.rate(DEFAULT_RATE),
    }));
    if (!members.length) {
      this.unplaceAll();
      this.infeasible = null;
      this.makespanMs = this.evenMs = null;
      return this.note(false, 'no devices in the flock');
    }

    let proposed;
    try {
      const devices = ordered.map(b => device({
        id: b.peerId, label: b.label, bind: b.caps.bind, budget: b.caps.budget,
        rate: b.rate.rate(DEFAULT_RATE),
        // A bird running in a browser ON the coordinator machine shares that GPU
        // with the embedding, output_norm and LM head. Its measured rate does not
        // know that, so without this the fastest device on the network gets the
        // biggest layer share while already being the busiest. Charging it the
        // coordinator's own measured per-token cost keeps it in the flock -- the
        // point is to use spare capacity, not to exclude the best hardware -- but
        // stops it being double-counted.
        reservedMs: b.onCoordinator ? (this.coordMs || 0) : 0,
      }));
      proposed = force ? adjust(this.layers, current, devices)
                       : allocate(this.layers, devices);
      this.infeasible = null;
    } catch (e) {
      if (!(e instanceof Infeasible)) throw e;
      // The devices present cannot hold the model. Take the layers away rather than
      // leaving a stale assignment that looks covered: /chat must refuse with this
      // reason, not start a turn against an assignment nobody can serve.
      this.infeasible = {message: e.message, detail: e.detail || null};
      this.unplaceAll();
      this.makespanMs = this.evenMs = null;
      return this.note(false, e.message);
    }

    const trusted = members.filter(b => b.rate.trusted()).length;
    const d = force
      ? {move: true, gain: null,
         reason: proposed.sticky ? 'membership changed (local edit)'
                                 : 'membership changed'}
      : shouldRebalance({
          current, proposed, now: this.now(), lastMoveAt: this.lastMoveAt,
          trustedCount: trusted, deviceCount: members.length,
        });
    if (!d.move) {
      // Keep the numbers even when refusing: /status shows what the move WOULD
      // have bought next to why it was not made, which is the only way a human can
      // tell "the gate is working" from "the gate is stuck".
      this.proposedMs = proposed.makespanMs;
      return this.note(false, d.reason, d.gain);
    }

    const moved = [];
    for (const a of proposed.assign) {
      const b = this.byPeer(a.id);
      if (!b) continue;
      if (b.start !== a.start || b.end !== a.end) {
        moved.push(b.peerId);
        // A device whose range changed has to re-download the layers it gained and
        // its K/V cache is stale for the ones it lost, so its old timings describe
        // work it is no longer doing. It is also no longer READY: whatever it
        // confirmed was the old range, and nothing may be sent to it until it
        // confirms the new one. Cleared explicitly, not left to the range
        // comparison: a device moved BACK to a range it confirmed earlier would
        // otherwise look ready while it is still streaming -- seen once, as a
        // replay frame arriving at a device holding nothing.
        b.rate.reset();
        b.confirmed = null;
      }
      b.start = a.start; b.end = a.end;
      b.bytes = a.bytes;
      b.why = explain(a);
    }
    // Chain order is allocation order, which is join order: index === slot, and the
    // hidden state passes through the birds in increasing layer order.
    const order = new Map(proposed.assign.map((a, i) => [a.id, i]));
    this.birds.sort((x, y) => (order.get(x.peerId) ?? 1e9) - (order.get(y.peerId) ?? 1e9));
    this.birds.forEach((b, i) => { b.slot = b.placed() ? i : -1; });
    for (const b of this.birds) if (!order.has(b.peerId)) this.unplace(b);

    this.makespanMs = proposed.makespanMs;
    this.evenMs = proposed.evenMs;
    this.proposedMs = proposed.makespanMs;
    if (moved.length) this.lastMoveAt = this.now();
    const note = this.note(true, d.reason, d.gain, moved);
    if (moved.length) this.onPlan?.(proposed, moved);
    return note;
  }

  unplace(b) {
    b.start = null; b.end = null; b.slot = -1; b.bytes = 0;
    b.why = 'holds no layers';
  }
  unplaceAll() { for (const b of this.birds) this.unplace(b); }

  note(moved, reason, gain = null, ids = []) {
    this.decision = {at: this.now(), moved, reason, gain: gain ?? null,
                     moved_peers: ids};
    return {move: moved, reason, gain: gain ?? null, moved: ids};
  }

  /**
   * Feed one token's timings into each device's rate. Called at a TOKEN BOUNDARY,
   * so a rate is a bytes/ms pair whose halves belong to the same token.
   *
   * Recording and acting are split on purpose. Acting -- moving layers because
   * the timings say so -- drops the conversation (the K/V cache is sharded by
   * layer), so the server only calls plan() for speed at the START of a
   * conversation, when there is nothing to lose. Recording happens every token
   * so that decision has evidence when it comes.
   */
  recordTimings() {
    for (const b of this.birds) {
      if (b.lastMs != null && b.bytes > 0) b.rate.observe(b.bytes, b.lastMs);
    }
  }

  /** Record this token's timings and ask whether the assignment should move. */
  observe() {
    this.recordTimings();
    return this.plan();
  }

  /** A membership change arrived while a token was in flight. */
  defer() { this.deferred = true; }

  /** Apply a deferred membership change. Returns the decision, or null if none. */
  applyPending() {
    if (!this.deferred) return null;
    this.deferred = false;
    return this.plan({force: true});
  }

  /** Who each bird forwards to. The last one replies to the coordinator. */
  chainFor(peerId) {
    const b = this.byPeer(peerId);
    if (!b || !b.placed()) {
      // Still a member, just not holding layers: tell it so explicitly. Silence
      // here is what made a bird sit forever showing "waiting for activations"
      // when it had in fact been reassigned out of the chain.
      return b ? {t: 'chain', slot: -1, start: null, end: null,
                  next_peer: null, next_range: null} : null;
    }
    const placed = this.chain();
    const at = placed.indexOf(b);
    const next = placed[at + 1] || null;
    return {t: 'chain', slot: b.slot, start: b.start, end: b.end,
            next_peer: next ? next.peerId : null,
            next_range: next ? `${next.start}-${next.end}` : null};
  }

  /** The placed birds in layer order: the actual pipeline. */
  chain() {
    return this.birds.filter(b => b.placed()).sort((a, b) => a.start - b.start);
  }

  /** Tell every bird its range and its successor — call whenever either changes. */
  announce() {
    for (const b of this.birds) {
      // No socket yet: nothing to send on. That bird is caught up on its hello
      // instead -- see the chainFor() send in the server's websocket handler. This
      // gap is why a device could be told one range at /join and moved to another
      // before it ever connected.
      if (!b.ws) continue;
      const c = this.chainFor(b.peerId);
      if (c) try { b.ws.send(JSON.stringify(c)); } catch {}
    }
  }

  /** Send one lap through every bird, in layer order.
   *
   *  Two shapes, decided by whether a bird can reach its successor directly:
   *
   *    chained   coord -> A -> B -> coord      (one round trip; B answers)
   *    relayed   coord -> A -> coord -> B ...  (fallback, one trip per bird)
   *
   *  A bird forwards peer-to-peer when it has an open channel to the next one,
   *  so we can only await the tail for the longest chained run from the head.
   */
  async lap(floats, seq, hidden, offset, flags = {}) {
    const chain = this.chain();
    if (!chain.length) throw new Error('no device holds any layer');
    let flat = floats, i = 0;
    while (i < chain.length) {
      // How far does the direct chain reach from bird i?
      let j = i;
      while (j + 1 < chain.length && chain[j].forwardsDirectly) j++;
      const got = await chain[i].sendChained(flat, seq, hidden, offset, chain[j], flags);
      // teardown() resolves a pending waiter with null so the lap does not hang for
      // the full timeout when a device's socket dies mid-frame. Say what happened:
      // destructuring the null instead reported "Cannot destructure property 'data'",
      // which tells the person holding the phone nothing at all.
      if (!got) {
        throw new Error(`the device holding layers ${chain[j].start}-${chain[j].end} ` +
                        `(${chain[j].label}) dropped its connection mid-token`);
      }
      // A NACK: the device refused the frame. The token is torn either way (every
      // device before it has already appended this position to its cache), so the
      // turn has to end -- but the device stays a member. The reason travels up so
      // the caller can tell "still streaming" from "cannot compute".
      if (got.nack) {
        const n = got.nack;
        const who = chain.find(b => b.peerId === n.peer_id) || chain[j];
        const e = new Error(n.reason === 'failed'
          ? `${who.label} (layers ${who.start}-${who.end}) could not compute a ` +
            `frame: ${n.detail || 'no detail'}`
          : `${who.label} was sent a frame before it held layers ` +
            `${who.start}-${who.end} (it reported: ${n.reason})`);
        e.nack = n; e.bird = who;
        throw e;
      }
      flat = got.data;
      i = j + 1;
    }
    return {data: flat};
  }

  byPeer(id) { return this.birds.find(b => b.peerId === id) || null; }

  /** Can a lap run right now: covered, AND every placed bird has confirmed the
   *  range it is assigned? The second half is the readiness handshake; without it
   *  a lap could start into a device that is still streaming its weights. */
  ready() {
    return this.covered() && this.pending().length === 0;
  }

  /** The placed birds that have not yet confirmed the range they were assigned:
   *  what /status shows as "still loading", and what a turn waits on. */
  pending() {
    return this.chain().filter(b => !b.isReady());
  }

  /**
   * Wait until every placed bird has confirmed, or until it is clear that will
   * not happen. Resolves {ok: true} or {ok: false, reason}. `onWait` is called
   * about once a second with the birds still pending, so a turn can say what it
   * is waiting for instead of appearing to hang.
   */
  async awaitReady({timeoutMs = 90000, onWait = null, pollMs = 200} = {}) {
    const t0 = Date.now();
    let lastSaid = 0;
    for (;;) {
      if (this.infeasible) return {ok: false, reason: this.infeasible.message};
      // Assigned to a member that is still connecting is a reason to wait;
      // assigned to nobody, or to a device that has gone quiet, is not.
      if (!this.assigned()) {
        const miss = this.missing();
        return {ok: false, reason: miss.length
          ? `waiting for devices to cover layer${miss.length > 1 || miss[0].includes('-')
              ? 's' : ''} ${miss.join(', ')}`
          : 'waiting for the devices that hold the layers to answer'};
      }
      const pend = this.chain().filter(b => !b.isReady() || !b.alive());
      if (!pend.length && this.covered()) return {ok: true};
      const waited = Date.now() - t0;
      if (waited > timeoutMs) {
        return {ok: false, reason: `gave up after ${(waited / 1000).toFixed(0)}s: ` +
          pend.map(b => `${b.label} has not confirmed layers ${b.start}-${b.end}`).join('; ')};
      }
      if (onWait && Date.now() - lastSaid > 1000) {
        lastSaid = Date.now();
        onWait(pend, waited);
      }
      await new Promise(r => setTimeout(r, pollMs));
    }
  }

  /** Is every bird-side layer assigned to a device that is answering? */
  covered() {
    if (this.infeasible) return false;
    const chain = this.chain();
    if (!chain.length || !chain.every(b => b.alive())) return false;
    return this.contiguous(chain);
  }

  /** Is every bird-side layer assigned to a MEMBER -- answering, or admitted and
   *  still connecting? The difference from covered() is the join window: a device
   *  that was just given layers over HTTP has no socket for a moment, and a turn
   *  sent in that moment should wait for it, not report its layers uncovered. */
  assigned() {
    if (this.infeasible) return false;
    const chain = this.chain();
    if (!chain.length || !chain.every(b => b.claimed())) return false;
    return this.contiguous(chain);
  }

  /** Every layer, contiguously. A gap here would silently skip layers and produce
   *  plausible wrong text, which is the failure this checks for -- it cannot
   *  happen through plan(), but the chain is also filtered by liveness, and a dead
   *  bird in the middle leaves exactly such a gap. */
  contiguous(chain) {
    let want = this.layers[0].layer;
    for (const b of chain) {
      if (b.start !== want) return false;
      want = b.end + 1;
    }
    return want === this.layers[this.layers.length - 1].layer + 1;
  }

  /** Which layers nobody is answering for, as ranges, for the UI's "waiting for". */
  missing() {
    if (!this.layers.length) return [];
    const live = new Set();
    for (const b of this.chain()) {
      if (!b.alive()) continue;
      for (let i = b.start; i <= b.end; i++) live.add(i);
    }
    const out = [];
    let run = null;
    for (const l of this.layers) {
      if (live.has(l.layer)) { if (run) { out.push(run); run = null; } continue; }
      if (run && run.end === l.layer - 1) run.end = l.layer;
      else { if (run) out.push(run); run = {start: l.layer, end: l.layer}; }
    }
    if (run) out.push(run);
    return out.map(r => r.start === r.end ? `${r.start}` : `${r.start}-${r.end}`);
  }

  /** Everything a human needs to see why the split is what it is. */
  allocation() {
    const members = this.members();
    return {
      devices: members.length,
      layers: this.layers.length,
      bird_layers: this.layers.length
        ? `${this.layers[0].layer}-${this.layers[this.layers.length - 1].layer}` : '',
      bird_bytes: this.layers.reduce((a, l) => a + l.bytes, 0),
      // The predicted cost of a token with this assignment, and what an even
      // by-layer-count split would have cost with the same rates. The pair is the
      // evidence that weighting by bytes and speed bought anything.
      makespan_ms: this.makespanMs,
      even_split_ms: this.evenMs,
      proposed_ms: this.proposedMs ?? null,
      min_samples: MIN_SAMPLES,
      decision: this.decision,
      deferred: this.deferred,
      // Assigned but not yet confirmed: the devices a turn would wait for.
      loading: this.pending().map(b => ({peer_id: b.peerId, label: b.label,
                                          range: `${b.start}-${b.end}`})),
      infeasible: this.infeasible,
      // Per-layer bytes, so the 13% size spread is visible rather than asserted.
      layer_mb: this.layers.map(l => +(l.bytes / 1e6).toFixed(1)),
    };
  }
}

export {currentMakespan};
