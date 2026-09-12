// flock mesh — the flock of birds, and the links between them.
//
// Every link is a WebRTC data channel: coordinator <=> bird, and bird <=> bird.
// The websocket exists ONLY to carry offers/answers/ICE. Once a channel opens,
// no activation touches it again.
import nodeDataChannel from 'node-datachannel';
import {pack, unpack} from '../../web/js/wire.mjs';

const ICE = ['stun:stun.l.google.com:19302'];

export class Bird {
  constructor(start, end, slot) {
    Object.assign(this, {start, end, slot});
    this.peerId = null; this.label = '?';
    this.lastSeen = 0; this.lastMs = null; this.claimedAt = 0;
    this.frames = 0;
    this.transport = 'none';
    this.resetPending = false;
    this.forwardsDirectly = false;   // set when the bird reports a live peer link
    this.ws = null; this.pc = null; this.chan = null;
    this._waiter = null;
  }

  alive(grace = 40000) {
    const linked = !!(this.chanOpen() || this.ws);
    return linked && (Date.now() - this.lastSeen) < grace;
  }

  /** Is this slot spoken for? Alive, or claimed over HTTP and still connecting.
   *
   *  /join happens over plain HTTP, so a just-claimed slot has no websocket yet
   *  and so is not yet `alive()`. Without this window the next /join hands out
   *  the SAME slot and overwrites peerId, which is why several birds joining at
   *  once all ended up as slot 0 and the flock never covered its layers. */
  claimed(hold = 15000) {
    return this.alive() || (!!this.peerId && Date.now() - this.claimedAt < hold);
  }

  chanOpen() {
    try { return !!(this.chan && this.chan.isOpen()); } catch { return false; }
  }

  info() {
    return {slot: this.slot, start: this.start, end: this.end,
            n_layers: this.end - this.start + 1,
            alive: this.alive(), last_ms: this.lastMs,
            label: this.label, transport: this.transport,
            frames: this.frames, peer_id: this.peerId,
            // How long since we last heard anything: the UI can say "12s ago"
            // rather than a bare alive/dead flag that hides a stalling device.
            last_seen_ms: this.lastSeen ? Date.now() - this.lastSeen : null,
            forwards_directly: this.forwardsDirectly};
  }

  /** Push into this bird but wait for `awaitOn` to answer (the chain's tail).
   *  When this bird IS the tail, `awaitOn` is itself — so this one method covers
   *  both the chained and the single-bird case. */
  sendChained(floats, seq, hidden, offset, awaitOn, timeout = 120000) {
    const frame = pack(floats, {seq, hidden, offset, reset: this.resetPending});
    this.resetPending = false;
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
    const {data, meta} = unpack(
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    if (this._waiter) this._waiter({data, meta});
  }

  reset() { this.resetPending = true; }

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

export class Flock {
  constructor(ranges) {
    this.birds = ranges.map(([s, e], i) => new Bird(s, e, i));
  }

  /** Who each bird forwards to. The last one replies to the coordinator. */
  chainFor(peerId) {
    const b = this.byPeer(peerId);
    if (!b) return null;
    const next = this.birds[b.slot + 1] || null;
    return {t: 'chain', slot: b.slot, start: b.start, end: b.end,
            next_peer: next ? next.peerId : null,
            next_range: next ? `${next.start}-${next.end}` : null};
  }

  /** Tell every bird its successor — call whenever membership changes. */
  announce() {
    for (const b of this.birds) {
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
  async lap(floats, seq, hidden, offset) {
    let flat = floats, i = 0;
    while (i < this.birds.length) {
      // How far does the direct chain reach from bird i?
      let j = i;
      while (j + 1 < this.birds.length && this.birds[j].forwardsDirectly) j++;
      const {data} = await this.birds[i].sendChained(
        flat, seq, hidden, offset, this.birds[j]);
      flat = data;
      i = j + 1;
    }
    return {data: flat};
  }
  claim(peerId, label) {
    for (const b of this.birds) if (b.peerId === peerId) {
      b.lastSeen = Date.now(); b.claimedAt = Date.now(); return b;
    }
    for (const b of this.birds) if (!b.claimed()) {
      b.peerId = peerId; b.label = label;
      b.lastSeen = Date.now(); b.claimedAt = Date.now();
      b.frames = 0;
      return b;
    }
    return null;
  }
  byPeer(id) { return this.birds.find(b => b.peerId === id) || null; }
  ready() { return this.birds.every(b => b.alive()); }
  missing() { return this.birds.filter(b => !b.alive()).map(b => `${b.start}-${b.end}`); }
  reset() { this.birds.forEach(b => b.reset()); }
}
