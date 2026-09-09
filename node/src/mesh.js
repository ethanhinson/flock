// flock mesh — the flock of birds, and the links between them.
//
// Every link is a WebRTC data channel: coordinator <=> bird, and bird <=> bird.
// The websocket exists ONLY to carry offers/answers/ICE. Once a channel opens,
// no activation touches it again.
import nodeDataChannel from 'node-datachannel';
import {pack, unpack} from './wire.js';

const ICE = ['stun:stun.l.google.com:19302'];

export class Bird {
  constructor(start, end, slot) {
    Object.assign(this, {start, end, slot});
    this.peerId = null; this.label = '?';
    this.lastSeen = 0; this.lastMs = null;
    this.transport = 'none';
    this.resetPending = false;
    this.ws = null; this.pc = null; this.chan = null;
    this._waiter = null;
  }

  alive(grace = 40000) {
    return (this.chan || this.ws) && (Date.now() - this.lastSeen) < grace;
  }

  info() {
    return {slot: this.slot, start: this.start, end: this.end,
            alive: this.alive(), last_ms: this.lastMs,
            label: this.label, transport: this.transport};
  }

  /** Push activations to this bird and await the reply. */
  send(floats, seq, hidden, offset, timeout = 120000) {
    const frame = pack(floats, {seq, hidden, offset, reset: this.resetPending});
    this.resetPending = false;
    const viaRTC = !!(this.chan && this.chan.isOpen());
    const link = viaRTC ? this.chan : this.ws;
    if (!link) return Promise.reject(new Error(
      `no device holding layers ${this.start}-${this.end}`));
    this.transport = viaRTC ? 'webrtc' : 'ws';   // observed, not self-reported
    return new Promise((res, rej) => {
      const timer = setTimeout(() => {
        this._waiter = null;
        rej(new Error(`device holding layers ${this.start}-${this.end} stopped ` +
                      `responding — is its screen on and the tab in front?`));
      }, timeout);
      this._waiter = v => { clearTimeout(timer); this._waiter = null; res(v); };
      if (this.chan && this.chan.isOpen()) this.chan.sendMessageBinary(Buffer.from(frame));
      else this.ws.send(Buffer.from(frame));
    });
  }

  deliver(buf) {
    this.lastSeen = Date.now();
    const {data, meta} = unpack(
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    if (this._waiter) this._waiter({data, meta});
  }

  reset() { this.resetPending = true; }

  /** Offer a direct data channel to this bird, signalled over its websocket. */
  openRTC(onOpen) {
    // A reloaded bird signals again; drop the previous connection or stale
    // PeerConnections accumulate and the wrong one wins.
    try { this.chan?.close(); this.pc?.close(); } catch {}
    this.chan = null;
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
  claim(peerId, label) {
    for (const b of this.birds) if (b.peerId === peerId) { b.lastSeen = Date.now(); return b; }
    for (const b of this.birds) if (!b.alive()) {
      b.peerId = peerId; b.label = label; b.lastSeen = Date.now(); return b;
    }
    return null;
  }
  byPeer(id) { return this.birds.find(b => b.peerId === id) || null; }
  ready() { return this.birds.every(b => b.alive()); }
  missing() { return this.birds.filter(b => !b.alive()).map(b => `${b.start}-${b.end}`); }
  reset() { this.birds.forEach(b => b.reset()); }
}
