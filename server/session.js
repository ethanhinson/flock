// flock/session -- the token that lets a device come back as itself.
//
// IDENTITY IS STILL THE COORDINATOR'S. /join mints the peer id, as it always
// has; what a session adds is a random, server-issued token bound to that peer
// id, so a page that refreshes (or a phone that comes back from the background)
// can prove it is the same device and RE-ATTACH: same peer id, same layers,
// nobody else moves. The token is not a client-chosen identity -- that was the
// design that let one device take over another's slot -- it is a capability the
// coordinator handed out and can check.
//
// What a token is bound to, and why each binding is there:
//   the peer id     it names exactly one membership
//   the source      a token that arrives from a different address is not the
//                   same device; a LAN peer that reads one off a screen gets a
//                   fresh join, never someone else's layers
//   the caps        the probe's numbers are static per device; a token
//                   presented with different limits is a different device
//   an expiry       ~10 minutes, refreshed on activity, so a token from a tab
//                   that died is useless after it -- and a paused device that
//                   never comes back is dropped when its token lapses
//
// Persisted to disk with the journal, so re-attaching also works across a
// coordinator restart: the sessions file carries each member's placement too,
// which is what lets the restarted coordinator put every device back on the
// layers it still holds instead of reallocating everyone.
//
// A token that fails any check is simply a fresh join. Never an error the client
// has to handle, never a hint about WHICH check failed.
import {existsSync, mkdirSync, readFileSync, renameSync, writeFileSync} from 'node:fs';
import {randomBytes} from 'node:crypto';
import path from 'node:path';

export const DEFAULT_TTL_MS = 10 * 60 * 1000;

/** What the binding compares: the probe's numbers, with unlimited as null so the
 *  same key comes out of a live object and out of the JSON it was saved as. */
export function capsKey(caps) {
  return JSON.stringify(capsObj(caps));
}
/** The same numbers as a plain object, which is how a record stores them. */
export function capsObj(caps) {
  const fin = v => (typeof v === 'number' && Number.isFinite(v)) ? v : null;
  return {bind: fin(caps?.bind), budget: fin(caps?.budget), cores: fin(caps?.cores),
          gpu: caps?.gpu ?? null};
}

export class Sessions {
  constructor({file = null, ttlMs = DEFAULT_TTL_MS, now = Date.now} = {}) {
    this.file = file;
    this.ttlMs = ttlMs;
    this.now = now;
    this.byToken = new Map();
    this.byPeer = new Map();
    this._saveTimer = null;
  }

  /**
   * Mint a token for a membership. Any earlier token for the same peer id is
   * revoked: one device, one token.
   */
  issue({peerId, from, caps, label = '?', onCoordinator = false, start = null, end = null}) {
    this.revoke(peerId, {save: false});
    const rec = {token: randomBytes(24).toString('hex'), peer_id: peerId, label,
                 from: from || null, caps: capsObj(caps), on_coordinator: !!onCoordinator,
                 start, end, paused: false, paused_at: null,
                 expires_at: this.now() + this.ttlMs};
    this.byToken.set(rec.token, rec);
    this.byPeer.set(peerId, rec);
    this.save();
    return rec.token;
  }

  /**
   * Check a presented token. {ok: true, peerId, record} when it is the same
   * device coming back; {ok: false, reason} otherwise. The reason is for the
   * coordinator's log, not for the client.
   */
  validate(token, {from = null, caps = null} = {}) {
    const rec = typeof token === 'string' ? this.byToken.get(token) : null;
    if (!rec) return {ok: false, reason: 'unknown token'};
    if (rec.expires_at <= this.now()) {
      this.revoke(rec.peer_id);
      return {ok: false, reason: 'expired'};
    }
    if (rec.from && from && rec.from !== from) {
      return {ok: false, reason: `from ${from}, issued to ${rec.from}`};
    }
    if (caps && capsKey(caps) !== capsKey(rec.caps)) {
      return {ok: false, reason: 'different device limits'};
    }
    return {ok: true, peerId: rec.peer_id, record: rec};
  }

  /** The device is active: push its expiry out. Saved lazily -- this is called
   *  on every heartbeat and every frame. */
  touch(peerId) {
    const rec = this.byPeer.get(peerId);
    if (!rec) return false;
    rec.expires_at = this.now() + this.ttlMs;
    this.save({lazy: true});
    return true;
  }

  /** Record placement or pause state, so a restart can restore it. */
  update(peerId, fields) {
    const rec = this.byPeer.get(peerId);
    if (!rec) return false;
    Object.assign(rec, fields);
    this.save();
    return true;
  }

  /** Every member's placement at once, saved once. */
  setPlacements(list) {
    let changed = false;
    for (const p of list) {
      const rec = this.byPeer.get(p.peer_id);
      if (!rec) continue;
      Object.assign(rec, {start: p.start ?? null, end: p.end ?? null,
                          paused: !!p.paused, paused_at: p.paused_at ?? null});
      changed = true;
    }
    if (changed) this.save();
  }

  revoke(peerId, {save = true} = {}) {
    const rec = this.byPeer.get(peerId);
    if (!rec) return false;
    this.byPeer.delete(peerId);
    this.byToken.delete(rec.token);
    if (save) this.save();
    return true;
  }

  get(peerId) { return this.byPeer.get(peerId) || null; }

  /** Every unexpired session. */
  live() {
    const t = this.now();
    return [...this.byPeer.values()].filter(r => r.expires_at > t);
  }

  /** Has this device's token lapsed? A paused device is kept exactly this long. */
  expired(peerId) {
    const rec = this.byPeer.get(peerId);
    return !rec || rec.expires_at <= this.now();
  }

  // --- persistence ---------------------------------------------------------

  save({lazy = false} = {}) {
    if (!this.file) return;
    if (lazy) {
      if (this._saveTimer) return;
      this._saveTimer = setTimeout(() => { this._saveTimer = null; this.save(); }, 5000);
      if (this._saveTimer.unref) this._saveTimer.unref();
      return;
    }
    mkdirSync(path.dirname(this.file), {recursive: true});
    // Write-then-rename, so a crash mid-write leaves the previous file intact
    // rather than a half-written one.
    const tmp = `${this.file}.tmp`;
    writeFileSync(tmp, JSON.stringify({version: 1, sessions: this.live()}, null, 1));
    renameSync(tmp, this.file);
  }

  /** Reload from disk, dropping anything expired. Returns the records restored. */
  load() {
    this.byToken.clear(); this.byPeer.clear();
    if (!this.file || !existsSync(this.file)) return [];
    let data;
    try { data = JSON.parse(readFileSync(this.file, 'utf8')); } catch { return []; }
    const t = this.now();
    for (const rec of data?.sessions || []) {
      if (!rec?.token || !rec?.peer_id || !(rec.expires_at > t)) continue;
      this.byToken.set(rec.token, rec);
      this.byPeer.set(rec.peer_id, rec);
    }
    return [...this.byPeer.values()];
  }
}
