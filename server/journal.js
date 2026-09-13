// flock/journal -- the conversation as the exact sequence of token ids fed, and
// the activation that left the coordinator's layers for each one.
//
// WHY A JOURNAL AND NOT THE HISTORY. The K/V cache is sharded by layer across
// the devices, and a device's shard lives in GPU buffers inside a browser tab: a
// refresh, a reassignment, a backgrounded phone or a coordinator restart
// destroys it. It cannot be preserved, so it has to be REBUILT, and the only way
// to rebuild it exactly is to feed the device the same activations in the same
// order. Re-tokenizing the chat history does not give that: the chat template
// scaffolds the current turn differently from past ones, so a re-encoded
// conversation diverges from what was actually fed (measured: at token 9 of
// 13). So the journal keeps the ids that WERE fed, never re-encodes, and replay
// feeds them back verbatim.
//
// WHY THE ACTIVATIONS TOO. The birds' input is the hidden state leaving the
// coordinator's layers, already rounded to f16 for the wire. Journaling those
// f16 bits (2 KB per token at hidden=1024) means a replay hands each bird
// exactly the bytes it saw the first time, without recomputing the coordinator's
// prefix -- and f16 -> f32 -> f16 is an exact round trip, so the replayed frame
// is bit-identical to the original one. Prefill is bit-identical to N decode
// steps (kernels/test_prefill.ts), so a bird prefilling the journal in chunks
// ends up with the same cache it would have built one token at a time.
//
// ON DISK, APPEND-ONLY. Each feed is one record, written as it happens, so a
// coordinator that dies mid-conversation comes back with the conversation up to
// the last lap that completed. A torn last record (killed mid-write) is cut off
// on load. Truncation -- rewinding an aborted turn -- is itself a record, so the
// file is only ever appended to or cleared, never rewritten in place.
import {appendFileSync, closeSync, existsSync, mkdirSync, openSync, readFileSync,
        truncateSync, writeSync} from 'node:fs';
import path from 'node:path';
// Buffer is a global under Node but not under Deno, which is what runs the server.
import {Buffer} from 'node:buffer';
import {f32to16, f16to32} from '../web/js/wire.mjs';

const MAGIC = 0x4a4b4c46;   // "FLKJ" little-endian
const VERSION = 1;
const HEADER = 16;
const KIND_FEED = 1, KIND_TURN = 2, KIND_TRUNCATE = 3;

export class Journal {
  /**
   * @param {object} opts  `hidden` is the activation width per token; `file` is
   *                       where to persist (null keeps it in memory only).
   */
  constructor({hidden, file = null} = {}) {
    if (!(hidden > 0)) throw new Error('journal needs the hidden size');
    this.hidden = hidden;
    this.file = file;
    this.records = [];      // [{offset, ids, acts: Uint16Array}] in order
    this.turns = 0;         // completed turns
    this.turnStart = 0;     // position where the turn in progress began
    this.length = 0;        // tokens fed
  }

  /** The exact ids fed, in order. */
  ids() {
    const out = [];
    for (const r of this.records) for (const id of r.ids) out.push(id);
    return out;
  }

  /**
   * Record one lap: `ids` were fed at `offset`, and `floats` (n * hidden, f32)
   * is what left the coordinator's layers for them. Stored as the f16 bits the
   * wire carried.
   */
  append(ids, offset, floats) {
    if (offset !== this.length) {
      throw new Error(`journal is at ${this.length}, cannot append at ${offset}`);
    }
    const n = ids.length;
    if (floats.length < n * this.hidden) {
      throw new Error(`${floats.length} floats for ${n} tokens; want ${n * this.hidden}`);
    }
    const acts = new Uint16Array(n * this.hidden);
    for (let i = 0; i < acts.length; i++) acts[i] = f32to16(floats[i]);
    const rec = {offset, ids: ids.slice(), acts};
    this.records.push(rec);
    this.length += n;
    this.write(this.encodeFeed(rec));
  }

  /** The turn in progress completed; the next one starts here. */
  endTurn() {
    this.turns++;
    this.turnStart = this.length;
    this.write(this.encodeTurn(this.length, this.turns));
  }

  /**
   * Drop everything fed after `pos`, which must be a record boundary -- in
   * practice the start of the turn in progress, when that turn failed or was
   * aborted: the caches hold a partial prompt or a torn reply that nothing can
   * be appended to, but the conversation before it is intact and worth keeping.
   */
  truncate(pos) {
    if (pos === this.length) return;
    if (pos > this.length) throw new Error(`cannot truncate ${this.length} up to ${pos}`);
    while (this.records.length && this.records[this.records.length - 1].offset >= pos) {
      this.records.pop();
    }
    this.length = this.records.length
      ? this.records[this.records.length - 1].offset + this.records[this.records.length - 1].ids.length
      : 0;
    if (this.length !== pos) {
      throw new Error(`${pos} is not a record boundary (nearest is ${this.length})`);
    }
    if (this.turnStart > this.length) this.turnStart = this.length;
    this.write(this.encodeTruncate(pos));
  }

  /** Rewind to the start of the turn in progress. Returns the new length. */
  rewindTurn() {
    this.truncate(this.turnStart);
    return this.length;
  }

  /** A new conversation: forget everything, on disk too. */
  clear() {
    this.records = [];
    this.turns = 0;
    this.turnStart = 0;
    this.length = 0;
    if (this.file) {
      mkdirSync(path.dirname(this.file), {recursive: true});
      const fd = openSync(this.file, 'w');
      try { writeSync(fd, this.header()); } finally { closeSync(fd); }
    }
  }

  /** The activations for positions [from, from + n), as f32, for replay. */
  activations(from, n) {
    const H = this.hidden;
    const out = new Float32Array(n * H);
    let got = 0;
    for (const r of this.records) {
      const end = r.offset + r.ids.length;
      if (end <= from || r.offset >= from + n) continue;
      const lo = Math.max(from, r.offset), hi = Math.min(from + n, end);
      for (let p = lo; p < hi; p++) {
        const src = (p - r.offset) * H, dst = (p - from) * H;
        for (let i = 0; i < H; i++) out[dst + i] = f16to32(r.acts[src + i]);
        got++;
      }
    }
    if (got !== n) throw new Error(`journal has ${got} of positions ${from}..${from + n - 1}`);
    return out;
  }

  /** The f16 bits for one position, for checking a recompute against. */
  activationBits(pos) {
    for (const r of this.records) {
      if (pos >= r.offset && pos < r.offset + r.ids.length) {
        const s = (pos - r.offset) * this.hidden;
        return r.acts.subarray(s, s + this.hidden);
      }
    }
    return null;
  }

  // --- persistence ---------------------------------------------------------

  header() {
    const b = Buffer.alloc(HEADER);
    b.writeUInt32LE(MAGIC, 0);
    b.writeUInt32LE(VERSION, 4);
    b.writeUInt32LE(this.hidden, 8);
    b.writeUInt32LE(0, 12);
    return b;
  }
  encodeFeed(rec) {
    const n = rec.ids.length;
    const b = Buffer.alloc(12 + n * 4 + rec.acts.byteLength);
    b.writeUInt32LE(KIND_FEED, 0);
    b.writeUInt32LE(rec.offset, 4);
    b.writeUInt32LE(n, 8);
    for (let i = 0; i < n; i++) b.writeUInt32LE(rec.ids[i], 12 + i * 4);
    Buffer.from(rec.acts.buffer, rec.acts.byteOffset, rec.acts.byteLength).copy(b, 12 + n * 4);
    return b;
  }
  encodeTurn(offset, turns) {
    const b = Buffer.alloc(12);
    b.writeUInt32LE(KIND_TURN, 0); b.writeUInt32LE(offset, 4); b.writeUInt32LE(turns, 8);
    return b;
  }
  encodeTruncate(offset) {
    const b = Buffer.alloc(12);
    b.writeUInt32LE(KIND_TRUNCATE, 0); b.writeUInt32LE(offset, 4); b.writeUInt32LE(0, 8);
    return b;
  }
  write(buf) {
    if (!this.file) return;
    if (!existsSync(this.file)) {
      mkdirSync(path.dirname(this.file), {recursive: true});
      appendFileSync(this.file, this.header());
    }
    appendFileSync(this.file, buf);
  }

  /**
   * Reload from disk. Returns {ok, tokens, turns, torn} -- `torn` is the number
   * of trailing bytes that did not make a whole record and were cut off.
   * A file for a different hidden size or version is ignored (and cleared), not
   * misread.
   */
  load() {
    this.records = []; this.turns = 0; this.turnStart = 0; this.length = 0;
    if (!this.file || !existsSync(this.file)) return {ok: false, tokens: 0, turns: 0, torn: 0};
    const buf = readFileSync(this.file);
    if (buf.length < HEADER || buf.readUInt32LE(0) !== MAGIC ||
        buf.readUInt32LE(4) !== VERSION || buf.readUInt32LE(8) !== this.hidden) {
      this.clear();
      return {ok: false, tokens: 0, turns: 0, torn: 0, reason: 'not a journal for this model'};
    }
    let at = HEADER, good = HEADER;
    const H = this.hidden;
    while (at + 12 <= buf.length) {
      const kind = buf.readUInt32LE(at), offset = buf.readUInt32LE(at + 4), n = buf.readUInt32LE(at + 8);
      if (kind === KIND_FEED) {
        const size = 12 + n * 4 + n * H * 2;
        if (at + size > buf.length) break;                 // torn tail
        if (offset !== this.length) break;                 // inconsistent: stop trusting it
        const ids = [];
        for (let i = 0; i < n; i++) ids.push(buf.readUInt32LE(at + 12 + i * 4));
        const acts = new Uint16Array(n * H);
        // Copy rather than view: the file buffer's offset is not 2-byte aligned in
        // general, and a Uint16Array over an unaligned offset throws.
        Buffer.from(acts.buffer).set(buf.subarray(at + 12 + n * 4, at + size));
        this.records.push({offset, ids, acts});
        this.length += n;
        at += size;
      } else if (kind === KIND_TURN) {
        if (offset !== this.length) break;
        this.turns = n; this.turnStart = offset;
        at += 12;
      } else if (kind === KIND_TRUNCATE) {
        if (offset > this.length) break;
        while (this.records.length && this.records[this.records.length - 1].offset >= offset) {
          this.records.pop();
        }
        this.length = this.records.length
          ? this.records[this.records.length - 1].offset + this.records[this.records.length - 1].ids.length
          : 0;
        if (this.length !== offset) break;
        if (this.turnStart > this.length) this.turnStart = this.length;
        at += 12;
      } else {
        break;
      }
      good = at;
    }
    const torn = buf.length - good;
    if (torn > 0) truncateSync(this.file, good);
    return {ok: true, tokens: this.length, turns: this.turns, torn};
  }
}
