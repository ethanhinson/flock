// flock wire format — the single implementation, shared by both sides.
//
// The coordinator (server/*) imports this file directly; browsers fetch it
// from /js/wire.mjs. One copy means the two ends cannot drift apart, which is
// the only way a binary format like this stays safe to change.
//
// Layout: "FLK1" | seq u32 | hidden u32 | offset u32 | flags u32 | f16 payload
//
// JS has no Float16Array, so we convert by hand. This is the same trick
// swarmllm uses (f16-packed Uint16Array): activations are small (|x| < 0.1)
// and derived from bf16 weights, so f16 costs no real precision while halving
// the bytes — and it's ~8x smaller than sending JSON floats.

const MAGIC = 0x314b4c46; // "FLK1" little-endian

// float32 -> float16 bits. Handles subnormals and overflow to inf.
//
// Rounds to nearest-even rather than truncating. Truncation looks harmless but
// costs ~3 bits of an 11-bit mantissa: measured 5.8e-2 worst-case relative
// error versus 4.9e-4 when rounding. That is the difference between f16 being
// lossless for activations and being visibly lossy.
export function f32to16(v) {
  const buf = new DataView(new ArrayBuffer(4));
  buf.setFloat32(0, v);
  const x = buf.getUint32(0);
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff, man = x & 0x7fffff;
  if (exp === 255) return sign | 0x7c00 | (man ? 0x200 : 0);      // inf / nan
  exp = exp - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;                             // overflow
  if (exp <= 0) {                                                  // subnormal
    if (exp < -10) return sign;
    man |= 0x800000;
    const shift = 14 - exp;
    const sub = man >> shift;
    // round-to-nearest-even on the bits we are dropping
    const rem = man & ((1 << shift) - 1), half = 1 << (shift - 1);
    return sign | (sub + ((rem > half || (rem === half && (sub & 1))) ? 1 : 0));
  }
  let h = (exp << 10) | (man >> 13);
  const rem = man & 0x1fff;                                        // dropped bits
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h += 1;         // may carry into exp
  return sign | h;
}

export function f16to32(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >>> 10) & 0x1f, man = h & 0x3ff;
  if (exp === 0) return sign * man * Math.pow(2, -24);
  if (exp === 31) return man ? NaN : sign * Infinity;
  return sign * (1 + man / 1024) * Math.pow(2, exp - 15);
}

export function pack(floats, {seq, hidden, offset = 0, reset = false}) {
  const out = new ArrayBuffer(20 + floats.length * 2);
  const dv = new DataView(out);
  dv.setUint32(0, MAGIC, true);
  dv.setUint32(4, seq, true);
  dv.setUint32(8, hidden, true);
  dv.setUint32(12, offset, true);
  dv.setUint32(16, reset ? 1 : 0, true);
  for (let i = 0; i < floats.length; i++) {
    dv.setUint16(20 + i * 2, f32to16(floats[i]), true);
  }
  return out;
}

export function unpack(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('not a flock frame');
  const meta = {
    seq: dv.getUint32(4, true), hidden: dv.getUint32(8, true),
    offset: dv.getUint32(12, true), reset: !!dv.getUint32(16, true),
  };
  const n = (buf.byteLength - 20) / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = f16to32(dv.getUint16(20 + i * 2, true));
  return {data: out, meta};
}
