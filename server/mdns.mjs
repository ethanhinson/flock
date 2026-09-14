// flock mDNS advertiser — announce the coordinator on the local network.
//
// flock is LAN-only: every bird is on the same wifi as the coordinator, and the
// only thing a new device needs to join is the coordinator's address. Today you
// get that by reading the address the server prints at startup and typing it
// into a phone. mDNS (Bonjour / DNS-SD) lets the coordinator *announce* itself,
// so a native tool like sim_bird.mjs can find it without being told, and macOS
// and iOS can see it in Bonjour browsing.
//
// A browser tab cannot do mDNS — there is no Web API for it — so this does not
// replace the printed address for bird.html. What it does is:
//
//   1. Advertise _flock._tcp.local on 224.0.0.251:5353, so anything on the LAN
//      can discover the coordinator by service type.
//   2. Let sim_bird.mjs query for it (see tools/mdns-discover.mjs) instead of
//      requiring a FLOCK_URL.
//
// This is a minimal but correct mDNS responder: it sends a gratuitous response
// on startup and answers any PTR query for _flock._tcp.local. It does not
// implement the full spec (no cache flush, no NSEC) — the point is "can a
// device find the coordinator", not RFC 6762 conformance. The packet format is
// correct: real DNS records that real resolvers (dns-sd, avahi) parse.
//
// RUNS UNDER DENO (same as server.js). Deno provides Deno.listenDatagram for
// UDP multicast. Disable with FLOCK_NO_MDNS=1.

const MDNS_ADDR = '224.0.0.251';
const MDNS_PORT = 5353;
const SERVICE_TYPE = '_flock._tcp.local';
const SERVICE_NAME = 'flock';

let socket = null;
let timer = null;
let advertising = false;

function lanIp() {
  try {
    const ni = Deno.networkInterfaces?.() ?? [];
    for (const i of ni) {
      if (i.family === 'IPv4' && !i.address.startsWith('127.') &&
          !i.address.startsWith('169.254.') && !i.internal) return i.address;
    }
  } catch {}
  return '127.0.0.1';
}

function encodeName(name) {
  const parts = name.split('.');
  const out = [];
  for (const p of parts) {
    if (p.length > 63) throw new Error('DNS label too long: ' + p);
    out.push(p.length, ...new TextEncoder().encode(p));
  }
  out.push(0);
  return new Uint8Array(out);
}

function encodeTxt(fields) {
  const out = [];
  for (const [k, v] of Object.entries(fields)) {
    const s = k + '=' + v;
    if (s.length > 255) throw new Error('TXT entry too long: ' + s);
    out.push(s.length, ...new TextEncoder().encode(s));
  }
  if (out.length === 0) out.push(0);
  return new Uint8Array(out);
}

function u16(buf, o, v) { buf[o] = (v >> 8) & 0xff; buf[o + 1] = v & 0xff; }
function u32(buf, o, v) {
  buf[o] = (v >> 24) & 0xff; buf[o+1] = (v >> 16) & 0xff;
  buf[o+2] = (v >> 8) & 0xff; buf[o+3] = v & 0xff;
}

export function buildResponse(port, ip, modelName) {
  const typePtr = encodeName(SERVICE_TYPE);
  const instPtr = encodeName(SERVICE_NAME + '.' + SERVICE_TYPE);
  const hostPtr = encodeName(SERVICE_NAME + '.local');
  const srvRdata = new Uint8Array([
    0, 0, 0, 0, (port >> 8) & 0xff, port & 0xff, ...hostPtr,
  ]);
  const txtRdata = encodeTxt({model: modelName, port: String(port)});
  const aRdata = ip.split('.').map(Number);

  const recSize = (nameLen, rdLen) => nameLen + 10 + rdLen;
  const ptrSz = recSize(typePtr.length, instPtr.length);
  const srvSz = recSize(instPtr.length, srvRdata.length);
  const txtSz = recSize(instPtr.length, txtRdata.length);
  const aSz = recSize(hostPtr.length, 4);
  const total = 12 + ptrSz + srvSz + txtSz + aSz;
  const buf = new Uint8Array(total);
  let o = 0;
  // Header: ID=0, flags=0x8400 (response+authoritative),
  // QDCOUNT=0, ANCOUNT=1 (PTR), NSCOUNT=0, ARCOUNT=3 (SRV+TXT+A).
  u16(buf, o, 0); o += 2;
  u16(buf, o, 0x8400); o += 2;
  u16(buf, o, 0); o += 2;        // QDCOUNT
  u16(buf, o, 1); o += 2;        // ANCOUNT
  u16(buf, o, 0); o += 2;        // NSCOUNT
  u16(buf, o, 3); o += 2;        // ARCOUNT

  // PTR: _flock._tcp.local -> flock._flock._tcp.local
  buf.set(typePtr, o); o += typePtr.length;
  u16(buf, o, 0x000c); o += 2; u16(buf, o, 0x8001); o += 2;
  u32(buf, o, 120); o += 4; u16(buf, o, instPtr.length); o += 2;
  buf.set(instPtr, o); o += instPtr.length;

  // SRV: flock._flock._tcp.local -> flock.local:port
  buf.set(instPtr, o); o += instPtr.length;
  u16(buf, o, 0x0021); o += 2; u16(buf, o, 0x8001); o += 2;
  u32(buf, o, 120); o += 4; u16(buf, o, srvRdata.length); o += 2;
  buf.set(srvRdata, o); o += srvRdata.length;

  // TXT: flock._flock._tcp.local -> model=... port=...
  buf.set(instPtr, o); o += instPtr.length;
  u16(buf, o, 0x0010); o += 2; u16(buf, o, 0x8001); o += 2;
  u32(buf, o, 120); o += 4; u16(buf, o, txtRdata.length); o += 2;
  buf.set(txtRdata, o); o += txtRdata.length;

  // A: flock.local -> <lan-ip> (A record on the SRV target hostname, per RFC 6763)
  buf.set(hostPtr, o); o += hostPtr.length;
  u16(buf, o, 0x0001); o += 2; u16(buf, o, 0x8001); o += 2;
  u32(buf, o, 120); o += 4; u16(buf, o, 4); o += 2;
  buf.set(new Uint8Array(aRdata), o); o += 4;

  return buf.slice(0, o);
}

export function buildQuery() {
  const name = encodeName(SERVICE_TYPE);
  const buf = new Uint8Array(12 + name.length + 4);
  let o = 0;
  u16(buf, o, 0); o += 2; u16(buf, o, 0); o += 2;
  u16(buf, o, 1); o += 2; u16(buf, o, 0); o += 2;
  u16(buf, o, 0); o += 2; u16(buf, o, 0); o += 2;
  buf.set(name, o); o += name.length;
  u16(buf, o, 0x000c); o += 2; u16(buf, o, 0x0001); o += 2;
  return buf.slice(0, o);
}

export async function startMdns(port, meta) {
  meta = meta || {};
  if (process.env.FLOCK_NO_MDNS) return;
  if (advertising) return;
  const ip = lanIp();
  const pkt = buildResponse(port, ip, meta.model || 'flock');

  try {
    socket = Deno.listenDatagram({port: MDNS_PORT, hostname: '0.0.0.0', transport: 'udp'});
    try { socket.joinMulticastGroup(MDNS_ADDR); } catch {}
  } catch (e) {
    console.error('mDNS: could not bind :' + MDNS_PORT + ' — ' + e.message);
    return;
  }
  advertising = true;

  const send = () => { try { socket.send(pkt, {hostname: MDNS_ADDR, port: MDNS_PORT}); } catch {} };
  send();
  timer = setInterval(send, 30000);

  // The receive loop is fire-and-forget, but stopMdns() closing the socket
  // throws inside the for-await — catch it so shutdown does not crash Deno
  // with an unhandled rejection.
  (async () => {
    try {
      for await (const [data] of socket) {
        if (data.length > 12 && (data[2] & 0x80) === 0) {
          const text = new TextDecoder().decode(data);
          if (text.includes('_flock')) send();
        }
      }
    } catch {}
  })();
}

export function stopMdns() {
  if (timer) { clearInterval(timer); timer = null; }
  if (socket) { try { socket.close(); } catch {} socket = null; }
  advertising = false;
}
