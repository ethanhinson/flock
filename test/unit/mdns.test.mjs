// The mDNS packet format and discovery parser — as arithmetic.
//
// buildResponse() in server/mdns.mjs encodes a DNS-SD response (PTR + SRV +
// TXT + A) as a raw binary packet. parseResponse() in tools/mdns-discover.mjs
// decodes it back. This test round-trips them and checks the header layout,
// record types, and extracted fields — the same way a real mDNS resolver
// (dns-sd, avahi) would parse the packet.
//
// Needs no GPU, no network, no coordinator — pure binary encode/decode.
//
//   node test/unit/mdns.test.mjs

import {buildResponse, buildQuery} from '../../server/mdns.mjs';
import {parseResponse} from '../../tools/mdns-discover.mjs';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};

const u16 = (buf, o) => (buf[o] << 8) | buf[o + 1];

// --- header layout ---------------------------------------------------------
{
  const pkt = buildResponse(8000, '192.168.1.10', 'test-model');

  ok('packet is longer than the 12-byte header', pkt.length > 12,
     `${pkt.length} bytes`);

  // The DNS header is 6 u16 fields: ID, FLAGS, QDCOUNT, ANCOUNT, NSCOUNT, ARCOUNT.
  // A gratuitous mDNS response has QDCOUNT=0, ANCOUNT=1 (the PTR), NSCOUNT=0,
  // ARCOUNT=3 (SRV + TXT + A). Getting these wrong makes a strict parser miss
  // records, and was the first bug the review caught.
  ok('ID = 0', u16(pkt, 0) === 0, `got ${u16(pkt, 0)}`);
  ok('flags = 0x8400 (response + authoritative)', u16(pkt, 2) === 0x8400,
     `got 0x${u16(pkt, 2).toString(16)}`);
  ok('QDCOUNT = 0 (no question section)', u16(pkt, 4) === 0,
     `got ${u16(pkt, 4)}`);
  ok('ANCOUNT = 1 (one PTR answer)', u16(pkt, 6) === 1,
     `got ${u16(pkt, 6)}`);
  ok('NSCOUNT = 0', u16(pkt, 8) === 0, `got ${u16(pkt, 8)}`);
  ok('ARCOUNT = 3 (SRV + TXT + A)', u16(pkt, 10) === 3, `got ${u16(pkt, 10)}`);
}

// --- round-trip: build -> parse -------------------------------------------
{
  const pkt = buildResponse(8000, '192.168.1.10', 'test-model');
  const r = parseResponse(pkt);

  ok('parse returns a result', r !== null);
  ok('port extracted from SRV record', r && r.port === 8000,
     r ? `got ${r.port}` : 'no result');
  ok('ip extracted from A record', r && r.ip === '192.168.1.10',
     r ? `got ${r.ip}` : 'no result');
  ok('service instance from SRV record',
     r && r.instance === 'flock._flock._tcp.local',
     r ? `got ${r.instance}` : 'no result');
}

// --- different port and IP ------------------------------------------------
{
  const pkt = buildResponse(9999, '10.0.0.5', 'another-model');
  const r = parseResponse(pkt);

  ok('different port', r && r.port === 9999, r ? `got ${r.port}` : 'no result');
  ok('different ip', r && r.ip === '10.0.0.5', r ? `got ${r.ip}` : 'no result');
}

// --- A record uses the SRV target hostname (flock.local), not the instance -
{
  // The A record should be named flock.local (the SRV target), not
  // flock._flock._tcp.local (the service instance). RFC 6763 says the A/AAAA
  // record belongs on the SRV target. If it uses the instance name, the A
  // record is longer by 12 bytes and a strict resolver looking up the SRV
  // target won't find it.
  const pkt = buildResponse(8000, '192.168.1.10', 'test');

  // Encode flock.local and flock._flock._tcp.local to compare lengths.
  function encodeName(name) {
    const parts = name.split('.');
    const out = [];
    for (const p of parts) out.push(p.length, ...new TextEncoder().encode(p));
    out.push(0);
    return new Uint8Array(out);
  }
  const flockLocal = encodeName('flock.local');             // 13 bytes
  const flockInstance = encodeName('flock._flock._tcp.local'); // 25 bytes

  // The A record is the last record. Its name is at the end of the packet,
  // minus 14 bytes (4 for IP + 10 for type/class/ttl/rdlen). Find it by
  // searching backwards from the end for the A record's rdata (4 IP bytes).
  // More robust: parse the whole packet and check the A record's name is
  // flock.local, not flock._flock._tcp.local. We do this by checking the
  // packet does NOT contain the instance name as the A record's name — the
  // A record's name should be 13 bytes (flock.local), not 25.
  //
  // Simplest check: the packet with hostPtr should be 12 bytes shorter than
  // one with instPtr (25 - 13 = 12), proving the A record uses the shorter
  // name. We can't build the wrong version anymore, but we CAN verify the
  // A record name is short enough to be flock.local by checking that the
  // total packet size matches the expected calculation.
  const typePtr = encodeName('_flock._tcp.local');
  const instPtr = encodeName('flock._flock._tcp.local');
  const hostPtr = encodeName('flock.local');

  // Expected: header(12) + PTR(typePtr + 10 + instPtr) + SRV(instPtr + 10 + srv) +
  //            TXT(instPtr + 10 + txt) + A(hostPtr + 10 + 4)
  const srvRdataLen = 6 + hostPtr.length; // priority(2) + weight(2) + port(2) + target
  // SRV rdata: 2+2+2 for pri/wt/port + hostPtr
  const expectedSize = 12 +
    typePtr.length + 10 + instPtr.length +        // PTR
    instPtr.length + 10 + srvRdataLen +            // SRV
    instPtr.length + 10 + (1 + 'model=test'.length + 1 + 'port=8000'.length) + // TXT
    hostPtr.length + 10 + 4;                       // A (uses hostPtr!)

  ok('packet size matches A-on-hostPtr calculation (not instPtr)',
     pkt.length === expectedSize,
     `expected ${expectedSize}, got ${pkt.length}`);
}

// --- malformed packets return null ----------------------------------------
{
  ok('empty packet returns null', parseResponse(new Uint8Array(0)) === null);
  ok('tiny packet returns null', parseResponse(new Uint8Array(5)) === null);
  ok('zeroed 100-byte packet returns null (no valid records)',
     parseResponse(new Uint8Array(100)) === null);
}

// --- buildQuery produces a valid query ------------------------------------
{
  const q = buildQuery();
  ok('query is longer than header', q.length > 12, `${q.length} bytes`);
  ok('query ID = 0', u16(q, 0) === 0);
  ok('query QDCOUNT = 1', u16(q, 4) === 1, `got ${u16(q, 4)}`);
  // QTYPE should be PTR (0x000c) — it's after the question name.
  // The name _flock._tcp.local encodes to: 6,_,f,l,o,c,k,7,_,4,_,t,c,p,5,l,o,c,a,l,0
  // = 21 bytes. QTYPE is at offset 12 + 21 = 33.
  const nameLen = 6 + 1 + 7 + 1 + 4 + 1 + 5 + 1; // rough; let's compute properly
  function encodeName(name) {
    const parts = name.split('.');
    const out = [];
    for (const p of parts) out.push(p.length, ...new TextEncoder().encode(p));
    out.push(0);
    return new Uint8Array(out);
  }
  const qName = encodeName('_flock._tcp.local');
  const qtypeOffset = 12 + qName.length;
  ok('query QTYPE = PTR (0x000c)', u16(q, qtypeOffset) === 0x000c,
     `got 0x${u16(q, qtypeOffset).toString(16)} at offset ${qtypeOffset}`);
  ok('query QCLASS = IN (0x0001)', u16(q, qtypeOffset + 2) === 0x0001,
     `got 0x${u16(q, qtypeOffset + 2).toString(16)}`);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
