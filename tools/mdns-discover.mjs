// flock mDNS discovery — find the coordinator on the local network.
//
// When FLOCK_URL is not set, sim_bird.mjs calls this to find the coordinator by
// querying _flock._tcp.local on the mDNS multicast address (224.0.0.251:5353).
// If a coordinator is advertising, its SRV + A records give us the host:port;
// if not, we fall back to http://127.0.0.1:8000 after the timeout.
//
// RUNS UNDER DENO (same as sim_bird.mjs). Uses Deno.listenDatagram for UDP.
// No npm dependencies — the mDNS query and response parser are both here.

const MDNS_ADDR = '224.0.0.251';
const MDNS_PORT = 5353;

function u16(buf, o) { return (buf[o] << 8) | buf[o + 1]; }

/** Parse a DNS name starting at offset o in buf, following compression pointers.
 *  Returns {name, offset} where offset is the position after the name. */
function parseName(buf, o) {
  let labels = [];
  let pos = o;
  let jumped = false;
  let endOffset = o;
  for (;;) {
    if (pos >= buf.length) break;
    const len = buf[pos];
    if (len === 0) {
      if (!jumped) endOffset = pos + 1;
      break;
    }
    if ((len & 0xc0) === 0xc0) {
      if (pos + 1 >= buf.length) break;
      const ptr = ((len & 0x3f) << 8) | buf[pos + 1];
      if (!jumped) { endOffset = pos + 2; jumped = true; }
      pos = ptr;
      continue;
    }
    pos++;
    labels.push(new TextDecoder().decode(buf.subarray(pos, pos + len)));
    pos += len;
  }
  return {name: labels.join('.'), offset: jumped ? endOffset : pos + 1};
}

/** Parse the answer + additional sections of an mDNS response, extracting the
 *  SRV port and A-record IP. Returns {port, ip, instance} or null. */
export function parseResponse(buf) {
  if (buf.length < 12) return null;
  const qdCount = u16(buf, 4);
  const anCount = u16(buf, 6);
  const arCount = u16(buf, 10);
  // Skip the question section (if any)
  let o = 12;
  for (let i = 0; i < qdCount; i++) {
    const n = parseName(buf, o);
    o = n.offset + 4; // name + QTYPE(2) + QCLASS(2)
  }

  let port = null;
  let ip = null;
  let serviceInstance = null;

  // Walk answer + additional records together (both contain useful records)
  const total = anCount + arCount;
  for (let i = 0; i < total; i++) {
    if (o >= buf.length) break;
    const n = parseName(buf, o);
    o = n.offset;
    if (o + 10 > buf.length) break;
    const rtype = u16(buf, o); o += 2;
    o += 2; // class
    o += 4; // TTL
    const rdLen = u16(buf, o); o += 2;
    const rdataStart = o;
    o += rdLen;

    if (rtype === 0x0001 && rdLen === 4) {
      // A record
      ip = buf[rdataStart] + '.' + buf[rdataStart+1] + '.' +
           buf[rdataStart+2] + '.' + buf[rdataStart+3];
    } else if (rtype === 0x0021) {
      // SRV record — its owner name IS the service instance (e.g.
      // flock._flock._tcp.local), not the service type. This takes priority
      // over the PTR owner name (which is _flock._tcp.local, the type).
      const srvPort = u16(buf, rdataStart + 4);
      const target = parseName(buf, rdataStart + 6);
      port = srvPort;
      serviceInstance = n.name;
    }
    // PTR records (0x000c) are skipped: the owner name is the service type,
    // not the instance. The SRV record carries the instance name.
  }

  if (port != null) return {port, ip: ip || null, instance: serviceInstance};
  return null;
}

function buildQuery() {
  // Minimal PTR query for _flock._tcp.local
  const name = '_flock._tcp.local';
  const parts = name.split('.');
  const enc = [];
  for (const p of parts) {
    enc.push(p.length, ...new TextEncoder().encode(p));
  }
  enc.push(0);
  const buf = new Uint8Array(12 + enc.length + 4);
  let o = 0;
  buf[o++] = 0; buf[o++] = 0;     // ID
  buf[o++] = 0; buf[o++] = 0;     // flags: standard query
  buf[o++] = 0; buf[o++] = 1;     // QDCOUNT
  buf[o++] = 0; buf[o++] = 0;     // ANCOUNT
  buf[o++] = 0; buf[o++] = 0;     // NSCOUNT
  buf[o++] = 0; buf[o++] = 0;     // ARCOUNT
  for (const b of enc) buf[o++] = b;
  buf[o++] = 0; buf[o++] = 0x0c;  // type PTR
  buf[o++] = 0; buf[o++] = 0x01;  // class IN
  return buf;
}

/** Discover a flock coordinator via mDNS.
 *
 *  Sends a PTR query for _flock._tcp.local on 224.0.0.251:5353 and waits for a
 *  response containing an SRV record (port) and optionally an A record (IP).
 *  Returns a base URL string like 'http://192.168.1.20:8000', or null if no
 *  coordinator was found within the timeout.
 *
 *  Uses Deno.listenDatagram — no npm dependencies. */
export async function discoverFlock(timeoutMs) {
  timeoutMs = timeoutMs || 3000;
  let sock;
  try {
    sock = Deno.listenDatagram({port: 0, hostname: '0.0.0.0', transport: 'udp'});
    try { sock.joinMulticastGroup(MDNS_ADDR); } catch {}
  } catch {
    return null; // no UDP — nothing to do
  }

  const query = buildQuery();
  try {
    sock.send(query, {hostname: MDNS_ADDR, port: MDNS_PORT});
  } catch {
    try { sock.close(); } catch {}
    return null;
  }

  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      try { sock.close(); } catch {}
      resolve(null);
    }, timeoutMs);

    (async () => {
      try {
        for await (const [data] of sock) {
          if (data.length > 12 && (data[2] & 0x80) !== 0) {
            const result = parseResponse(data);
            if (result && result.port != null) {
              clearTimeout(timer);
              try { sock.close(); } catch {}
              const host = result.ip || '127.0.0.1';
              resolve('http://' + host + ':' + result.port);
              return;
            }
          }
        }
      } catch { /* socket closed by timeout or success */ }
    })();
  });
}
