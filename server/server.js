// The coordinator: runs the whole flock with WebRTC on every link.
//
// It holds the embedding, layers 0..cut-1, output_norm and the vocab projection;
// each device (a "bird") holds a contiguous slice of the rest. Per token the
// hidden state makes one lap: coordinator -> bird -> bird -> back.
//
// The coordinator is itself a WebRTC peer: it offers a data channel to each bird,
// and once that opens the websocket carries nothing but signaling.
//
// RUN IT WITH DENO, NOT NODE (`npm start` does):
//
//   deno run --unstable-webgpu --allow-all server/server.js
//
// The coordinator runs its share of the layers as WGSL compute kernels, so it
// needs a GPU, and Node has no WebGPU -- no `navigator` at all, and no flag that
// adds one. Deno provides a device and also runs express, ws and node-datachannel
// unchanged through its node compatibility layer, so the whole server is one
// process on one runtime.
//
// THERE IS NO BUILD STEP. With GGUF a split is just a choice about byte ranges,
// so it is computed here at startup from the model's header: nothing is exported,
// nothing is written to disk, and the split can change without rebuilding anything.
import express from 'express';
import {WebSocketServer} from 'ws';
import {createServer} from 'node:http';
import {createServer as createHttpsServer} from 'node:https';
import {existsSync, readFileSync} from 'node:fs';
import {randomBytes} from 'node:crypto';
import {networkInterfaces} from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

import {Coordinator} from './coordinator.js';
import {Flock} from './mesh.js';
import {readModel} from './gguf.mjs';
import {layerPlan, Infeasible} from './allocate.js';
import {QWEN3_06B} from '../kernels/layer.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
process.chdir(ROOT);

// Where every device -- birds AND this coordinator -- reads its weights from.
// A plain URL rather than something we proxy, on purpose: the point of the GGUF
// path is that weights go from the model host straight to the device, so the
// coordinator never moves 67MB per bird and needs no build step and no disk.
const GGUF_URL = process.env.FLOCK_GGUF ||
  'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf';

// How the layers are divided. Only the model's HEADER is read to decide this --
// a few MB, not the weights -- which is what makes the topology a startup
// decision rather than a build artifact.
//
// THERE IS NO FLOCK_BIRDS ANY MORE, and removing it is the point. It fixed the
// number of devices at startup: a third phone pointed at a coordinator started
// with 2 got "flock full", and changing the count meant a restart. The birds' SHARE
// of the model is still a startup choice (FLOCK_BIRD_LAYERS, because the
// coordinator's half is loaded onto this GPU once and cannot move), but how many
// devices cover it, and which layers each one gets, is decided from whoever is
// present -- see src/allocate.js.
console.log(`reading GGUF header: ${GGUF_URL.split('/').pop()}`);
const header = await readModel(GGUF_URL);
const N_TOTAL = header.nLayers;
// How many layers the DEVICES get; the coordinator keeps the rest.
//
// This used to default to 4 -- a fossil from the ONNX era, when the coordinator's
// half was a pre-exported artifact and only a tiny tail was worth shipping to a
// phone. With streamed GGUF there is no such asymmetry, and leaving it at 4 means
// every device competes over layers 24-27 while the coordinator hoards 0-23, which
// looks exactly like an allocator that assigns everyone the same range.
//
// Default now: give the birds MOST of the model and keep a small coordinator
// share, since the coordinator also pays for the embedding and the LM head.
const COORD_LAYERS = Math.max(1, +(process.env.FLOCK_COORD_LAYERS || 4));
const BIRD_LAYERS = Math.min(N_TOTAL - 1,
  +(process.env.FLOCK_BIRD_LAYERS || (N_TOTAL - COORD_LAYERS)));
const CUT = N_TOTAL - BIRD_LAYERS;
// Per-layer BYTES and per-layer largest tensor, straight from the header. Not a
// layer count: Qwen3-14B Q4_K_M layers range 185.8-210.2 MB, a 13% spread, so an
// allocator that counts layers is off by that much before it starts.
const LAYERS = layerPlan(header, CUT, N_TOTAL - 1);

const META = {
  model: header.metadata['general.name'] || 'Qwen3-0.6B',
  n_total: N_TOTAL,
  hidden: QWEN3_06B.hidden,
  kv_heads: QWEN3_06B.nKvHeads,
  head_dim: QWEN3_06B.headDim,
  kv_cache: true,
  coord_layers: [0, CUT - 1],
};

console.log(`loading coordinator (layers 0-${CUT - 1}) on the GPU ...`);
const coord = await Coordinator.load(CUT);
const flock = new Flock(LAYERS, {
  onPlan: (p) => {
    for (const a of p.assign) {
      console.log(`  ${a.label} (${a.id}): layers ${a.start}-${a.end}  ` +
                  `${(a.bytes / 1e6).toFixed(1)}MB  ${(a.share * 100).toFixed(0)}%  ` +
                  `~${a.ms}ms`);
    }
    console.log(`  predicted per-token: ${p.makespanMs}ms  ` +
                `(an even split of the same layers: ${p.evenMs}ms)`);
  },
});
console.log(`coordinator holds layers 0-${CUT - 1}; birds cover ` +
            `${CUT}-${N_TOTAL - 1} (${(LAYERS.reduce((a, l) => a + l.bytes, 0) / 1e6)
              .toFixed(1)}MB), split across however many devices join`);
console.log('kv cache: ON  |  wire: f16 binary  |  links: webrtc  |  engine: wgsl');

// Drop devices we have not heard from. A fixed slot could sit red and wait for its
// phone to come back; a MEMBER cannot, because it is holding layers nobody runs, so
// the flock stays uncovered until the device is actually removed and its layers are
// given to someone else.
const SWEEP_MS = 5000;
setInterval(() => {
  const gone = flock.sweep();
  if (!gone.length) return;
  console.log(`dropped ${gone.length} unresponsive device(s): ${gone.join(', ')}`);
  // Mid-token this has to wait: reassigning now would move layers whose K/V cache
  // the current token is still writing into.
  if (convo.busy) { flock.defer(); return; }
  rebalance({force: true, why: 'a device stopped answering'});
}, SWEEP_MS);

/**
 * Recompute the assignment and tell everyone. Called on join, on leave, on a
 * sweep, and at token boundaries; `force` is for membership changes, where the
 * alternative is not covering some layers at all.
 *
 * ANY reassignment drops the conversation. The K/V cache is sharded BY LAYER across
 * the devices, so a layer that moves leaves its keys on the device that no longer
 * holds it -- and the device that now does starts from an empty cache at a position
 * offset the rest of the flock believes is already filled. Continuing would mix
 * keys for the same positions computed on two different devices, which is text that
 * is wrong without looking wrong. Re-prefilling the history instead is not
 * available: Coordinator.turnTokens documents why re-encoding a conversation
 * DIVERGES from what was actually fed. So the honest cost of adding or losing a
 * device is the conversation's context, said out loud, rather than silent garbage.
 */
function rebalance({force = false, why = ''} = {}) {
  const d = flock.plan({force});
  if (d.move) {
    if (why) console.log(`rebalancing: ${why}`);
    console.log(`  ${d.reason}`);
    if (d.moved.length) {
      resetConvo();
      lastRebalance = {at: Date.now(), reason: d.reason, moved: d.moved,
                       dropped_context: true};
    }
    flock.announce();
  }
  if (flock.infeasible) console.error(`cannot place layers: ${flock.infeasible.message}`);
  return d;
}
// The last reallocation that cost the conversation its context, so the chat page
// can say WHY the context went to zero rather than appearing to forget.
let lastRebalance = null;

/**
 * Why this specific device cannot be in the flock, named precisely.
 *
 * The allocator's own message is about the FLOCK ("no device can hold layer 40"),
 * which is the right message when nobody can hold a layer. When one device is the
 * problem, the device needs to be told about ITSELF -- and told with the tensor name
 * and both numbers, because "this device cannot participate" gives a phone's owner
 * nothing to act on.
 */
function whyRefused(bird) {
  const mb = n => `${(n / 1e6).toFixed(1)}MB`;
  const tooBig = LAYERS.filter(l => l.maxTensor > bird.caps.bind);
  if (tooBig.length) {
    const l = tooBig[0];
    return `this device cannot hold layer ${l.layer}: its largest tensor ` +
      `${l.biggest} is ${mb(l.maxTensor)}, but this device reported a ` +
      `maxStorageBufferBindingSize of ${mb(bird.caps.bind)}. One tensor cannot be ` +
      `split across devices, so ${tooBig.length === LAYERS.length
        ? 'there is no layer in this model this device could take'
        : `${tooBig.length} of the ${LAYERS.length} bird layers are out of reach`}. ` +
      `Open /check on this device to see its limits.`;
  }
  const smallest = Math.min(...LAYERS.map(l => l.bytes));
  if (bird.caps.budget < smallest) {
    return `this device's memory budget is ${mb(bird.caps.budget)}, and the ` +
      `smallest layer in this model is ${mb(smallest)}, so there is nothing it ` +
      `could hold.`;
  }
  // It fits on its own but not alongside the devices already here: with the chain
  // held in order, there is no run this device can take.
  return `this device cannot be fitted into the chain alongside the ` +
    `${flock.members().length} device(s) already in the flock: with layers held in ` +
    `order, every contiguous run it could take is blocked by its limits ` +
    `(${mb(bird.caps.bind)} per tensor, ${mb(bird.caps.budget)} budget). ` +
    `There are ${LAYERS.length} bird layers in total.`;
}

// The LAN address is the one thing you cannot guess, and you need it to open
// /flock on a phone. Print it rather than making the user go find it.
function localAddresses() {
  return Object.values(networkInterfaces()).flat()
    .filter(i => i && i.family === 'IPv4' && !i.internal).map(i => i.address);
}

const app = express();
app.use(express.json());
app.use('/js', express.static('web/js'));

// A bird runs the SAME kernels the tests validate, fetched from here rather than
// carrying a copy -- so a kernel fix reaches every bird on reload, and there is
// no second implementation to keep in step with kernels/.
//
// The .wgsl files go out as-is: wgslSource() in kernels/layer.ts resolves them
// against its own module URL, which is a file: URL under Deno and an http: one in
// the page, so one expression serves both runtimes.
//
// The .ts files cannot go out as-is, because browsers do not execute TypeScript.
// They are transpiled ON REQUEST and their `./x.ts` import specifiers rewritten to
// `./x.ts.js`, so the browser follows the same module graph Deno does. This is
// deliberately not a build step: nothing is written to disk, there is no artifact
// to rebuild or forget to rebuild, and the file the bird runs is derived from the
// file the tests run every time it is asked for. Transpiling is type STRIPPING
// only -- no type checking, which is what `deno check` is for -- and the result is
// cached in memory per mtime so a reload does not re-emit.
const TS_CACHE = new Map();
app.get(/^\/kernels\/(.+\.ts)\.js$/, async (req, res) => {
  const rel = req.params[0];
  // Only the kernels directory, and no traversal out of it: `rel` reaches the
  // filesystem, so a crafted path must not escape.
  if (rel.includes('..') || !/^[\w./-]+$/.test(rel)) {
    return res.status(400).type('text/plain').send('bad kernel path');
  }
  const abs = path.join(ROOT, 'kernels', rel);
  try {
    const {mtimeMs} = await import('node:fs/promises').then(fs => fs.stat(abs));
    const hit = TS_CACHE.get(abs);
    if (hit && hit.mtimeMs === mtimeMs) {
      return res.type('application/javascript').send(hit.js);
    }
    const {transpile} = await import('jsr:@deno/emit');
    const url = pathToFileURL(abs);
    const out = await transpile(url);
    // Rewrite relative .ts specifiers to the .ts.js route above, so the graph
    // resolves in the browser. Only RELATIVE ones: a bare specifier would be a
    // dependency the page has no import map for, and silently rewriting it would
    // produce a 404 that looks like a missing kernel.
    // kernels/ sits next to web/ on disk, so a kernel importing
    // "../web/js/wire.mjs" is correct as a FILE path but a 404 as a URL: that
    // file is served at /js/, not /web/js/. Left alone it takes down the whole
    // module graph before any page code runs, which looks like a page that
    // simply never loads.
    const js = out.get(url.href)
      .replace(/(from\s*["'])(\.\.?\/[^"']+\.ts)(["'])/g, '$1$2.js$3')
      .replace(/(import\s*\(\s*["'])(\.\.?\/[^"']+\.ts)(["']\s*\))/g, '$1$2.js$3')
      .replace(/(["'])\.\.\/web\/js\//g, '$1/js/');
    TS_CACHE.set(abs, {mtimeMs, js});
    res.type('application/javascript').send(js);
  } catch (e) {
    console.error(`[kernels] ${rel}: ${e.message}`);
    res.status(404).type('text/plain').send(`cannot serve ${rel}: ${e.message}`);
  }
});
app.use('/kernels', express.static('kernels'));

// Claim a place in the flock. There is no "full" any more: a device that turns up
// is capacity, and what it changes is the assignment, not whether it is let in.
//
// `caps` is what the device measured about ITSELF with the /check probe -- its real
// maxStorageBufferBindingSize above all, which is the one limit no split can work
// around. A device that reports nothing is treated as unconstrained, because
// refusing it outright would be worse than the OOM it might hit, and an OOM comes
// back through /diag either way.
app.post('/join', (req, res) => {
  // IDENTITY IS THE COORDINATOR'S TO ASSIGN. It used to be whatever the client
  // sent, kept in localStorage across reloads, which is wrong twice over: a stale
  // id let one device silently take over another's slot, and two devices that
  // both report navigator.platform "MacIntel" (a Mac and an iPad do) were
  // indistinguishable in the logs while knocking each other out of the flock.
  //
  // Every /join is a NEW member, with ONE exception. A reload is a new device as
  // far as the flock is concerned, and the old registration is reaped by the
  // liveness sweeper -- the correct reading: the page that held those layers is
  // gone, along with the K/V cache for them.
  //
  // The exception is a RETRY. A device that joins while a turn is running is
  // answered `wait` and told to ask again with the id it was given; that id was
  // minted here a moment ago and the device has not connected yet, so it is the
  // same device asking the same question. Minting a fresh id for every retry
  // admitted one member per retry -- measured: a bird waiting out one 30-token
  // turn became 27 members, the allocator refused ("27 devices but only 8
  // layers"), and the sweeper spent the next minute dropping phantoms, which read
  // in the log as joined -> dropped -> joined. A stale id from some other source
  // is still refused: only an id of a member that has never connected is reused.
  const retry = req.body.peer_id ? flock.byPeer(String(req.body.peer_id)) : null;
  const pid = retry && !retry.everLinked ? retry.peerId : randomBytes(8).toString('hex');
  // A join can still break the flock: a device reporting a binding limit too small
  // for any layer makes every assignment infeasible. Keep enough to roll back.
  const was = flock.byPeer(pid);
  const prevCaps = was ? {...was.caps} : null;
  // A bird reaching us over loopback is running on this very machine, so its GPU
  // is the same one doing the embedding and the LM head. It still gets layers --
  // wasting the fastest hardware on the network would be foolish -- but the
  // allocator has to know, or it double-counts that GPU. See Flock.reallocate().
  const from = (req.socket?.remoteAddress || '').replace('::ffff:', '');
  const onCoordinator = from === '127.0.0.1' || from === '::1' || from === 'localhost';
  const bird = flock.claim(pid, req.body.label || 'phone', req.body.caps || null,
                           {onCoordinator});
  // Who is actually joining, and is this a new device or a returning one? Two
  // devices reporting the same navigator.platform ("MacIntel" for both a Mac and
  // an iPad) are indistinguishable in a log line without this.
  console.log(`[join] ${bird.label} peer=${pid} from=${from || '?'} ` +
              `cores=${bird.caps.cores ?? '?'} ` +
              `bind=${Math.round((bird.caps.bind || 0) / 1e6)}MB ` +
              `${was ? 'REJOIN' : 'new'} -> now ${flock.members().length} member(s)`);

  // A join mid-token cannot take effect mid-token: the running token is filling
  // K/V caches on the devices that hold those layers now. Stage it and let the
  // token finish. The device waits one token, which is milliseconds.
  const deferred = convo.busy;
  if (deferred) flock.defer();
  else rebalance({force: true, why: `${bird.label} joined`});

  if (flock.infeasible) {
    // THIS JOIN IS WHAT BROKE IT -- so undo it, rather than letting one bad device
    // poison a flock that was working. Measured: a phone reporting a 1MB binding limit
    // joined, made every assignment infeasible, and stayed a member forever, so the
    // flock never recovered even after a capable device arrived. A device that cannot
    // be part of a working flock is not a member of it.
    //
    // Two shapes of undo, because the two arrivals are different. A NEW device is
    // removed outright. A REJOINING one keeps its membership and gets its old caps
    // back: its layers and its loaded weights are still good under the numbers it was
    // admitted with, and throwing it out for reporting a worse limit on a reload would
    // take a working device out of the flock to punish it for being honest.
    const refuse = reason => {
      flock.announce();
      console.log(`refused ${bird.label} (${pid}): ${reason}`);
      return res.status(409).json({
        error: whyRefused(bird), detail: {kind: 'device-cannot-participate'},
        peer_id: pid, wait: false});
    };
    if (!prevCaps) {
      flock.release(pid);
      const after = flock.plan({force: true});
      if (!flock.infeasible) return refuse(after.reason);
    } else {
      const rejected = whyRefused(bird);
      flock.setCaps(bird, prevCaps);
      const after = flock.plan({force: true});
      if (!flock.infeasible) {
        flock.announce();
        console.log(`kept ${bird.label} (${pid}) on its previous limits: ${rejected}`);
        // It stays in the flock on the range it already has, and is told why the new
        // numbers were not taken -- silently ignoring them would leave a device
        // believing a limit the coordinator is not planning against.
        return res.status(409).json({
          error: `${rejected} The coordinator kept this device on the limits it ` +
                 `joined with, so its current layers are unchanged. Reload to try ` +
                 `again, or open /check to see what this device now reports.`,
          detail: {kind: 'caps-rejected'}, peer_id: pid, wait: false});
      }
    }
    // Say so BEFORE the device downloads anything. This is the whole reason the
    // capability probe happens before the weights: the alternative is a phone
    // pulling gigabytes and then failing to allocate a buffer.
    return res.status(409).json({
      error: flock.infeasible.message, detail: flock.infeasible.detail,
      peer_id: pid, wait: false});
  }
  if (!bird.placed()) {
    // Admitted but not yet holding layers -- only happens when a token is in
    // flight. Tell the device to ask again rather than guessing a range for it.
    //
    // `peer_id` matters here and got this wrong once: a device that joins without one
    // is given a fresh id, and if it then retried WITHOUT sending that id back it was
    // admitted again as a SECOND member. Measured: three retries produced three
    // members with the same label, the allocator gave each a slice, and the flock had
    // two phantom devices it would wait forever on. So the id is returned on the
    // `wait` answer too, and callers must send it back.
    return res.json({peer_id: pid, wait: true, retry_ms: 1200,
                     reason: deferred ? 'a token is in flight; layers are assigned '
                       + 'at the next token boundary' : 'no layers assigned yet',
                     gguf: GGUF_URL, model: META.model, n_total: META.n_total});
  }
  res.json({peer_id: pid, slot: bird.slot, start: bird.start, end: bird.end,
            n_layers: bird.end - bird.start + 1, hidden: META.hidden,
            kv_heads: META.kv_heads, head_dim: META.head_dim,
            kv_cache: META.kv_cache, n_total: META.n_total, model: META.model,
            gguf: GGUF_URL, why: bird.why, bytes: bird.bytes,
            devices: flock.members().length});
});

/** Leave on purpose. A device that says so gets its layers handed on immediately
 *  instead of costing the flock the 40s liveness grace period first. */
app.post('/leave', (req, res) => {
  const pid = req.body?.peer_id;
  if (!pid) return res.status(400).json({error: 'no peer_id'});
  const bird = flock.byPeer(pid);
  if (!bird) return res.status(404).json({error: 'not a member of this flock'});
  const label = bird.label, range = bird.placed() ? `${bird.start}-${bird.end}` : 'none';
  // Close the socket, not just the membership. Without this the device's heartbeat
  // keeps arriving on a Bird that is no longer in the flock -- harmless but invisible,
  // and the page would go on showing a range it does not hold. A closed socket makes
  // the page's own reconnect logic the thing that decides what happens next.
  try { bird.ws?.close(); } catch {}
  flock.release(pid);
  if (convo.busy) flock.defer();
  else rebalance({force: true, why: `${label} left (held ${range})`});
  res.json({ok: true, left: pid, held: range,
            devices: flock.members().length,
            ready: flock.ready(), missing: flock.missing(),
            infeasible: flock.infeasible});
});

/** Throw a device out from the coordinator side -- a phone whose tab is asleep and
 *  whose heartbeat is stalling the flock, where nobody can reach the phone itself. */
app.post('/evict', (req, res) => {
  const pid = req.body?.peer_id;
  if (!pid) return res.status(400).json({error: 'no peer_id'});
  const bird = flock.byPeer(pid);
  if (!bird) return res.status(404).json({error: 'not a member of this flock'});
  try { bird.ws?.close(); } catch {}
  flock.release(pid);
  if (convo.busy) flock.defer();
  else rebalance({force: true, why: `${bird.label} evicted`});
  res.json({ok: true, evicted: pid, devices: flock.members().length,
            ready: flock.ready(), missing: flock.missing()});
});

// Birds have no readable console, so they POST failures here. Kept in a ring
// buffer as well as logged: /status serves it, so the chat page can show what a
// phone reported without anyone reading the terminal.
const DIAG = [];
app.post('/diag', (req, res) => {
  const {stage, detail, ua, slot} = req.body || {};
  const dev = /iPad/.test(ua) ? 'iPad' : /iPhone/.test(ua) ? 'iPhone'
            : /Android/.test(ua) ? 'Android'
            : /Macintosh/.test(ua) ? 'Mac/iPad' : 'device';
  const text = typeof detail === 'object' ? JSON.stringify(detail) : String(detail);
  console.log(`[diag ${dev}${slot != null ? ` slot ${slot}` : ''}] ${stage}: ${text}`);
  DIAG.push({at: Date.now(), device: dev, slot: slot ?? null,
             stage: String(stage || '?'), detail: text.slice(0, 400)});
  if (DIAG.length > 60) DIAG.shift();
  res.json({ok: true});
});

// Serve relative to ROOT, not as an absolute path: send() rejects any path with
// a dot-prefixed segment as a hidden file, so an absolute path through a
// checkout under e.g. ~/.worktrees/ 404s every page. With `root` set it only
// inspects the relative part, which also keeps a crafted :slot from escaping.
const page = rel => (_, res) => res.sendFile(rel, {root: ROOT});
app.get('/', page('web/chat.html'));
app.get('/flock', page('web/bird.html'));
// A standalone hardware report, and the same probe a bird runs before it joins:
// one probe, one set of numbers, so what the allocator plans against is what a
// human can open in a browser and read. Separate from /flock because it must work
// even when the bird page cannot: it claims no slot and loads no weights.
app.get('/check', page('web/inspect.html'));

app.get('/status', (_, res) => res.json({
  // `ready` means a lap could start now: every layer assigned to a live device AND
  // every device has confirmed it holds what it was assigned. `covered` is the
  // first half alone; the difference is the devices still streaming weights,
  // listed under allocation.loading.
  ready: flock.ready(), covered: flock.covered(), missing: flock.missing(),
  pending: flock.pending().map(b => ({peer_id: b.peerId, label: b.label,
                                       range: `${b.start}-${b.end}`})),
  coord_layers: `0-${CUT - 1}`, coord_n_layers: CUT,
  birds: flock.birds.map(b => b.info()),
  n_total: META.n_total, hidden: META.hidden, kv_cache: META.kv_cache,
  wire: 'f16', runtime: 'deno', engine: 'wgsl', model: META.model,
  // WHY the split is what it is: per-device bytes and rate are on each bird, and
  // this is the flock-level view -- the predicted per-token cost, what an even
  // split of the same layers would have cost with the same measured rates, and the
  // last decision the rebalancer made or refused, in its own words. A
  // speed-weighted allocator whose reasoning is invisible cannot be told apart
  // from a broken one.
  allocation: flock.allocation(),
  // Set when a reallocation dropped the conversation, so the chat page can explain
  // a context that went to zero instead of appearing to have forgotten.
  last_rebalance: lastRebalance,
  // Conversation state, so the chat UI can show how much context is cached and
  // whether a turn is already in flight.
  cached_tokens: convo.fed, turns: convo.turns, busy: convo.busy,
  diag: DIAG.slice(-12),
}));

// Liveness for scripts and for `curl` on a phone: one object, no model load, and
// an HTTP status that says whether the flock could answer a prompt right now.
app.get('/health', (_, res) => {
  const birds = flock.birds.map(b => b.info());
  const ok = flock.ready();
  res.status(ok ? 200 : 503).json({
    ok, uptime_s: +process.uptime().toFixed(0),
    missing: flock.missing(),
    birds_alive: birds.filter(b => b.alive).length, birds_total: birds.length,
    // No "of N" -- there is no N. A covered flock is one whose layers are all
    // answered for, whatever the device count is.
    devices: flock.members().length,
    makespan_ms: flock.makespanMs,
    infeasible: flock.infeasible ? flock.infeasible.message : null,
    busy: convo.busy, cached_tokens: convo.fed,
    rss_mb: +(process.memoryUsage().rss / 1e6).toFixed(0),
  });
});

// Conversation state. `fed` is how many tokens the caches have seen, which is
// also the next position id — the coordinator's cache and every bird's cache all
// advance together, so one counter describes the whole flock.
const convo = {fed: 0, turns: 0, busy: false};

/** Drop every cache in the flock and start the conversation over. */
function resetConvo() {
  coord.reset();
  flock.reset();     // each bird clears its own shard of the cache on next frame
  convo.fed = 0;
  convo.turns = 0;
}

app.post('/reset', (_, res) => {
  if (convo.busy) return res.status(409).json({error: 'a turn is in flight'});
  resetConvo();
  // A conversation just ended, so there is nothing to lose by moving layers: this
  // is one of the two moments the speed rebalancer is allowed to act.
  const d = rebalanceForSpeed('the conversation was cleared');
  res.json({ok: true, rebalanced: d?.move ? d.reason : null});
});

/**
 * Let the timings move layers -- ONLY when no conversation would be lost.
 *
 * Every token records the devices' rates (Flock.recordTimings), but acting on
 * them drops the sharded K/V cache, so a speed rebalance never fires during a
 * conversation: not mid-turn, and not between the turns of one either. It is
 * tried at the two moments there is no context to lose -- /reset, and a /chat
 * that starts over -- under the gain gate and cooldown in speed.js, which refuse
 * far more often than they allow.
 */
function rebalanceForSpeed(why) {
  if (convo.busy || convo.fed > 0) return null;
  const d = flock.plan();
  if (d.move && d.moved.length) {
    console.log(`rebalancing for speed (${why}): ${d.reason}`);
    lastRebalance = {at: Date.now(), reason: d.reason, moved: d.moved,
                     dropped_context: false};
    flock.announce();
  }
  return d;
}

/** How long a turn will wait for reassigned devices to finish streaming before
 *  giving up. A phone re-streams 67MB in 3-18s measured; a first download of a
 *  larger share can take longer, so this is generous rather than tight. */
const READY_WAIT_MS = +(process.env.FLOCK_READY_WAIT_MS || 90000);

app.post('/chat', async (req, res) => {
  const {prompt, max_tokens = 96, reset = false} = req.body;
  res.set({'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache',
           'X-Accel-Buffering': 'no'});
  const sse = o => res.write(`data: ${JSON.stringify(o)}\n\n`);

  if (!prompt || !String(prompt).trim()) {
    sse({type: 'error', text: 'empty prompt'});
    return res.end();
  }
  // An unsatisfiable flock is a different failure from an uncovered one, and the
  // fixes are opposite: uncovered wants another device, unsatisfiable wants a
  // different one. Reported separately so the message names the actual fix.
  if (flock.infeasible) {
    sse({type: 'error', text: flock.infeasible.message});
    return res.end();
  }
  // Assigned, not covered: a device that was just admitted and is still opening
  // its socket counts, because the readiness wait below will hold the turn for it.
  if (!flock.assigned()) {
    const miss = flock.missing();
    sse({type: 'error', text: miss.length
      ? `waiting for devices to cover layer${miss.length > 1 || miss[0].includes('-')
          ? 's' : ''} ${miss.join(', ')} — open /flock on each phone and tap join`
      : 'waiting for the devices that hold the layers to answer'});
    return res.end();
  }
  // One conversation, one cache, no batching: a second concurrent turn would
  // interleave writes into the same K/V and corrupt both answers.
  if (convo.busy) {
    sse({type: 'error', text: 'the flock is already generating — one turn at a time'});
    return res.end();
  }
  convo.busy = true;

  // Stop generating if the browser goes away mid-turn. Watch the RESPONSE, not
  // the request: express.json() consumes and destroys the request stream while
  // parsing the body, so req.destroyed is already true by the time we get here
  // and would abort every turn after its first token.
  let gone = false;
  res.on('close', () => { gone = true; });

  try {
    if (reset) {
      resetConvo();
      // Starting over is the other moment nothing is lost by moving layers.
      convo.busy = false;
      rebalanceForSpeed('a new conversation started');
      convo.busy = true;
    }
    // THE READINESS GATE. Devices that were just (re)assigned are streaming their
    // weights and hold nothing yet; a frame sent now would arrive at a device
    // with no layers. Wait for every placed device to confirm, saying who is
    // still loading so the chat page can show it rather than appear hung.
    const readiness = await flock.awaitReady({
      timeoutMs: READY_WAIT_MS,
      onWait: (pend, waited) => sse({type: 'waiting', waited_ms: waited,
        pending: pend.map(b => ({label: b.label, range: `${b.start}-${b.end}`,
                                 peer_id: b.peerId}))}),
    });
    if (!readiness.ok) {
      sse({type: 'error', text: readiness.reason});
      return res.end();
    }
    // Append this turn to the warm cache rather than re-prefilling the history.
    // See Coordinator.turnTokens for why re-encoding would be wrong here.
    let stepIds = coord.turnTokens(prompt, convo.turns === 0);
    let offset = convo.fed;
    const promptTokens = stepIds.length;
    const out = [];
    let stop = 'length';

    sse({type: 'turn', turn: convo.turns, prompt_tokens: promptTokens,
         cached_tokens: offset});

    for (let step = 0; step < max_tokens; step++) {
      if (gone) { stop = 'aborted'; break; }

      const t0 = performance.now();
      let flat = await coord.forward(stepIds, offset);
      const coordMs = +(performance.now() - t0).toFixed(1);
      // Remember what the coordinator's own share costs per token, so a bird
      // sharing this GPU can be charged for it in the next allocation.
      flock.noteCoordMs(coordMs);

      const t1 = performance.now();
      try {
        // One lap: into the first bird, out of the last. Birds forward to each
        // other directly, so the coordinator sees a single round trip no matter
        // how many devices are in the chain.
        const {data} = await flock.lap(flat, stepIds.length, META.hidden, offset);
        flat = data;
      } catch (e) {
        // A bird dropping mid-turn, or refusing a frame, leaves the caches at an
        // offset no device agrees on any more, so the conversation cannot be
        // continued. What it does NOT do is remove the device: a device that is
        // between assignments, or that failed one frame, is still a member. Only a
        // device that fails to compute repeatedly has shown it cannot serve.
        resetConvo();
        if (e.nack && e.bird) {
          const b = e.bird;
          if (e.nack.reason === 'failed' && b.failures >= 3) {
            console.log(`[flock] ${b.label} failed ${b.failures} frames in a row; ` +
                        `removing it (held ${b.start}-${b.end})`);
            flock.release(b.peerId);
            flock.defer();
          }
        }
        sse({type: 'error', text: e.message});
        return res.end();
      }
      const stats = flock.chain().map(b => ({
        slot: b.slot, range: `${b.start}-${b.end}`, ms: b.lastMs,
        label: b.label, transport: b.transport}));
      const netMs = +(performance.now() - t1).toFixed(1);

      const nxt = await coord.project(flat, stepIds.length);
      const nFloats = stepIds.length * META.hidden;

      // Positions are consumed whether or not we keep the token, so advance the
      // cache offset before deciding to stop -- otherwise the next turn would
      // feed overlapping position ids into a cache that already holds them.
      offset += stepIds.length;
      convo.fed = offset;

      sse({type: 'hop', step, coord_ms: coordMs, birds: stats,
           roundtrip_ms: netMs, prefill: step === 0,
           seq: stepIds.length, cached: offset,
           kb: +(nFloats * 2 / 1024).toFixed(1),
           kb_json: +(nFloats * 21 / 1024).toFixed(1)});

      if (nxt === coord.meta.eos) { stop = 'eos'; break; }
      out.push(nxt);
      sse({type: 'token', text: coord.decode([nxt])});
      stepIds = [nxt];

      // ---- THE TOKEN BOUNDARY -------------------------------------------------
      // Timings are recorded here, where a rate's two halves (bytes held, ms
      // taken) belong to the same token. Nothing MOVES here. The assignment used
      // to be allowed to change at this boundary, which ended the answer with
      // stop:'rebalanced' whenever a device joined; now a join waits for the turn
      // to finish (see the `finally` below) and a speed rebalance waits for a
      // conversation to start, so an answer that has begun always completes.
      flock.recordTimings();
    }

    // An aborted turn stops mid-answer: the caches hold a partial assistant
    // reply with no closing token, so the next turn cannot append cleanly.
    // Start over rather than continue from a torn conversation.
    if (stop === 'aborted') {
      resetConvo();
      return res.end();
    }

    // The generated tokens are in the cache now, so the next turn continues from
    // here. Count the turn only if it produced something to continue from.
    if (out.length) convo.turns++;
    sse({type: 'done', text: coord.decode(out), stop,
         tokens: out.length, prompt_tokens: promptTokens, cached: convo.fed,
         turn: convo.turns});
    res.end();
  } catch (e) {
    console.error(`[chat] ${e.stack || e.message}`);
    resetConvo();
    try { sse({type: 'error', text: `coordinator failed: ${e.message}`}); } catch {}
    try { res.end(); } catch {}
  } finally {
    convo.busy = false;
    // THE END OF A TURN IS WHERE MEMBERSHIP CHANGES LAND. A device that joined or
    // left while the answer was being generated was deferred; it is applied now,
    // as a sticky edit, so the devices that need not move do not. The next turn
    // then waits for the ones that did move to confirm their new layers.
    //
    // Doing it here rather than at a token boundary is what lets an answer
    // complete: the K/V cache is sharded by layer, so a move mid-answer had to end
    // the answer. The context is still dropped -- that is the documented cost of a
    // reassignment -- but the user gets their whole reply first.
    if (flock.deferred) {
      const d = flock.applyPending();
      if (d?.move) {
        console.log(`applied a deferred membership change: ${d.reason}`);
        if (d.moved.length) {
          resetConvo();
          lastRebalance = {at: Date.now(), reason: d.reason, moved: d.moved,
                           dropped_context: true};
        }
        flock.announce();
      }
    }
  }
});

// Chrome and Edge expose WebGPU only in a SECURE CONTEXT, so a bird reached over
// plain http:// on a LAN sees no navigator.gpu at all -- which looks exactly like
// a device with no GPU and is not. Safari does not gate it this way, which is why
// it works where Chrome cannot. Serving https with a self-signed cert makes every
// browser usable; the cert has to be trusted once per device.
const CERT = path.join(ROOT, '.certs', 'cert.pem');
const KEY = path.join(ROOT, '.certs', 'key.pem');
// FLOCK_NO_TLS is for the test runner: it starts a coordinator on a spare port
// and talks to it from Node and Deno clients that would each need their own way
// of trusting a self-signed certificate. Plain http on loopback needs none.
const hasCert = !process.env.FLOCK_NO_TLS && existsSync(CERT) && existsSync(KEY);
const SCHEME = hasCert ? 'https' : 'http';
const server = hasCert
  ? createHttpsServer({cert: readFileSync(CERT), key: readFileSync(KEY)}, app)
  : createServer(app);
const wss = new WebSocketServer({server, path: '/ws'});

wss.on('connection', ws => {
  let bird = null;
  ws.on('message', (data, isBinary) => {
    if (!isBinary) {
      const m = JSON.parse(data.toString());
      if (m.peer_id && !bird) {                 // hello
        bird = flock.byPeer(m.peer_id);
        if (!bird) return ws.send(JSON.stringify({error: 'unknown peer — rejoin'}));
        bird.ws = ws; bird.label = m.label || bird.label;
        // Do NOT clobber an open data channel. `transport` is what actually
        // carries frames, and a hello can arrive after the channel is up (a
        // reconnect, or a reassignment that re-runs the handshake) -- writing 'ws'
        // here made the coordinator report 'ws' while the device correctly showed
        // 'webrtc', because the page reads its real channel state and this field
        // was just the last thing written.
        if (!bird.chanOpen()) bird.transport = 'ws';
        bird.lastSeen = Date.now();
        // A device that was admitted but not placed (it joined mid-token) becomes
        // placeable the moment it is answering. Between turns that is now.
        if (!bird.placed() && !convo.busy) {
          rebalance({force: true, why: `${bird.label} connected`});
        }
        // ALWAYS tell this bird its current range, even if it has not changed.
        // /join answers with the range at that instant, but a later device's join
        // reallocates -- and announce() skips anyone without a socket yet, so the
        // first device was told 4-27 at join, silently moved to 4-15, and never
        // heard about it. Two devices then displayed 4-27 and 16-27, which looks
        // like overlapping assignment and would compute layers 16-27 twice.
        const mine = flock.chainFor(bird.peerId);
        if (mine) try { ws.send(JSON.stringify(mine)); } catch {}
        // Offer the bird a direct data channel; the websocket then only
        // carries signaling.
        bird.openRTC(b => console.log(`webrtc link open to ${b.label} (${b.start}-${b.end})`));
        flock.announce();          // tell everyone who their successor is
        return;
      }
      if (!bird) return;
      if (m.t === 'signal') {
        // `to` means bird->bird: we are only the introducer, exactly the role
        // PeerJS plays for swarmllm. No `to` means it is meant for us.
        if (m.to) {
          const dst = flock.byPeer(m.to);
          if (dst?.ws) dst.ws.send(JSON.stringify(
            {t: 'signal', from: bird.peerId, data: m.data}));
        } else {
          bird.onSignal(m.data);
        }
      }
      // A bird's own per-token time. Recorded here, turned into a bytes/ms rate at
      // the token boundary in /chat -- not here, because a `stats` message can
      // arrive twice for one token (websocket and data channel both deliver) and
      // double-counting would bias the EWMA toward whichever bird is chattiest.
      else if (m.t === 'stats') { bird.lastMs = m.ms; bird.lastSeen = Date.now(); }
      // THE READINESS HANDSHAKE. The device has streamed and built the layers for
      // [start, end] and can take frames for them. Until this arrives for the
      // range it is assigned, no lap starts -- see Flock.awaitReady.
      else if (m.t === 'ready') {
        const was = bird.isReady();
        const now = bird.confirm(m.start, m.end);
        bird.lastSeen = Date.now();
        if (now && !was) {
          console.log(`[ready] ${bird.label} holds layers ${m.start}-${m.end}` +
                      (flock.ready() ? ' -- the flock is ready' :
                       ` -- still waiting for ${flock.pending().map(b => b.label).join(', ')}`));
        } else if (!now) {
          console.log(`[ready] ${bird.label} confirmed ${m.start}-${m.end} but is ` +
                      `assigned ${bird.start}-${bird.end}; waiting for that one`);
        }
      }
      // The device refused a frame -- it is between assignments, or it tried to
      // compute and could not. Either way the lap waiting on it is told at once
      // instead of at the 120s timeout, and the device is NOT removed for it.
      else if (m.t === 'nack') {
        console.log(`[nack] ${bird.label} (${bird.start}-${bird.end}): ${m.reason}` +
                    (m.detail ? ` -- ${m.detail}` : ''));
        bird.nack({...m, peer_id: bird.peerId});
      }
      // A device can update what it knows about itself after joining -- the /check
      // probe finishing, or the real device's limits differing from the adapter's.
      else if (m.t === 'caps' && m.caps) {
        flock.setCaps(bird, m.caps);
        if (!convo.busy) rebalance({force: true, why: `${bird.label} reported limits`});
      }
      // Whether this bird can hand off peer-to-peer decides how far the
      // coordinator's chained wait should reach.
      else if (m.t === 'forwards') bird.forwardsDirectly = !!m.direct;
      // note: transport is set from the send path in mesh.js, not from here --
      // what actually carried the frame is the only honest answer.
      // Liveness heartbeat. Without it an idle bird ages past alive()'s grace
      // period and the flock reports itself uncovered until a turn starts.
      // `pull` is the old name from the long-polling transport -- still accepted
      // so an un-refreshed bird page keeps its slot.
      else if (m.t === 'ping' || m.t === 'pull') bird.lastSeen = Date.now();
    } else if (bird) {
      bird.deliver(data);
    }
  });
  ws.on('close', () => {
    if (!bird) return;
    // A refresh closes the websocket; the data channel it signalled is dead
    // too, so tear the whole link down rather than leaving chan dangling.
    if (bird.ws !== ws) return;
    bird.teardown();
    flock.announce();
    // Do NOT remove it from the flock here. A refresh and a departure look
    // identical at this point, and a refreshing phone comes back with the same
    // peer id within a second or two -- reassigning immediately would move layers
    // (and so drop the conversation) for what is about to be the same device
    // holding the same range. The sweeper removes it once the grace period says it
    // is really gone; a device that means to leave says so at POST /leave, and gets
    // its layers handed on at once.
  });
});

const PORT = +(process.env.PORT || 8000);

// A raw EADDRINUSE stack says nothing about what to do next, and this is the
// most common way starting the coordinator fails (a previous run still holding
// the port). Name the fix instead.
server.on('error', e => {
  if (e.code === 'EADDRINUSE') {
    console.error(`port ${PORT} is already in use — another flock is running.\n` +
                  `    lsof -nP -iTCP:${PORT} -sTCP:LISTEN     # who has it\n` +
                  `    PORT=${PORT + 1} npm start              # or use another port`);
    process.exit(1);
  }
  console.error(`server error: ${e.message}`);
  process.exit(1);
});

server.listen(PORT, '0.0.0.0', () => {
  // Only the real LAN address is reachable from a phone; bridge and vpn
  // interfaces just add noise to a line someone has to type on a tablet.
  const addrs = localAddresses();
  const lan = addrs.filter(a => /^192\.168\.|^10\.|^172\.(1[6-9]|2\d|3[01])\./.test(a)
                                && !a.endsWith('.0'));
  for (const a of (lan.length ? lan : addrs)) {
    console.log(`  ${SCHEME}://${a}:${PORT}`);
  }
  console.log(`flock listening on port ${PORT} — chat at / , birds join at /flock` +
              `, device check at /check`);
  // No slot count to print any more: layers are divided among whoever joins.
  console.log(`  birds share layers ${CUT}-${N_TOTAL - 1}; ` +
              `however many devices show up, the split follows them`);
  if (!hasCert) {
    console.log('\n  NOTE: serving plain http, so Chrome and Edge will hide WebGPU');
    console.log('  (they require a secure context; Safari does not). To fix:');
    console.log('    npm run cert');
  } else {
    console.log('\n  https with a self-signed cert: each device must trust it once');
    console.log('  (open the URL, accept the warning), then WebGPU works in any browser');
  }
});
