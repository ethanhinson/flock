// Drive bird.html's real logic against a live coordinator, without a browser.
//
// RUN IT WITH DENO, NOT NODE:
//
//   FLOCK_URL=... deno run --unstable-webgpu --allow-all test/e2e/bird_ui.test.mjs
//
// Same idea as chat_ui.test.mjs: extract the page's module script, stub the
// browser surface it touches, and let it actually join the flock and compute
// frames. What is new is that NOTHING about the inference is stubbed any more.
//
// The old version shimmed onnxruntime-node in place of onnxruntime-web and was
// explicit that it tested the page's UI state machine and not the numerics,
// because the two ONNX runtimes take their weights differently and shimming one
// for the other tests the shim. That compromise is gone: the page now runs WGSL
// kernels, Deno provides a real WebGPU device, and the page's own
// gguf-stream + Layer path runs unmodified. So this test streams real Q8_0
// weights from HuggingFace into real GPU buffers and computes real frames --
// the UI state AND the inference, on the same code a phone runs.
//
// What is still stubbed is only the browser: the DOM, localStorage, indexedDB and
// RTCPeerConnection. Those are the parts a headless runtime genuinely lacks.
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import WebSocket from 'ws';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BASE = process.env.FLOCK_URL || 'http://127.0.0.1:8000';
const html = readFileSync(path.join(ROOT, 'web/bird.html'), 'utf8');
const src = /<script type="module">([\s\S]*?)<\/script>/.exec(html)[1];

let fails = 0;
const check = (name, cond, extra = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${extra ? '  ' + extra : ''}`);
  if (!cond) fails++;
};

// --- browser surface -------------------------------------------------------
const nodes = new Map();
const mk = () => {
  const n = {
    className: '', id: '', textContent: '', innerHTML: '', style: {},
    children: [], disabled: false, readyState: 1,
    classList: {
      _s: new Set(),
      add(...c) { c.forEach(x => this._s.add(x)); },
      remove(...c) { c.forEach(x => this._s.delete(x)); },
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    append(...k) { this.children.push(...k); },
    prepend(...k) { this.children.unshift(...k); },
  };
  return n;
};
for (const id of ['sub', 'card', 'layers', 'of', 'count', 'dl', 'dlwhat', 'dlpct',
                  'dlfill', 'backend', 'link', 'cache', 'ms', 'kb', 'next',
                  'problem', 'ptitle', 'pdetail', 'pfix', 'join', 'status', 'binding']) {
  const n = mk(); n.id = id; nodes.set(id, n);
}
// Match the markup: the join button ships disabled and the load path enables it.
// Starting it enabled let the readiness wait fall straight through.
nodes.get('join').disabled = true;
globalThis.document = {
  getElementById: id => nodes.get(id) || null,
  createElement: mk,
  addEventListener() {},
  visibilityState: 'visible',
};
// The page also registers a window-level pagehide listener; stub the global.
globalThis.addEventListener = () => {};
globalThis.window = {isSecureContext: true};
globalThis.localStorage = {
  _m: new Map(),
  getItem(k) { return this._m.has(k) ? this._m.get(k) : null; },
  setItem(k, v) { this._m.set(k, String(v)); },
};
// The page keeps its session in sessionStorage (per tab, dies with the tab).
// A fresh harness is a fresh tab: empty.
globalThis.sessionStorage = {
  _m: new Map(),
  getItem(k) { return this._m.has(k) ? this._m.get(k) : null; },
  setItem(k, v) { this._m.set(k, String(v)); },
  removeItem(k) { this._m.delete(k); },
};
// navigator.gpu is the REAL one -- Deno's. The page requires WebGPU now (the
// layers are compute shaders, so there is no fallback to degrade to), and giving
// it a real device is what lets this test cover the inference rather than a stub.
globalThis.navigator = {
  platform: 'test-bird', userAgent: 'deno test harness', gpu: navigator.gpu,
};
globalThis.indexedDB = undefined;             // exercise the no-cache path
globalThis.location = {host: BASE.replace(/^https?:\/\//, '')};
globalThis.WebSocket = WebSocket;
globalThis.RTCPeerConnection = undefined;     // websocket transport only

// The page imports wire.mjs, gguf-stream.mjs and the kernels by absolute path
// (/js/..., /kernels/...), which only resolve against the coordinator. Point them
// at this file's own copies instead, so the test runs the working tree's code
// rather than whatever the server happens to be serving.
//
// The kernels are imported as '/kernels/layer.ts.js' by the page, because a
// BROWSER cannot execute TypeScript and the server transpiles on request. Deno
// can, so here it resolves straight to the .ts file. Same source either way --
// the transpile is type stripping, not a rewrite.
const wireURL = new URL('../../web/js/wire.mjs', import.meta.url).href;
const ggufURL = new URL('../../web/js/gguf-stream.mjs', import.meta.url).href;
const layerURL = new URL('../../kernels/layer.ts', import.meta.url).href;

const rewritten = src
  .replace("from '/js/wire.mjs'", `from '${wireURL}'`)
  // The REAL probe: it runs against this machine's actual adapter, so the caps this
  // harness sends at /join are the same ones a phone would send.
  .replace("from '/js/probe.mjs'",
           `from '${new URL('../../web/js/probe.mjs', import.meta.url).href}'`)
  .replace("from '/js/gguf-stream.mjs'", `from '${ggufURL}'`)
  .replace("from '/kernels/layer.ts.js'", `from '${layerURL}'`)
  // Same-origin paths -> the coordinator under test.
  .replace(/fetch\('\//g, `fetch('${BASE}/`)
  // The page routes load failures to log()+report()+problem(); also print, so a
  // harness gap is not mistaken for a page bug.
  .replace('})().catch(e => {', '})().catch(e => { console.log(\'  load threw:\', e && e.message || e);');

process.on('unhandledRejection', e => console.log('  unhandled:', e?.stack || e));
// A data: URL rather than a temp file, so nothing is written to disk. Encoded via
// TextEncoder + btoa rather than Buffer: this runs under Deno, where Buffer is not
// a global, and btoa alone would mangle the page's non-ASCII characters (it has
// en-dashes and an ellipsis) because it takes latin-1 code units.
const bytes = new TextEncoder().encode(rewritten);
const b64 = btoa(Array.from(bytes, b => String.fromCharCode(b)).join(''));
const mod = `data:text/javascript;base64,${b64}`;
await import(mod);

// Wait for the page's own load path to finish claiming + building.
for (let i = 0; i < 300 && nodes.get('join').disabled; i++)
  await new Promise(r => setTimeout(r, 200));

const logLines = () => nodes.get('status').children.map(c => c.textContent);
console.log('page log (newest first):');
for (const l of logLines().slice(0, 8)) console.log('    ' + l);
console.log('after load:');
console.log(`  layers   : ${nodes.get('layers').textContent} ${nodes.get('of').textContent}`);
console.log(`  backend  : ${nodes.get('backend').textContent}`);
console.log(`  progress : ${nodes.get('dlwhat').textContent} ${nodes.get('dlpct').textContent}`);
const problem = nodes.get('problem').classList.contains('on');
if (problem) console.log(`  PROBLEM  : ${nodes.get('ptitle').textContent} — ${nodes.get('pdetail').textContent}`);

check('claimed a slot and shows its layer range',
  /^\d+–\d+$/.test(nodes.get('layers').textContent), nodes.get('layers').textContent);
check('shows total layers and its own count',
  /of \d+ · \d+ layer/.test(nodes.get('of').textContent), nodes.get('of').textContent);
check('reports webgpu as its backend (there is no CPU fallback any more)',
  nodes.get('backend').textContent === 'webgpu',
  nodes.get('backend').textContent);
check('download progress reached ready', nodes.get('dlpct').textContent === 'ready',
  nodes.get('dlpct').textContent);
check('no problem banner after a clean load', !problem);
check('join became enabled', !nodes.get('join').disabled);

// The load path is now GGUF-only, so the log has to show it actually streamed
// weights rather than falling back to something. These are the lines that would
// be absent if the page had quietly taken another route.
const allLog = logLines().join('\n');
check('streamed its layers from the GGUF file',
  /gguf: \d+MB in [\d.]+s, \d+ tensors/.test(allLog),
  (allLog.match(/gguf: [^\n]*/) || ['(no gguf line)'])[0]);
check('the log mentions no ONNX anywhere', !/onnx/i.test(allLog),
  (allLog.match(/[^\n]*onnx[^\n]*/i) || ['clean'])[0]);

// --- join, then make the coordinator drive a real turn --------------------
console.log('\njoining and computing a real turn:');
await nodes.get('join').onclick();
for (let i = 0; i < 100; i++) {
  if (['websocket', 'webrtc'].includes(nodes.get('link').textContent)) break;
  await new Promise(r => setTimeout(r, 100));
}
console.log(`  link     : ${nodes.get('link').textContent}`);
console.log(`  forwards : ${nodes.get('next').textContent}`);
check('link reports the transport it actually got',
  ['websocket', 'webrtc'].includes(nodes.get('link').textContent),
  nodes.get('link').textContent);
check('forwards-to names the coordinator or the next range',
  /coordinator|layers \d+-\d+/.test(nodes.get('next').textContent),
  nodes.get('next').textContent);

// Only slot 0 was claimed by this harness; the rest of the flock must be covered
// for /chat to run, so just report if it is not.
const health = await (await fetch(`${BASE}/health`)).json();
if (!health.ok) {
  console.log(`  -- flock not fully covered (missing ${health.missing.join(', ')});` +
              ' run sim_bird.mjs for the other slots to exercise frames');
} else {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: 'hi', max_tokens: 4, reset: true}),
  });
  await res.text();
  await new Promise(r => setTimeout(r, 300));
  console.log(`  frames   : ${nodes.get('count').textContent}`);
  console.log(`  kv cache : ${nodes.get('cache').textContent}`);
  console.log(`  last ms  : ${nodes.get('ms').textContent}`);
  console.log(`  frame in : ${nodes.get('kb').textContent}`);
  check('computed at least one frame', +nodes.get('count').textContent > 0,
    nodes.get('count').textContent);
  check('reports its kv cache in tokens and bytes',
    /\d+ tokens · [\d.]+MB/.test(nodes.get('cache').textContent),
    nodes.get('cache').textContent);
  check('reports last compute time in ms', /ms$/.test(nodes.get('ms').textContent),
    nodes.get('ms').textContent);
  check('reports the inbound frame size as f16',
    /KB f16$/.test(nodes.get('kb').textContent), nodes.get('kb').textContent);
  check('no problem banner after real frames',
    !nodes.get('problem').classList.contains('on'));
}

// ==========================================================================
// A SECOND DEVICE JOINS -> this page must SHOW its new range, not the old one.
//
// This is the case that was broken and that no test covered: the page displayed
// whatever /join told it and never updated, so two devices showed overlapping
// ranges while the coordinator held a correct split. Asserting the DOM rather than
// the server is the whole point -- /status was right the entire time.
// ==========================================================================
{
  const before = nodes.get('layers').textContent;
  console.log(`\na second device joins (this page holds ${before}):`);

  // A real second member: claim a slot and hold a socket, so the coordinator
  // reallocates for two devices rather than treating it as a dead registration.
  const j = await (await fetch(`${BASE}/join`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({label: 'second', caps: {
      maxStorageBufferBindingSize: 1073741824, maxBufferSize: 1073741824, cores: 8}}),
  })).json();
  const ws = new WebSocket(`${BASE.replace('https', 'wss')}/ws`);
  await new Promise(r => { ws.onopen = r; ws.onerror = r; });
  ws.send(JSON.stringify({peer_id: j.peer_id, session: j.session, label: 'second'}));

  // Give the page time to receive its chain message and re-stream.
  for (let i = 0; i < 150 && nodes.get('layers').textContent === before; i++)
    await new Promise(r => setTimeout(r, 200));
  const after = nodes.get('layers').textContent;
  console.log(`  coordinator gave the second device ${j.start}-${j.end}`);
  console.log(`  this page now shows ${after} (was ${before})`);

  check('the page updated its range when a second device joined',
    after !== before, `${before} -> ${after}`);

  // The real symptom: two devices claiming the same layers.
  const [a0, a1] = after.split('\u2013').map(Number);
  const overlap = !(a1 < j.start || a0 > j.end);
  check('this page and the second device do not overlap',
    !overlap, `page ${after} vs other ${j.start}-${j.end}`);

  const status = await (await fetch(`${BASE}/status`)).json();
  const live = (status.birds || []).filter(b => b.alive && b.start != null);
  const held = {};
  for (const b of live) for (let L = b.start; L <= b.end; L++) held[L] = (held[L] || 0) + 1;
  const twice = Object.keys(held).filter(L => held[L] > 1);
  check('the coordinator holds no layer twice', twice.length === 0, twice.join(','));

  check('the page agrees with what the coordinator thinks it holds',
    live.some(b => `${b.start}\u2013${b.end}` === after),
    `page ${after}, server ${live.map(b => `${b.start}-${b.end}`).join(' ')}`);

  try { ws.close(); } catch {}
}

console.log(`\n${fails ? `${fails} CHECKS FAILED` : 'ALL CHECKS PASSED'}`);
process.exit(fails ? 1 : 0);
