// Drive bird.html's real logic against a live coordinator, without a browser.
//
// Same idea as chat_ui.test.mjs: extract the page's module script, stub the
// browser surface it touches, and let it actually join the flock and compute
// frames -- with onnxruntime-node standing in for onnxruntime-web, since the
// page's inference calls are the same shape in both.
//
// This is what proves the page's UI state (my layers, backend, kv cache size,
// link type, frames, download progress, plain-language errors) is driven by real
// data rather than only parsing.
//
//   node src/server.js &   node test/bird_ui.test.mjs
import {readFileSync, writeFileSync, unlinkSync} from 'fs';
import path from 'path';
import {fileURLToPath, pathToFileURL} from 'url';
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
                  'problem', 'ptitle', 'pdetail', 'pfix', 'join', 'status']) {
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
globalThis.localStorage = {
  _m: new Map(),
  getItem(k) { return this._m.has(k) ? this._m.get(k) : null; },
  setItem(k, v) { this._m.set(k, String(v)); },
};
globalThis.navigator = {platform: 'test-bird', userAgent: 'node test harness'};
globalThis.indexedDB = undefined;             // exercise the no-cache path
globalThis.location = {host: BASE.replace(/^https?:\/\//, '')};
globalThis.WebSocket = WebSocket;
globalThis.RTCPeerConnection = undefined;     // websocket transport only

// The page imports its inference runtime from a CDN and wire.mjs from /js.
//
// Inference itself is stubbed rather than shimmed onto onnxruntime-node: the two
// runtimes take their weights differently (a path vs a graph buffer plus an
// external-data view), so shimming tests the shim. What is under test here is
// the page's UI state machine -- which layers it claims, what it reports as its
// backend, download progress, the shape of its K/V bookkeeping, the link label,
// and whether failures arrive as plain language. A stub that returns correctly
// shaped tensors exercises all of that, and is honest about what it does not
// cover: the numerics, which node/test/gguf.test.mjs and the export step own.
const wireURL = new URL('../../web/js/wire.mjs', import.meta.url).href;
const shimPath = path.join(ROOT, 'node/test/.ort-shim.mjs');
writeFileSync(shimPath, `
class Tensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}
// Echoes the hidden state through and grows K/V by seq, the same contract the
// page depends on: out.output.data to forward, out.new_k0.dims[2] for cache size.
const InferenceSession = {
  async create() {
    let held = 0;
    return {
      async run(feed) {
        const [, seq, hidden] = feed.hidden.dims;
        held += seq;
        const out = {output: new Tensor('float32', feed.hidden.data, [1, seq, hidden])};
        for (let i = 0; ; i++) {
          if (!(\`past_k\${i}\` in feed)) break;
          out[\`new_k\${i}\`] = new Tensor('float32', new Float32Array(0), [1, 8, held, 128]);
          out[\`new_v\${i}\`] = new Tensor('float32', new Float32Array(0), [1, 8, held, 128]);
        }
        return out;
      },
    };
  },
};
export {Tensor, InferenceSession};
export const env = {wasm: {}};
export default {Tensor, InferenceSession, env};
`);
const ortShim = pathToFileURL(shimPath).href;

const rewritten = src
  .replace(/from 'https:\/\/cdn\.jsdelivr\.net\/[^']*'/, `from '${ortShim}'`)
  .replace("from '/js/wire.mjs'", `from '${wireURL}'`)
  // Same-origin paths -> the coordinator under test. The shard URLs are built at
  // the grab() call sites rather than at a fetch(), so rewrite those too.
  .replace(/fetch\('\//g, `fetch('${BASE}/`)
  .replace(/grab\(`\//g, `grab(\`${BASE}/`)
  // The page routes load failures to log()+report()+problem(); also print, so a
  // harness gap is not mistaken for a page bug.
  .replace('})().catch(e => {', '})().catch(e => { console.log(\'  load threw:\', e && e.message || e);');

process.on('unhandledRejection', e => console.log('  unhandled:', e?.stack || e));
const mod = `data:text/javascript;base64,${Buffer.from(rewritten).toString('base64')}`;
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
check('reports a backend', ['webgpu', 'wasm'].includes(nodes.get('backend').textContent),
  nodes.get('backend').textContent);
check('download progress reached ready', nodes.get('dlpct').textContent === 'ready',
  nodes.get('dlpct').textContent);
check('no problem banner after a clean load', !problem);
check('join became enabled', !nodes.get('join').disabled);

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

try { unlinkSync(shimPath); } catch {}
console.log(`\n${fails ? `${fails} CHECKS FAILED` : 'ALL CHECKS PASSED'}`);
process.exit(fails ? 1 : 0);
