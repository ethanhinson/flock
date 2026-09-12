// Does bird.html's GGUF gate open only when it should, and does the page survive
// when the GGUF load fails?
//
// The gate is three conditions ANDed together -- a coordinator that handed out a
// url, a device with WebGPU, and an explicit opt-in -- and the important property
// is not that it works when all three hold. It is that the ONNX path still runs
// when they do not, and STILL runs when the GGUF load blows up, because the ONNX
// session is the reference the WGSL engine is validated against. A bird that lost
// its slot to a failed experiment would be a worse bird.
//
// Self-contained on purpose: it stubs the coordinator rather than needing one, so
// it runs without the Python export step and without a GPU. The real streaming is
// covered by web/js/gguf-stream.test.ts against real hardware and the real file;
// what is under test HERE is only the branch.
//
//   node test/bird_gguf.test.mjs
import {readFileSync, writeFileSync, unlinkSync} from 'fs';
import {createServer} from 'http';
import path from 'path';
import {fileURLToPath, pathToFileURL} from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const html = readFileSync(path.join(ROOT, 'web/bird.html'), 'utf8');
const src = /<script type="module">([\s\S]*?)<\/script>/.exec(html)[1];

let fails = 0, nextShim = 0;
const check = (name, cond, extra = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${extra ? '  ' + extra : ''}`);
  if (!cond) fails++;
};

// A coordinator just complete enough for the load path: /join and the two shard
// files. `gguf` is whatever the scenario asks for.
function stubCoordinator(gguf) {
  const srv = createServer((req, res) => {
    if (req.url === '/join') {
      let body = '';
      req.on('data', c => body += c);
      req.on('end', () => {
        res.setHeader('content-type', 'application/json');
        res.end(JSON.stringify({
          peer_id: 'test', slot: 0, start: 24, end: 27, n_layers: 4,
          hidden: 1024, kv_heads: 8, head_dim: 128, kv_cache: true,
          n_total: 28, model: 'Qwen/Qwen3-0.6B', gguf,
        }));
      });
      return;
    }
    if (req.url === '/diag') { res.end('{"ok":true}'); return; }
    if (req.url.startsWith('/shard/')) {
      const b = Buffer.alloc(1024);
      res.setHeader('content-length', String(b.length));
      res.end(b);
      return;
    }
    res.statusCode = 404; res.end('{}');
  });
  return new Promise(r => srv.listen(0, '127.0.0.1', () => r(srv)));
}

const mk = () => ({
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
});

/**
 * Run the page's load path once under a given scenario and report what the GGUF
 * branch did.
 *
 * `gpu` is a fake navigator.gpu: null for a device without WebGPU, or an object
 * whose requestAdapter behaviour the scenario controls.
 */
async function run({gguf, optIn, gpu, ggufModule}) {
  const srv = await stubCoordinator(gguf);
  const BASE = `http://127.0.0.1:${srv.address().port}`;
  const nodes = new Map();
  for (const id of ['sub', 'card', 'layers', 'of', 'count', 'dl', 'dlwhat', 'dlpct',
                    'dlfill', 'backend', 'link', 'cache', 'ms', 'kb', 'next',
                    'problem', 'ptitle', 'pdetail', 'pfix', 'join', 'status']) {
    const n = mk(); n.id = id; nodes.set(id, n);
  }
  nodes.get('join').disabled = true;

  globalThis.document = {
    getElementById: id => nodes.get(id) || null,
    createElement: mk, addEventListener() {}, visibilityState: 'visible',
  };
  const store = new Map();
  if (optIn) store.set('flock_gguf', '1');
  globalThis.localStorage = {
    getItem: k => store.has(k) ? store.get(k) : null,
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k),
  };
  globalThis.navigator = {platform: 'test', userAgent: 'node harness', gpu};
  globalThis.indexedDB = undefined;
  globalThis.location = {host: BASE.replace(/^https?:\/\//, ''), search: ''};
  globalThis.WebSocket = class { constructor() { this.readyState = 0; } send() {} close() {} };
  globalThis.RTCPeerConnection = undefined;
  globalThis.URLSearchParams = URLSearchParams;
  // The page assigns window.flockWeights. `window` is a fresh object per scenario
  // rather than globalThis, or a success in one scenario is still visible in the
  // next and every later check reads the wrong run.
  globalThis.window = {};

  // The GGUF module is stubbed rather than imported for real: this test is about
  // the branch, and the module's own behaviour is proven on real hardware
  // elsewhere. A scenario can make it throw, which is the case that matters most.
  //
  // The filename is unique per scenario, and that is load-bearing rather than
  // tidy: Node's ESM loader caches by resolved path, so reusing one path made
  // every scenario import the FIRST scenario's shim -- the throwing one never
  // threw, and two checks passed for the wrong reason.
  const tag = `${process.pid}-${nextShim++}`;
  const ggufPath = path.join(ROOT, `node/test/.gguf-shim-${tag}.mjs`);
  writeFileSync(ggufPath, ggufModule);
  const ortPath = path.join(ROOT, `node/test/.ort-gguf-shim-${tag}.mjs`);
  writeFileSync(ortPath, `
class Tensor { constructor(t, d, s) { this.type = t; this.data = d; this.dims = s; } }
const InferenceSession = {async create() { return {async run() { return {}; }}; }};
export {Tensor, InferenceSession};
export const env = {wasm: {}};
`);

  const rewritten = src
    .replace(/from 'https:\/\/cdn\.jsdelivr\.net\/[^']*'/, `from '${pathToFileURL(ortPath).href}'`)
    .replace("from '/js/wire.mjs'", `from '${new URL('../../web/js/wire.mjs', import.meta.url).href}'`)
    .replace("import('/js/gguf-stream.mjs')", `import('${pathToFileURL(ggufPath).href}')`)
    .replace(/fetch\('\//g, `fetch('${BASE}/`)
    .replace(/grab\(`\//g, `grab(\`${BASE}/`)
    // Only the load path is under test; the transport would hang on a fake socket.
    .replace(/^\s*connect\(\);\s*$/m, '');

  // A data: URL with identical bytes is cached by the ESM loader, so two
  // scenarios whose rewritten source happens to match would silently run the same
  // module twice -- reporting the first scenario's outcome for both. The salt is
  // what makes each run a fresh module.
  const salted = `// run ${Math.random()}\n` + rewritten;
  const mod = `data:text/javascript;base64,${Buffer.from(salted).toString('base64')}`;
  await import(mod);
  for (let i = 0; i < 100 && nodes.get('join').disabled; i++) {
    await new Promise(r => setTimeout(r, 20));
  }

  const log = nodes.get('status').children.map(c => c.textContent);
  srv.close();
  try { unlinkSync(ggufPath); unlinkSync(ortPath); } catch {}
  return {
    log, joined: !nodes.get('join').disabled,
    weights: globalThis.window.flockWeights || null,
    optedIn: store.get('flock_gguf') || null,
  };
}

// A GGUF module that succeeds, reporting the numbers the real one reports.
const OK_MODULE = `
export async function loadBirdWeights(device, url, first, last, opts) {
  opts.onNote?.('directory: 310 tensors, 28 layers');
  opts.onProgress?.({tensor: 'blk.24.attn_q.weight', bytesDone: 66900000, bytesTotal: 66900000});
  return {model: {url}, layers: {24: {}}, stats: {
    bytesTotal: 66900000, tensors: 44, requests: 1, fromCache: false, seconds: 3.3}};
}
`;
const THROWS_MODULE = `
export async function loadBirdWeights() { throw new Error('simulated GPU OOM'); }
`;

const GGUF_URL = 'https://example.invalid/model.gguf';
const fakeGPU = {
  requestAdapter: async () => ({
    limits: {maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 28},
    info: {vendor: 'test'},
    requestDevice: async () => ({}),
  }),
};

console.log('all three conditions hold -> the GGUF path runs:');
let r = await run({gguf: GGUF_URL, optIn: true, gpu: fakeGPU, ggufModule: OK_MODULE});
check('reached the ONNX session anyway (the reference path still runs)', r.joined);
check('logged the GGUF numbers', r.log.some(l => /gguf: 67MB .*44 tensors, 1 range request/.test(l)),
  r.log.find(l => l.startsWith('gguf:')) || '(no gguf line)');
check('handed the buffers to the engine via window.flockWeights',
  !!r.weights && r.weights.stats.tensors === 44);

console.log('\nno opt-in -> the GGUF path is skipped entirely:');
r = await run({gguf: GGUF_URL, optIn: false, gpu: fakeGPU, ggufModule: THROWS_MODULE});
check('the ONNX path still ran', r.joined);
check('nothing GGUF was attempted', !r.log.some(l => /^gguf/.test(l)),
  r.log.filter(l => /gguf/.test(l)).join(' | ') || '(none, as expected)');
check('no weights handed over', r.weights === null);

console.log('\nno GGUF url from the coordinator -> skipped even with the opt-in:');
r = await run({gguf: '', optIn: true, gpu: fakeGPU, ggufModule: THROWS_MODULE});
check('the ONNX path still ran', r.joined);
check('nothing GGUF was attempted', !r.log.some(l => /^gguf/.test(l)));

console.log('\nno WebGPU on this device -> skipped even with url and opt-in:');
r = await run({gguf: GGUF_URL, optIn: true, gpu: null, ggufModule: THROWS_MODULE});
check('the ONNX path still ran', r.joined);
check('nothing GGUF was attempted', !r.log.some(l => /^gguf/.test(l)));

// The one that matters most: an experimental path must not be able to cost a bird
// its slot, because the ONNX session is what actually answers frames today.
console.log('\nthe GGUF load throws -> the bird still joins on the ONNX path:');
r = await run({gguf: GGUF_URL, optIn: true, gpu: fakeGPU, ggufModule: THROWS_MODULE});
check('the ONNX path still ran despite the GGUF failure', r.joined);
check('said so in plain language rather than dying',
  r.log.some(l => /gguf load failed, continuing on the onnx path/.test(l)),
  r.log.find(l => /gguf/.test(l)) || '(no gguf line)');
check('no half-built weights left on window', r.weights === null);
check('no user-facing problem banner for an opt-in experiment',
  !r.log.some(l => /could not load my layers/.test(l)));

console.log(`\n${fails ? `${fails} failed` : 'all checks passed'}`);
process.exit(fails ? 1 : 0);
