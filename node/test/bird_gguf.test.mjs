// bird.html's load path, now that GGUF + WGSL is the ONLY path.
//
// THIS TEST'S PREMISE INVERTED, and the inversion is the point. It used to assert
// that a GGUF failure was survivable BECAUSE the ONNX session was the real path:
// "an experimental path must not cost a bird its slot". There is no ONNX session
// any more and no opt-in gate, so the contract is the opposite one. A bird that
// cannot load its layers must NOT end up enabled and claiming a slot it cannot
// serve -- the coordinator would wait on it and the whole flock would stall on a
// device that can do nothing. So every failure here has to end in a plain-language
// problem banner and a join button that stays disabled.
//
// Self-contained on purpose: it stubs the coordinator rather than needing one, so
// it runs in CI without a GPU and without the network. The real streaming is
// covered by web/js/gguf-stream.test.ts against real hardware and the real file,
// and the real layers by node/test/bird_ui.test.mjs against a live coordinator;
// what is under test HERE is only the branching and the failure reporting.
//
//   node test/bird_gguf.test.mjs
import {readFileSync, writeFileSync, unlinkSync} from 'node:fs';
import {createServer} from 'node:http';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const html = readFileSync(path.join(ROOT, 'web/bird.html'), 'utf8');
const src = /<script type="module">([\s\S]*?)<\/script>/.exec(html)[1];

let fails = 0, nextShim = 0;
const check = (name, cond, extra = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${extra ? '  ' + extra : ''}`);
  if (!cond) fails++;
};

// A coordinator just complete enough for the load path: /join and /diag. There are
// no shard routes to stub any more -- they are gone from the server too.
/**
 * @param gguf    the weights url to hand out
 * @param join    what /join answers. A function gets the attempt number, so a
 *                scenario can answer `wait` a few times before assigning layers --
 *                which is what the coordinator does when a token is in flight, and is
 *                a branch the page has to survive rather than treat as a refusal.
 *                Also records what the page SENT, so the caps payload is checkable.
 */
function stubCoordinator(gguf, join = null) {
  const seen = {joins: [], attempts: 0};
  const srv = createServer((req, res) => {
    if (req.url === '/join') {
      let body = '';
      req.on('data', c => body += c);
      req.on('end', () => {
        try { seen.joins.push(JSON.parse(body)); } catch { seen.joins.push(null); }
        const n = seen.attempts++;
        res.setHeader('content-type', 'application/json');
        res.end(JSON.stringify(join ? join(n) : {
          peer_id: 'test', slot: 0, start: 24, end: 27, n_layers: 4,
          hidden: 1024, kv_heads: 8, head_dim: 128, kv_cache: true,
          n_total: 28, model: 'Qwen/Qwen3-0.6B', gguf,
        }));
      });
      return;
    }
    if (req.url === '/diag') { res.end('{"ok":true}'); return; }
    res.statusCode = 404; res.end('{}');
  });
  return new Promise(r => srv.listen(0, '127.0.0.1', () => { srv.seen = seen; r(srv); }));
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
 * Run the page's load path once under a given scenario and report what happened.
 *
 * `gpu` is a fake navigator.gpu: null for a device without WebGPU, or an object
 * whose requestAdapter behaviour the scenario controls. `layerModule` stands in
 * for kernels/layer.ts, so a scenario can make building the layers throw without
 * needing a real device.
 */
async function run({gguf, gpu, ggufModule, layerModule, join = null}) {
  const srv = await stubCoordinator(gguf, join);
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

  // The GGUF loader and the kernels are stubbed rather than imported for real:
  // this test is about the branching, and both modules' real behaviour is proven
  // on real hardware elsewhere. A scenario can make either one throw, which is
  // the case that matters most.
  //
  // The filenames are unique per scenario, and that is load-bearing rather than
  // tidy: Node's ESM loader caches by resolved path, so reusing one path made
  // every scenario import the FIRST scenario's shim -- the throwing one never
  // threw, and two checks passed for the wrong reason.
  const tag = `${process.pid}-${nextShim++}`;
  const ggufPath = path.join(ROOT, `node/test/.gguf-shim-${tag}.mjs`);
  writeFileSync(ggufPath, ggufModule);
  const layerPath = path.join(ROOT, `node/test/.layer-shim-${tag}.mjs`);
  writeFileSync(layerPath, layerModule);

  const rewritten = src
    .replace("from '/js/wire.mjs'", `from '${new URL('../../web/js/wire.mjs', import.meta.url).href}'`)
    // The REAL probe, not a stub: it takes navigator.gpu as an argument, so a
    // scenario's fake adapter drives it exactly as a real one would. That keeps this
    // test honest about the caps the page actually sends at /join.
    .replace("from '/js/probe.mjs'", `from '${new URL('../../web/js/probe.mjs', import.meta.url).href}'`)
    .replace("from '/js/gguf-stream.mjs'", `from '${pathToFileURL(ggufPath).href}'`)
    .replace("from '/kernels/layer.ts.js'", `from '${pathToFileURL(layerPath).href}'`)
    .replace(/fetch\('\//g, `fetch('${BASE}/`)
    // Only the load path is under test; the transport would hang on a fake socket.
    .replace(/^\s*connect\(\);\s*$/m, '');

  // A data: URL with identical bytes is cached by the ESM loader, so two
  // scenarios whose rewritten source happens to match would silently run the same
  // module twice -- reporting the first scenario's outcome for both. The salt is
  // what makes each run a fresh module.
  const salted = `// run ${Math.random()}\n` + rewritten;
  const mod = `data:text/javascript;base64,${Buffer.from(salted).toString('base64')}`;
  await import(mod);
  // Wait for the load path to settle either way: enabled, or a problem banner.
  for (let i = 0; i < 100; i++) {
    if (!nodes.get('join').disabled) break;
    if (nodes.get('problem').classList.contains('on')) break;
    await new Promise(r => setTimeout(r, 20));
  }

  const log = nodes.get('status').children.map(c => c.textContent);
  const seen = srv.seen;
  srv.close();
  try { unlinkSync(ggufPath); unlinkSync(layerPath); } catch {}
  return {
    log, joined: !nodes.get('join').disabled,
    problem: nodes.get('problem').classList.contains('on'),
    title: nodes.get('ptitle').textContent,
    detail: nodes.get('pdetail').textContent,
    backend: nodes.get('backend').textContent,
    built: globalThis.__built || 0,
    joins: seen.joins, attempts: seen.attempts,
  };
}

// A GGUF loader that succeeds, reporting the numbers the real one reports. Layer
// keys carry the `.weight` suffix, as both the real loader and real_weights.ts do.
const OK_GGUF = `
export async function loadBirdWeights(device, url, first, last, opts) {
  opts.onNote?.('directory: 310 tensors, 28 layers');
  opts.onProgress?.({tensor: 'blk.24.attn_q.weight', bytesDone: 66900000, bytesTotal: 66900000});
  const layers = {};
  for (let l = first; l <= last; l++) layers[l] = {'attn_q.weight': {rows: 2048, cols: 1024}};
  return {model: {url}, layers, stats: {
    bytesTotal: 66900000, tensors: 44, requests: 1, fromCache: false, seconds: 3.3}};
}
`;
const THROWS_GGUF = `
export async function loadBirdWeights() { throw new Error('simulated GPU OOM'); }
`;
// A loader that returns FEWER layers than the coordinator assigned. This is the
// quiet failure worth a test: the page would otherwise build 2 layers, report
// itself ready, and compute a wrong hidden state for every frame forever.
const SHORT_GGUF = `
export async function loadBirdWeights(device, url, first, last, opts) {
  return {model: {url}, layers: {[first]: {'attn_q.weight': {rows: 2048, cols: 1024}}},
          stats: {bytesTotal: 1, tensors: 11, requests: 1, fromCache: false, seconds: 0.1}};
}
`;

const OK_LAYER = `
export const QWEN3_06B = {hidden: 1024, nHeads: 16, nKvHeads: 8, headDim: 128,
                          ffn: 3072, eps: 1e-6, ropeBase: 1e6, maxKeys: 2048,
                          maxPrefill: 512, ropePairing: 'neox'};
export class Layer {
  static async fromBuffers() { globalThis.__built = (globalThis.__built || 0) + 1; return new Layer(); }
  reset() {}
}
`;
const THROWS_LAYER = `
export const QWEN3_06B = {hidden: 1024, nHeads: 16, nKvHeads: 8, headDim: 128,
                          ffn: 3072, eps: 1e-6, ropeBase: 1e6, maxKeys: 2048,
                          maxPrefill: 512, ropePairing: 'neox'};
export class Layer {
  static async fromBuffers() { throw new Error('layer is missing 1 tensor(s): attn_q.weight'); }
  reset() {}
}
`;

const GGUF_URL = 'https://example.invalid/model.gguf';
// A fake adapter, complete enough for the REAL probe in /js/probe.mjs to run its
// compute pass against -- the page probes this device before it joins, and what the
// probe reports is what the allocator plans against, so stubbing the probe away would
// leave the most important number in the whole flow untested here.
globalThis.GPUBufferUsage = {STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8};
globalThis.GPUMapMode = {READ: 1};
const fakeDevice = () => ({
  createShaderModule: () => ({}),
  createComputePipeline: () => ({getBindGroupLayout: () => ({})}),
  createBuffer: ({size}) => ({
    destroy() {}, unmap() {}, mapAsync: async () => {},
    // The probe checks that element 63 reads back 63*63; anything else means the
    // device advertised limits it cannot honour.
    getMappedRange: () => { const a = new Uint32Array(size / 4); a[63] = 63 * 63;
                            return a.buffer; },
  }),
  createBindGroup: () => ({}),
  createCommandEncoder: () => ({
    beginComputePass: () => ({setPipeline() {}, setBindGroup() {},
                              dispatchWorkgroups() {}, end() {}}),
    copyBufferToBuffer() {}, finish: () => ({}),
  }),
  queue: {submit() {}},
  destroy() {},
});
const fakeGPU = {
  requestAdapter: async () => ({
    limits: {maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 28,
             maxComputeWorkgroupStorageSize: 16384,
             maxComputeInvocationsPerWorkgroup: 256},
    info: {vendor: 'test'},
    requestDevice: async () => fakeDevice(),
  }),
};

console.log('the happy path -> streams, builds every layer, and joins:');
globalThis.__built = 0;
let r = await run({gguf: GGUF_URL, gpu: fakeGPU, ggufModule: OK_GGUF, layerModule: OK_LAYER});
check('the bird became ready', r.joined);
check('no problem banner', !r.problem, r.title);
check('logged the GGUF numbers',
  r.log.some(l => /gguf: 67MB .*44 tensors, 1 range request/.test(l)),
  r.log.find(l => l.startsWith('gguf:')) || '(no gguf line)');
check('built one Layer per assigned layer (24-27 is four)', r.built === 4,
  `built ${r.built}`);
check('reports webgpu as the backend', r.backend === 'webgpu', r.backend);
check('mentions no ONNX anywhere in the log', !r.log.some(l => /onnx/i.test(l)),
  r.log.find(l => /onnx/i.test(l)) || 'clean');

// The inverted contract. A bird that cannot load must not claim a slot: the
// coordinator would wait on it and the flock would stall on a useless device.
console.log('\nno WebGPU -> refuses to join, and says why in plain language:');
globalThis.__built = 0;
r = await run({gguf: GGUF_URL, gpu: null, ggufModule: OK_GGUF, layerModule: OK_LAYER});
check('did NOT become ready', !r.joined);
check('showed a problem banner', r.problem);
check('names WebGPU as the missing thing', /WebGPU/i.test(r.title + r.detail),
  r.title);
check('built nothing', r.built === 0, `built ${r.built}`);

console.log('\nno GGUF url from the coordinator -> refuses to join:');
globalThis.__built = 0;
r = await run({gguf: '', gpu: fakeGPU, ggufModule: OK_GGUF, layerModule: OK_LAYER});
check('did NOT become ready', !r.joined);
check('showed a problem banner', r.problem);
check('says the weights location is missing', /weights are|GGUF url/i.test(r.title + r.detail),
  r.title);

console.log('\nthe GGUF load throws -> refuses to join, reports the reason:');
globalThis.__built = 0;
r = await run({gguf: GGUF_URL, gpu: fakeGPU, ggufModule: THROWS_GGUF, layerModule: OK_LAYER});
check('did NOT become ready', !r.joined);
check('showed a problem banner', r.problem);
check('the underlying error survives to the UI', /simulated GPU OOM/.test(r.detail),
  r.detail.slice(0, 80));
check('built nothing', r.built === 0, `built ${r.built}`);

console.log('\nbuilding a layer throws -> refuses to join, names the tensor:');
globalThis.__built = 0;
r = await run({gguf: GGUF_URL, gpu: fakeGPU, ggufModule: OK_GGUF, layerModule: THROWS_LAYER});
check('did NOT become ready', !r.joined);
check('showed a problem banner', r.problem);
check('the missing tensor reaches the UI rather than a slot number',
  /attn_q\.weight/.test(r.detail), r.detail.slice(0, 90));

// The quiet one: fewer layers than assigned. Without the check in the page this
// builds a short chain, reports ready, and computes a wrong hidden state forever.
console.log('\nthe loader returns fewer layers than assigned -> refuses to join:');
globalThis.__built = 0;
r = await run({gguf: GGUF_URL, gpu: fakeGPU, ggufModule: SHORT_GGUF, layerModule: OK_LAYER});
check('did NOT become ready on a short load', !r.joined);
check('showed a problem banner', r.problem);
check('names the layer it did not get', /layer 2[567]/.test(r.detail),
  r.detail.slice(0, 90));

// --- dynamic membership -----------------------------------------------------
// These four are the behaviours that only exist because FLOCK_BIRDS is gone.

console.log('\nit reports its MEASURED limits at /join, not a guess:');
globalThis.__built = 0;
r = await run({gguf: GGUF_URL, gpu: fakeGPU, ggufModule: OK_GGUF, layerModule: OK_LAYER});
{
  const caps = r.joins[0]?.caps;
  check('the join carried a caps object', !!caps, JSON.stringify(caps));
  check('  with the adapter\'s real maxStorageBufferBindingSize',
    caps.maxStorageBufferBindingSize === (1 << 28), String(caps.maxStorageBufferBindingSize));
  check('  and maxBufferSize', caps.maxBufferSize === (1 << 30));
  // The one the allocator cannot work around, so the one that must not be guessed.
  check('  and a real compute pass result, not just a limits table',
    caps.computeOk === true, String(caps.computeOk));
  check('the page logged the limits where a human can see them',
    r.log.some(l => /gpu limits: buffer \d+MB, binding \d+MB/.test(l)),
    r.log.find(l => l.startsWith('gpu limits')) || '(none)');
}

console.log('\n/join answering "wait" (a token is in flight) -> it retries, not fails:');
globalThis.__built = 0;
r = await run({gguf: GGUF_URL, gpu: fakeGPU, ggufModule: OK_GGUF, layerModule: OK_LAYER,
  // Twice "wait", then the real assignment. This is what the coordinator does when a
  // device joins mid-generation: the running token finishes against the old topology
  // and layers are assigned at the next boundary.
  join: n => n < 2
    ? {peer_id: 'test', wait: true, retry_ms: 10, gguf: GGUF_URL,
       reason: 'a token is in flight; layers are assigned at the next token boundary'}
    : {peer_id: 'test', slot: 0, start: 24, end: 27, n_layers: 4, hidden: 1024,
       kv_heads: 8, head_dim: 128, kv_cache: true, n_total: 28,
       model: 'Qwen/Qwen3-0.6B', gguf: GGUF_URL}});
check('it kept asking rather than giving up', r.attempts === 3, `${r.attempts} attempts`);
check('and ended up ready with its layers built', r.joined && r.built === 4,
  `built ${r.built}`);
check('no problem banner: waiting is not a failure', !r.problem, r.title);
check('it said what it was waiting for', r.log.some(l => /token is in flight/.test(l)),
  r.log.find(l => /flight/.test(l)) || '(nothing logged)');

console.log('\na flock that cannot hold the model -> named before anything downloads:');
globalThis.__built = 0;
r = await run({gguf: GGUF_URL, gpu: fakeGPU, ggufModule: OK_GGUF, layerModule: OK_LAYER,
  join: () => ({peer_id: 'test',
    error: 'no device can hold layer 40: its largest tensor output.weight is ' +
           '638.0MB, and the roomiest device in the flock (iPhone) allows 134.2MB ' +
           'per tensor.',
    detail: {kind: 'tensor', layer: 40, tensor: 'output.weight'}})});
check('did NOT become ready', !r.joined);
check('showed a problem banner', r.problem);
check('the tensor and both numbers reach the UI',
  /output\.weight/.test(r.detail) && /638\.0MB/.test(r.detail) &&
  /134\.2MB/.test(r.detail), r.detail.slice(0, 100));
check('and it is NOT reported as "flock full" any more',
  !/full/i.test(r.title + r.detail), r.title);
check('nothing was downloaded and nothing was built', r.built === 0, `built ${r.built}`);

console.log(`\n${fails ? `${fails} failed` : 'all checks passed'}`);
process.exit(fails ? 1 : 0);
