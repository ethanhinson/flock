// Tests range-planning against the REAL Qwen3-0.6B file on HuggingFace.
// Network-dependent by design: the claim under test is that range requests
// work against a real CDN, which a fixture cannot show.
import {readModel, layerTensors, byteRanges, totalBytes, fetchRange, splitLayers}
  from '../src/gguf.mjs';

const URL = 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf';
const FILE_BYTES = 639446688;
let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};

console.log('reading directory over HTTP range requests…');
const t0 = Date.now();
const model = await readModel(URL);
console.log(`got it in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

ok('reads architecture', model.arch === 'qwen3', model.arch);
ok('reads layer count', model.nLayers === 28, String(model.nLayers));
ok('finds all 310 tensors', model.tensors.length === 310, String(model.tensors.length));

// --- one layer -----------------------------------------------------------
const l24 = layerTensors(model, 24);
ok('layer 24 has 11 tensors', l24.length === 11, String(l24.length));
const r24 = byteRanges(model, [24]);
ok('layer 24 is one contiguous range', r24.length === 1,
   `${(totalBytes(r24) / 1e6).toFixed(1)}MB`);

// --- a bird's whole slice -------------------------------------------------
const rb = byteRanges(model, [24, 25, 26, 27]);
const mb = totalBytes(rb) / 1e6;
ok('a 4-layer bird fetches ONE merged range', rb.length === 1, `${mb.toFixed(1)}MB`);
ok('comparable to the int8 ONNX export (63MB)', mb < 80, `${mb.toFixed(1)}MB`);

// --- sizes must agree with the file ---------------------------------------
// The strongest check available: if any offset or block size were wrong, the
// computed end of the last tensor would not land on the real file size.
const last = [...model.tensors].sort((a, b) => a.offset - b.offset).pop();
const computedEnd = model.dataStart + last.offset + last.bytes;
ok('computed file size matches Content-Length',
   Math.abs(computedEnd - FILE_BYTES) < 4096,
   `computed ${computedEnd}, actual ${FILE_BYTES}`);

// --- splitting ------------------------------------------------------------
ok('splits 4 layers across 2 birds evenly',
   JSON.stringify(splitLayers(24, 27, 2)) === '[[24,25],[26,27]]');
ok('spreads the remainder over the first birds',
   JSON.stringify(splitLayers(24, 27, 3)) === '[[24,25],[26,26],[27,27]]');

// --- actually fetch a slice ----------------------------------------------
const buf = await fetchRange(URL, r24[0]);
ok('range fetch returns exactly the requested bytes',
   buf.byteLength === r24[0].end - r24[0].start, `${buf.byteLength} bytes`);

// A Q8_0 block is 32 int8 plus an f16 scale. Finite scales are a cheap check
// that we landed on real weight data rather than padding or the wrong offset.
const dv = new DataView(buf);
let sane = 0;
for (let i = 0; i < 64; i++) if (((dv.getUint16(i * 34, true) >>> 10) & 0x1f) !== 0x1f) sane++;
ok('fetched bytes look like Q8_0 blocks', sane === 64, `${sane}/64 finite scales`);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
