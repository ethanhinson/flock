// Tests the GGUF parser against the REAL Qwen3-0.6B file on HuggingFace.
// Network-dependent by design: the point is that range-fetching works against
// a real CDN, not that a fixture parses.
import {loadDirectory, httpRange, layerTensors, byteRanges, totalBytes, GGML}
  from '../../web/js/gguf.mjs';

const URL = 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf';
let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};

console.log('fetching directory over HTTP range requests…');
const t0 = Date.now();
const dir = await loadDirectory(httpRange(URL));
console.log(`got it in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

ok('parses a real GGUF v3 file', dir.version === 3);
ok('finds all 310 tensors', dir.tensors.size === 310, `got ${dir.tensors.size}`);
ok('reads architecture metadata', dir.meta['general.architecture'] === 'qwen3',
   dir.meta['general.architecture']);
ok('reads block count', dir.meta['qwen3.block_count'] === 28,
   String(dir.meta['qwen3.block_count']));
ok('directory is a small prefix', dir.directoryBytes < 8e6,
   `${(dir.directoryBytes / 1e6).toFixed(1)}MB of a 639MB file`);
ok('data starts after the directory', dir.dataStart >= dir.directoryBytes);

// --- one layer -----------------------------------------------------------
const l24 = layerTensors(dir, 24);
ok('layer 24 has 11 tensors', l24.length === 11, `got ${l24.length}`);
ok('layer tensors are Q8_0 or F32',
   l24.every(t => GGML[t.type].name === 'Q8_0' || GGML[t.type].name === 'F32'));

const r24 = byteRanges(dir, [24]);
ok('layer 24 is one contiguous range', r24.length === 1,
   `${(totalBytes(r24) / 1e6).toFixed(1)}MB`);

// --- a bird's whole slice -------------------------------------------------
const birdLayers = [24, 25, 26, 27];
const rb = byteRanges(dir, birdLayers);
const mb = totalBytes(rb) / 1e6;
ok('a 4-layer bird fetches one merged range', rb.length === 1, `${mb.toFixed(1)}MB`);
ok('that is far less than the 252MB ONNX export', mb < 80,
   `${(252 / mb).toFixed(1)}x smaller`);

// --- sizes must agree with the file ---------------------------------------
const last = [...dir.tensors.values()].sort((a, b) => a.offset - b.offset).pop();
const computedEnd = dir.dataStart + last.offset + last.bytes;
ok('computed file size matches Content-Length', Math.abs(computedEnd - 639446688) < 4096,
   `computed ${computedEnd}, actual 639446688`);

// --- actually fetch a slice and check we got the bytes we asked for -------
const one = r24[0];
const buf = await httpRange(URL)(one.start, one.end - 1);
ok('range fetch returns exactly the requested bytes',
   buf.byteLength === one.end - one.start,
   `${buf.byteLength} bytes`);

// A Q8_0 block is 32 int8 + an f16 scale. Scales should be small and finite --
// a cheap sanity check that we landed on real weight data, not padding.
const dv = new DataView(buf);
let sane = 0;
for (let i = 0; i < 64; i++) {
  const h = dv.getUint16(i * 34, true);
  const exp = (h >>> 10) & 0x1f;
  if (exp !== 0x1f) sane++;                 // not inf/nan
}
ok('fetched bytes look like Q8_0 blocks', sane === 64, `${sane}/64 finite scales`);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
