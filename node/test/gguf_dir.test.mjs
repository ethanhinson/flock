// Pin the browser-side GGUF directory parser against the Node one.
//
// There are two parsers on purpose: node/src/gguf.mjs uses @huggingface/gguf,
// which is a bare npm specifier and so cannot be served to a phone, and
// web/js/gguf-dir.mjs parses the header itself. Two implementations of the same
// format is a standing invitation to drift -- a block-size table updated in one
// place, an offset convention changed in the other -- and the failure mode is not
// an error: a wrong offset fetches bytes that decode as valid-looking Q8_0 and
// produce plausible garbage.
//
// So this test does not check the browser parser against a fixture. It checks it
// against the OTHER parser, tensor by tensor, over the real file. If they ever
// disagree about a single offset, this is where it surfaces.
//
// Needs no GPU, which is why it lives in `npm test` rather than in the Deno suite.
import {readModel} from '../src/gguf.mjs';
import {readDirectory, layerTensors, byteRanges, totalBytes}
  from '../../web/js/gguf-dir.mjs';

const URL = 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf';
let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};

console.log('reading the same header with both parsers…');
const [hf, own] = await Promise.all([readModel(URL), readDirectory(URL)]);

ok('same architecture', hf.arch === own.arch, own.arch);
ok('same layer count', hf.nLayers === own.nLayers, String(own.nLayers));
ok('same tensor count', hf.tensors.length === own.tensors.length,
   `${own.tensors.length} tensors`);

// dataStart is the field the whole path hangs on: every tensor offset is relative
// to it, so a disagreement here is silently wrong for every tensor at once.
ok('same tensor-data offset', hf.dataStart === own.dataStart, String(own.dataStart));

// Every tensor, not a sample. The point of this test is to catch ONE offset being
// wrong, and a sample is exactly what would miss it.
const byName = new Map(own.tensors.map(t => [t.name, t]));
let mismatch = null;
for (const a of hf.tensors) {
  const b = byName.get(a.name);
  if (!b) { mismatch = `${a.name} missing from the browser parser`; break; }
  if (Number(b.offset) !== a.offset) {
    mismatch = `${a.name} offset ${b.offset} vs ${a.offset}`; break;
  }
  if (b.bytes !== a.bytes) {
    mismatch = `${a.name} bytes ${b.bytes} vs ${a.bytes}`; break;
  }
  if (b.dtype !== a.dtype) {
    mismatch = `${a.name} dtype ${b.dtype} vs ${a.dtype}`; break;
  }
  if (b.shape.map(Number).join(',') !== a.shape.join(',')) {
    mismatch = `${a.name} shape ${b.shape} vs ${a.shape}`; break;
  }
}
ok('every tensor agrees on dtype, shape, offset and byte length',
   mismatch === null, mismatch || `all ${hf.tensors.length} checked`);

// The strongest available check that the parse is right rather than merely
// self-consistent: if any offset or block size were wrong, the computed end of the
// last tensor would not land on the real file size.
const last = [...own.tensors].sort((a, b) => a.offset - b.offset).pop();
ok('computed file size matches the real file',
   own.dataStart + last.offset + last.bytes === 639446688,
   String(own.dataStart + last.offset + last.bytes));

// And the planning on top of it, which is what decides the request count.
const ts = [24, 25, 26, 27].flatMap(l => layerTensors(own, l));
const r = byteRanges(own, ts);
ok('a 4-layer bird is ONE merged range', r.length === 1,
   `${(totalBytes(r) / 1e6).toFixed(1)}MB`);
ok('44 tensors across 4 layers', ts.length === 44, String(ts.length));
ok('the merged range matches the Node planner byte for byte',
   r[0].start === Math.min(...ts.map(t => own.dataStart + t.offset)) &&
   r[0].end === Math.max(...ts.map(t => own.dataStart + t.offset + t.bytes)),
   `${r[0].start}-${r[0].end}`);

// The header is ~6MB of a 639MB file and reading it is the whole reason range
// requests are worth the trouble, so the cost is asserted rather than assumed.
ok('the header cost a few MB, not the file', own.headerFetched < 8e6,
   `${(own.headerFetched / 1e6).toFixed(1)}MB in ${own.headerRequests} requests`);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
