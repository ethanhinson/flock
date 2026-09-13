// plan.mjs — what would each device fetch, for a given model and split?
//
// Answers the question the GGUF path exists to answer, without building
// anything: no export, no host disk. Only the model's header is read.
//
//   npm run plan -- --birds 2
//   npm run plan -- --url <gguf> --coord 20 --birds 3
import {readModel, byteRanges, totalBytes, splitLayers} from '../server/gguf.mjs';

const arg = (k, d) => {
  const i = process.argv.indexOf('--' + k);
  return i > 0 ? process.argv[i + 1] : d;
};
const URL = arg('url',
  'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf');
const nBirds = +arg('birds', 1);

console.log(`reading header: ${URL.split('/').pop()}`);
const m = await readModel(URL);
const coordLayers = +arg('coord', Math.max(1, m.nLayers - 4));
console.log(`  ${m.arch}, ${m.nLayers} layers\n`);

const coord = byteRanges(m, [...Array(coordLayers).keys()]);
console.log(`coordinator  layers 0-${coordLayers - 1}   ` +
            `${(totalBytes(coord) / 1e6).toFixed(1)} MB  (${coord.length} range)`);

let sum = totalBytes(coord) / 1e6;
for (const [i, [a, b]] of splitLayers(coordLayers, m.nLayers - 1, nBirds).entries()) {
  const r = byteRanges(m, [...Array(b - a + 1).keys()].map(k => a + k));
  const mb = totalBytes(r) / 1e6;
  console.log(`bird ${i}       layers ${a}-${b}  ${mb.toFixed(1)} MB  ` +
              `(${r.length} range${r.length > 1 ? 's' : ''})`);
  sum += mb;
}
console.log(`\n  total weights moved: ${sum.toFixed(0)} MB`);
console.log(`  host build step:     none — every device fetches its own bytes`);
