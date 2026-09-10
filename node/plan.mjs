// plan.mjs — what would each device fetch, for a given model and split?
//
// Answers the question the GGUF path exists to answer, without building
// anything: no export, no host disk, just the model's directory.
//
//   node plan.mjs --birds 2
//   node plan.mjs --url <gguf> --coord 20 --birds 3
import {loadDirectory, httpRange, byteRanges, totalBytes} from '../web/js/gguf.mjs';

const arg = (k, d) => {
  const i = process.argv.indexOf('--' + k);
  return i > 0 ? process.argv[i + 1] : d;
};
const URL = arg('url',
  'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf');
const nBirds = +arg('birds', 1);

console.log(`reading directory: ${URL.split('/').pop()}`);
const dir = await loadDirectory(httpRange(URL));
const arch = dir.meta['general.architecture'];
const nLayers = dir.meta[`${arch}.block_count`];
const coordLayers = +arg('coord', Math.max(1, nLayers - 4));

console.log(`  ${arch}, ${nLayers} layers, directory ${(dir.directoryBytes/1e6).toFixed(1)}MB\n`);

const birdFirst = coordLayers;
const total = nLayers - birdFirst;
if (nBirds > total) { console.error(`can't split ${total} layers across ${nBirds} birds`); process.exit(1); }
const per = Math.floor(total / nBirds), extra = total % nBirds;

let cur = birdFirst, sum = 0;
const coordRange = byteRanges(dir, [...Array(coordLayers).keys()]);
const coordMB = totalBytes(coordRange) / 1e6;
console.log(`coordinator  layers 0-${coordLayers-1}   ${coordMB.toFixed(1)} MB  (${coordRange.length} range${coordRange.length>1?'s':''})`);
sum += coordMB;

for (let i = 0; i < nBirds; i++) {
  const n = per + (i < extra ? 1 : 0);
  const layers = [...Array(n).keys()].map(k => cur + k);
  const r = byteRanges(dir, layers);
  const mb = totalBytes(r) / 1e6;
  console.log(`bird ${i}       layers ${layers[0]}-${layers[n-1]}  ${mb.toFixed(1)} MB  ` +
              `(${r.length} range${r.length>1?'s':''}, +${(dir.directoryBytes/1e6).toFixed(1)}MB directory)`);
  sum += mb; cur += n;
}
console.log(`\n  total weights moved: ${sum.toFixed(0)} MB`);
console.log(`  host build step:     none — every device fetches its own bytes`);
