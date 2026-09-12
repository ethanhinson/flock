// Prove the streaming loader against the real Qwen3-0.6B file on HuggingFace.
//
//   npm run test:gpu       (from node/)
//   deno run --config kernels/deno.json --unstable-webgpu --allow-all \
//     web/js/gguf-stream.test.ts
//
// The kernels config is needed only because this test imports the kernels' own
// reference loader to compare against, and that one is Node-side.
//
// Three claims are under test, and each is the kind that a unit test over a
// fixture cannot make:
//
//   1. CORRECTNESS. The bytes that land in the GPU buffers are byte-identical to
//      fetchRange + splitQ8, and -- for a whole layer -- to what realLayer() in
//      kernels/real_weights.ts hands the engine today. That is the reference the
//      kernels are already validated against, so matching it is what makes these
//      buffers substitutable rather than merely plausible.
//
//   2. PEAK MEMORY. The claim is that peak JS memory tracks the STAGING size, not
//      the slice size -- the entire reason this module exists. Measured against a
//      real 67MB slice, against a measured floor for the fetch stack alone, and
//      against a control that does it the naive way so the comparison is real.
//
//   3. REQUEST COUNT. A 4-layer bird should fetch its ~67MB in one range
//      request, because GGUF lays layers out contiguously.
//
// Network-dependent by design. It also runs under Deno rather than in a browser,
// which is the honest limitation: Deno's wgpu is not WebKit, so this shows the
// loader correct and bounded, not that a specific iPad survives.

import {
  loadLayersToGPU, readDirectory, layerTensors, byteRanges, totalBytes,
  streamTensorToGPU, fetchRange, STAGING,
} from "./gguf-stream.mjs";
import { splitQ8 } from "../../kernels/lib.ts";
import { realLayer } from "../../kernels/real_weights.ts";

const MODEL =
  "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf";

let pass = 0, fail = 0;
const ok = (name: string, cond: boolean, extra = "") => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? "  " + extra : ""}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? "  " + extra : ""}`); }
};
const MB = (n: number) => (n / 1e6).toFixed(1) + "MB";

// deno-lint-ignore no-explicit-any
const dev: any = await (async () => {
  const adapter = await (navigator as any).gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  return adapter.requestDevice();
})();

/** Read a GPU buffer back as bytes, so it can be diffed against a reference. */
async function readBytes(buf: GPUBuffer, bytes: number): Promise<Uint8Array> {
  const size = (bytes + 3) & ~3;
  const rd = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rd, 0, size);
  dev.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(rd.getMappedRange().slice(0, bytes));
  rd.unmap(); rd.destroy();
  return out;
}

function firstDiff(a: Uint8Array, b: Uint8Array): number {
  if (a.byteLength !== b.byteLength) return -2;
  for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return i;
  return -1;
}

// WHICH METER, AND WHY. Getting this right took three attempts and each wrong
// answer was wrong in an interesting way, so all three are recorded:
//
//   heapUsed alone UNDERSTATES everything. V8 allocates a large ArrayBuffer's
//   backing store outside the JS heap, so a naive loader holding a 17MB
//   arrayBuffer() reports ~2MB of heap. Measured: 1.9MB heap for a 16.7MB slice.
//   A module measured this way looks free while it holds the whole tensor.
//
//   rss alone OVERSTATES it. RSS grows by roughly the slice size no matter what
//   the loader does, because the weights really do end up resident -- in the GPU
//   driver's allocations, which is where they are supposed to be. Verified
//   directly: 16 writeBuffer calls of 4MB from ONE reused 4MB source array grew
//   RSS by 71.8MB and `external` by 0.0MB.
//
//   heapUsed + external is the honest meter for JS-owned memory, and that same
//   experiment is what establishes it: `external` counts backing stores the JS
//   side owns and does NOT count the driver's staging.
//
// One more correction on top of that. Deno's own fetch body reader buffers
// internally, and it is not small: reading a 67MB range as a bare stream, with no
// GPU and no splitting at all, peaks at +13.2MB external / +2.5MB heap. That is
// the HTTP stack, not this module, and no loader written on top of it can be
// cheaper. So the floor is measured here rather than assumed, and the loader is
// charged only for what it adds above it.
const rss = () => Deno.memoryUsage().rss;
const heap = () => Deno.memoryUsage().heapUsed;
const external = () => Deno.memoryUsage().external;
const jsMem = () => heap() + external();

/** What Deno's fetch stack alone costs to read `bytes` as a stream. */
async function streamFloor(start: number, bytes: number) {
  await new Promise((r) => setTimeout(r, 50));
  const base = jsMem();
  let peak = base;
  const res = await fetch(MODEL, {
    headers: { Range: `bytes=${start}-${start + bytes - 1}` },
  });
  const rd = res.body!.getReader();
  for (;;) {
    const { value, done } = await rd.read();
    if (done) break;
    if (value.byteLength === 0) continue;
    const m = jsMem();
    if (m > peak) peak = m;
    await new Promise((r) => setTimeout(r, 0));
  }
  return peak - base;
}

console.log("reading the directory over range requests...");
const model = await readDirectory(MODEL);
ok("arch is qwen3", model.arch === "qwen3", model.arch);
ok("28 layers", model.nLayers === 28, String(model.nLayers));
ok("310 tensors", model.tensors.length === 310, String(model.tensors.length));
// The strongest available check on the parse: if any offset or block size were
// wrong, the computed end of the last tensor would not land on the real file size.
const last = [...model.tensors].sort((a, b) => a.offset - b.offset).pop()!;
ok("computed file size matches the real file",
  model.dataStart + last.offset + last.bytes === 639446688,
  String(model.dataStart + last.offset + last.bytes));
ok("header cost only a few MB", model.headerFetched < 8e6,
  `${MB(model.headerFetched)} in ${model.headerRequests} requests`);

// ---------------------------------------------------------------- correctness
//
// One real tensor, streamed to the GPU, read back, and compared against the
// reference the kernels already trust. ffn_down is the largest in a layer (3.3MB
// = 98304 blocks), so it crosses many staging boundaries and exercises the
// partial-block carry rather than getting a lucky alignment.

console.log("\nstreaming blk.24.ffn_down.weight and diffing against splitQ8...");
const t = model.tensors.find((x) => x.name === "blk.24.ffn_down.weight")!;
const absolute = model.dataStart + t.offset;
const up = await streamTensorToGPU(
  dev, { ...t, absolute },
  (s: number, e: number) => fetch(MODEL, { headers: { Range: `bytes=${s}-${e - 1}` } }),
  { staging: STAGING },
);

const packed = await fetchRange(MODEL, { start: absolute, end: absolute + t.bytes });
const ref = splitQ8(packed, up.rows, up.cols);
const gotQs = await readBytes(up.qs!, up.qsBytes!);
const gotSc = await readBytes(up.scales!, up.scaleBytes!);
ok("qs bytes are byte-identical to splitQ8", firstDiff(gotQs, ref.qs) === -1,
  `${MB(gotQs.byteLength)}, first diff at ${firstDiff(gotQs, ref.qs)}`);
ok("scale bytes are byte-identical to splitQ8", firstDiff(gotSc, ref.scales) === -1,
  `${MB(gotSc.byteLength)}, first diff at ${firstDiff(gotSc, ref.scales)}`);
ok("shape read as [cols, rows] from GGUF's [in, out]",
  up.rows === 1024 && up.cols === 3072, `${up.rows}x${up.cols}`);

// A staging size that is NOT a multiple of 34 is the case that breaks a naive
// splitter: every flush lands mid-block, so the partial-block carry runs on
// every single chunk instead of occasionally.
console.log("\nre-streaming with a deliberately block-hostile staging size (100003 bytes)...");
const odd = await streamTensorToGPU(
  dev, { ...t, absolute },
  (s: number, e: number) => fetch(MODEL, { headers: { Range: `bytes=${s}-${e - 1}` } }),
  { staging: 100003 },
);
const oddQs = await readBytes(odd.qs!, odd.qsBytes!);
const oddSc = await readBytes(odd.scales!, odd.scaleBytes!);
ok("misaligned staging still produces identical qs", firstDiff(oddQs, ref.qs) === -1,
  `first diff at ${firstDiff(oddQs, ref.qs)}`);
ok("misaligned staging still produces identical scales", firstDiff(oddSc, ref.scales) === -1,
  `first diff at ${firstDiff(oddSc, ref.scales)}`);
odd.qs!.destroy(); odd.scales!.destroy();
up.qs!.destroy(); up.scales!.destroy();

// --------------------------------------------------------- a whole 4-layer bird
//
// The real workload: layers 24-27, which is what a bird owns in the default
// split. Memory is sampled on every progress callback, so the peak reported is
// the peak DURING the streaming rather than after it.

console.log("\nloading layers 24-27 (a real bird's slice) with memory sampling...");
// Let the header's fetch buffers go before taking a baseline: they are ~6MB of
// directory, which would otherwise be charged to the streaming claim.
await new Promise((r) => setTimeout(r, 50));
const rss0 = rss(), heap0 = heap(), ext0 = external(), js0 = heap0 + ext0;
let peakRss = rss0, peakHeap = heap0, peakExt = ext0, peakJs = js0, samples = 0;
let lastNote = "";

const t0 = performance.now();
const { layers, stats } = await loadLayersToGPU(dev, model, [24, 25, 26, 27], {
  staging: STAGING,
  // No IndexedDB under Deno, and the cache fill would hold the whole 67MB slice
  // in JS on purpose -- which is exactly what this measurement must exclude.
  cache: false,
  onNote: (m: string) => { lastNote = m; },
  onProgress: () => {
    samples++;
    const r = rss(), h = heap(), e = external();
    if (r > peakRss) peakRss = r;
    if (h > peakHeap) peakHeap = h;
    if (e > peakExt) peakExt = e;
    if (h + e > peakJs) peakJs = h + e;
  },
});
const secs = (performance.now() - t0) / 1000;

console.log(`  streamed ${MB(stats.bytesTotal)} in ${secs.toFixed(1)}s ` +
  `(${(stats.bytesTotal / 1e6 / secs).toFixed(0)} MB/s), ${samples} progress samples`);
console.log(`  rss   baseline ${MB(rss0)}  peak ${MB(peakRss)}  delta ${MB(peakRss - rss0)}`);
console.log(`  heap  baseline ${MB(heap0)}  peak ${MB(peakHeap)}  delta ${MB(peakHeap - heap0)}`);
console.log(`  ext   baseline ${MB(ext0)}  peak ${MB(peakExt)}  delta ${MB(peakExt - ext0)}`);
console.log(`  js    heap+ext delta ${MB(peakJs - js0)}   (rss delta is the GPU driver's copy)`);
if (lastNote) console.log(`  note: ${lastNote}`);

ok("a 4-layer bird is ONE range request", stats.requests === 1, `${stats.requests} requests`);
ok("one merged range covers the slice", stats.ranges === 1, `${stats.ranges} ranges`);
ok("44 tensors across 4 layers", stats.tensors === 44, String(stats.tensors));
ok("fetched ~67MB", stats.bytesTotal > 66e6 && stats.bytesTotal < 68e6, MB(stats.bytesTotal));

// THE claim. Peak JS growth should be on the order of the staging buffers, not
// the 67MB slice. The bound is 4x staging rather than 1x because the streamer
// legitimately holds several: the coalescing buffer, the qs output, the scales
// output, and one network chunk in flight. What it must NOT hold is a tensor.
const streamDelta = peakJs - js0;

// What does Deno's fetch stack cost on its own for the same number of bytes? The
// loader is charged only for what it adds on top, because nothing written over
// this fetch implementation can go below it.
console.log("\nmeasuring the floor: the same 67MB read as a bare stream, no GPU...");
const floor = await streamFloor(model.dataStart, stats.bytesTotal);
const overhead = streamDelta - floor;
console.log(`  fetch floor ${MB(floor)}   loader adds ${MB(overhead)}`);

// The loader's own working set is the coalescing buffer (4MB) plus the qs output
// (4MB) plus the scales output (0.25MB), so ~8.25MB is the predicted figure and
// 3x staging is the bound. What it must NOT be is the slice size.
const bound = 3 * STAGING;
ok("the loader's own JS memory is a few staging buffers, not the slice",
  overhead < bound,
  `${MB(overhead)} above the fetch floor < ${MB(bound)} (slice is ${MB(stats.bytesTotal)})`);
ok("peak JS memory is a small fraction of the slice",
  streamDelta < stats.bytesTotal / 3,
  `${MB(streamDelta)} = ${(streamDelta / stats.bytesTotal * 100).toFixed(1)}% of ${MB(stats.bytesTotal)}`);
ok("peak JS memory does not scale with the slice: 4 layers cost under 2x one layer's bytes",
  streamDelta < 2 * (stats.bytesTotal / 4),
  `${MB(streamDelta)} for 4 layers of ${MB(stats.bytesTotal / 4)} each`);

// The shape the kernels consume: layers[24].attn_q, keyed by the suffix after
// blk.N. -- so these buffers can be handed to kernels/layer.ts directly.
ok("keyed by layer then tensor suffix",
  !!(layers[24]?.["attn_q.weight"] && layers[27]?.["ffn_down.weight"] &&
     layers[24]?.["attn_norm.weight"]),
  Object.keys(layers[24] ?? {}).sort().join(","));
ok("norm gains are f32 single buffers, not split",
  !!layers[24]["attn_norm.weight"].data && !layers[24]["attn_norm.weight"].qs,
  `attn_norm ${layers[24]["attn_norm.weight"].dataBytes} bytes`);
ok("projections are split into qs + scales",
  !!(layers[24]["attn_q.weight"].qs && layers[24]["attn_q.weight"].scales),
  `attn_q qs ${MB(layers[24]["attn_q.weight"].qsBytes)} ` +
  `scales ${MB(layers[24]["attn_q.weight"].scaleBytes)}`);

// Total GPU bytes should be LESS than the file bytes: the split layout drops
// nothing, but it also does not pad -- 34 bytes on disk becomes 32 + 2.
let gpuBytes = 0;
for (const l of Object.values(layers) as any[]) {
  for (const v of Object.values(l) as any[]) {
    gpuBytes += (v.qsBytes ?? 0) + (v.scaleBytes ?? 0) + (v.dataBytes ?? 0);
  }
}
ok("GPU bytes equal file bytes (the split rearranges, it does not grow)",
  gpuBytes === stats.bytesTotal, `${MB(gpuBytes)} vs ${MB(stats.bytesTotal)}`);

// One spot-check that the whole-slice path lands the same bytes as the
// single-tensor path did: a different layer, a different tensor.
const spot = model.tensors.find((x) => x.name === "blk.26.attn_q.weight")!;
const spotAbs = model.dataStart + spot.offset;
const spotPacked = await fetchRange(MODEL, { start: spotAbs, end: spotAbs + spot.bytes });
const spotRef = splitQ8(spotPacked, Number(spot.shape[1]), Number(spot.shape[0]));
const spotGot = await readBytes(layers[26]["attn_q.weight"].qs,
  layers[26]["attn_q.weight"].qsBytes);
ok("a tensor from the middle of the slice matches splitQ8 too",
  firstDiff(spotGot, spotRef.qs) === -1,
  `blk.26.attn_q, first diff at ${firstDiff(spotGot, spotRef.qs)}`);

// ------------------------------------------- substitutable for the engine's own
//
// The strongest correctness claim available, and the one that actually matters to
// the WGSL engine: for a whole layer, do these GPU buffers hold the same bytes the
// engine gets TODAY from realLayer() + splitQ8? Every projection and every norm
// gain, not a sample.
//
// This is also what caught the one real integration bug in the loader:
// kernels/real_weights.ts keys its tensors WITH the trailing `.weight`
// (`attn_q.weight`), and the loader was stripping it. Every lookup in
// kernels/layer.ts would have returned undefined -- a silent failure, not an
// error. Comparing against the real reference is what surfaced it; comparing
// against a fixture of my own making would not have.

console.log("\nchecking a whole layer against kernels/real_weights.ts...");
const ref24 = await realLayer(24);
const { layers: one } = await loadLayersToGPU(dev, model, [24], { cache: false });
let checked = 0, bad: string[] = [];
for (const [k, ent] of Object.entries(ref24.q8)) {
  const got = one[24][k];
  if (!got?.qs) { bad.push(`${k} missing (key mismatch)`); continue; }
  const exp = splitQ8(ent.packed, ent.rows, ent.cols);
  if (firstDiff(await readBytes(got.qs, got.qsBytes), exp.qs) !== -1) bad.push(`${k} qs`);
  if (firstDiff(await readBytes(got.scales, got.scaleBytes), exp.scales) !== -1) {
    bad.push(`${k} scales`);
  }
  if (got.rows !== ent.rows || got.cols !== ent.cols) {
    bad.push(`${k} dims ${got.rows}x${got.cols} vs ${ent.rows}x${ent.cols}`);
  }
  checked++;
}
for (const [k, v] of Object.entries(ref24.f32)) {
  const got = one[24][k];
  if (!got?.data) { bad.push(`${k} missing (key mismatch)`); continue; }
  const exp = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  if (firstDiff(await readBytes(got.data, got.dataBytes), exp) !== -1) bad.push(`${k} f32`);
  checked++;
}
ok("every tensor of a layer is byte-identical to what the kernels load today",
  bad.length === 0 && checked === 11,
  bad.length ? bad.join("; ") : `${checked}/11 tensors, 7 quantized + 4 f32 gains`);
ok("keyed the way kernels/real_weights.ts keys them (suffix keeps `.weight`)",
  !!one[24]["attn_q.weight"] && !one[24]["attn_q"],
  Object.keys(one[24]).sort().join(","));

// ------------------------------------------------- the control: the naive path
//
// The streaming numbers above mean nothing without the thing they are compared
// against. This does it the obvious way -- arrayBuffer() the whole slice, then
// splitQ8 it -- and measures the same heap delta. If the naive path were also
// cheap, the streaming machinery would be unjustified complexity, so this is the
// measurement that decides whether the module should exist.

console.log("\ncontrol: the same slice the naive way (arrayBuffer + splitQ8)...");
const slice = byteRanges(model, layerTensors(model, 24));
await new Promise((r) => setTimeout(r, 50));
const nHeap0 = heap(), nExt0 = external();
let nPeakHeap = nHeap0, nPeakExt = nExt0;
{
  // One layer, not four: the control holds the bytes, so four would be 67MB and
  // the comparison can be made honestly at 17MB.
  const res = await fetch(MODEL, {
    headers: { Range: `bytes=${slice[0].start}-${slice[0].end - 1}` },
  });
  const all = new Uint8Array(await res.arrayBuffer());
  nPeakHeap = Math.max(nPeakHeap, heap());
  nPeakExt = Math.max(nPeakExt, external());
  const ts24 = layerTensors(model, 24);
  const big = ts24.find((x) => x.name === "blk.24.ffn_down.weight")!;
  const at = big.offset - ts24[0].offset;
  const s = splitQ8(all.subarray(at, at + big.bytes), 1024, 3072);
  nPeakHeap = Math.max(nPeakHeap, heap());
  nPeakExt = Math.max(nPeakExt, external());
  // Referenced so the optimizer cannot drop the work being measured.
  if (s.qs.byteLength === 0) throw new Error("unreachable");
}
const naiveDelta = (nPeakHeap - nHeap0) + (nPeakExt - nExt0);
const sliceBytes = totalBytes(slice);
console.log(`  naive:     heap +${MB(nPeakHeap - nHeap0)}  external +${MB(nPeakExt - nExt0)}` +
  `  = ${MB(naiveDelta)} for ONE ${MB(sliceBytes)} layer`);
console.log(`  streaming: heap +${MB(peakHeap - heap0)}  external +${MB(peakExt - ext0)}` +
  `  = ${MB(streamDelta)} for FOUR layers (${MB(stats.bytesTotal)})`);

// Per byte delivered is the comparison that survives the two paths moving
// different amounts of data. The naive path holds the slice, so its cost per byte
// is ~1; the streaming path's should be far below that and should FALL as the
// slice grows, which is the actual claim.
const naivePerByte = naiveDelta / sliceBytes;
const streamPerByte = streamDelta / stats.bytesTotal;
console.log(`  JS memory per byte delivered: naive ${naivePerByte.toFixed(2)}x, ` +
  `streaming ${streamPerByte.toFixed(2)}x`);
ok("the naive path holds most of what it downloads (so streaming is not busywork)",
  naivePerByte > 0.5, `${naivePerByte.toFixed(2)}x of ${MB(sliceBytes)}`);
ok("the streaming path holds a small fraction of what it downloads",
  streamPerByte < 0.4, `${streamPerByte.toFixed(2)}x of ${MB(stats.bytesTotal)}`);
ok("streaming is cheaper per byte than the naive path by a wide margin",
  streamPerByte < naivePerByte / 2,
  `${streamPerByte.toFixed(2)}x vs ${naivePerByte.toFixed(2)}x`);

console.log(`\n${pass} passed, ${fail} failed`);
Deno.exit(fail ? 1 : 0);
