// flock/gguf — plan which bytes of a GGUF model each device needs.
//
// A GGUF file is a directory followed by a blob: the first few MB list every
// tensor with its exact byte offset, and everything after is raw weights. So a
// device can range-fetch ONLY the layers it owns -- ~67MB for 4 of 28 layers
// instead of a whole model -- and the host needs no build step and no disk,
// because peers pull from the model host directly. It also makes the split
// DYNAMIC: layer ranges are just byte ranges, so the coordinator can decide
// them when devices join rather than baking them into exported files.
//
// The directory parsing itself is @huggingface/gguf, which already does this
// properly (and range-fetches the header in 2MB chunks). What is ours is the
// part that matters here: turning "which layers does this bird own" into the
// fewest possible HTTP range requests.

import {gguf, GGMLQuantizationType} from '@huggingface/gguf';

// block_size / type_size per ggml: a quantized "block" is N weights sharing
// one or two scale factors, so bytes-per-tensor is not elements * width.
const BLOCK = {
  [GGMLQuantizationType.F32]:  {block: 1,   bytes: 4},
  [GGMLQuantizationType.F16]:  {block: 1,   bytes: 2},
  [GGMLQuantizationType.Q4_0]: {block: 32,  bytes: 18},
  [GGMLQuantizationType.Q4_1]: {block: 32,  bytes: 20},
  [GGMLQuantizationType.Q5_0]: {block: 32,  bytes: 22},
  [GGMLQuantizationType.Q5_1]: {block: 32,  bytes: 24},
  [GGMLQuantizationType.Q8_0]: {block: 32,  bytes: 34},
  [GGMLQuantizationType.Q8_1]: {block: 32,  bytes: 40},
  [GGMLQuantizationType.Q2_K]: {block: 256, bytes: 84},
  [GGMLQuantizationType.Q3_K]: {block: 256, bytes: 110},
  [GGMLQuantizationType.Q4_K]: {block: 256, bytes: 144},
  [GGMLQuantizationType.Q5_K]: {block: 256, bytes: 176},
  [GGMLQuantizationType.Q6_K]: {block: 256, bytes: 210},
  [GGMLQuantizationType.Q8_K]: {block: 256, bytes: 292},
};

export function tensorBytes(shape, dtype) {
  const spec = BLOCK[dtype];
  if (!spec) throw new Error(`unsupported ggml type ${dtype}`);
  const n = shape.reduce((a, b) => a * Number(b), 1);
  if (n % spec.block !== 0) {
    throw new Error(`${n} elements not divisible by block ${spec.block}`);
  }
  return (n / spec.block) * spec.bytes;
}

/** Read a model's directory. Only the header is fetched, not the weights. */
export async function readModel(url) {
  const {metadata, tensorInfos, tensorDataOffset} = await gguf(url, {allowLocalFile: true});
  const arch = metadata['general.architecture'];
  const tensors = tensorInfos.map(t => ({
    name: t.name,
    shape: t.shape.map(Number),
    dtype: t.dtype,
    offset: Number(t.offset),
    bytes: tensorBytes(t.shape, t.dtype),
  }));
  return {
    url, arch, metadata, tensors,
    nLayers: Number(metadata[`${arch}.block_count`]),
    dataStart: Number(tensorDataOffset),
  };
}

/** Every tensor a given layer owns, e.g. layerTensors(m, 24) -> blk.24.*  */
export function layerTensors(model, layer) {
  const prefix = `blk.${layer}.`;
  return model.tensors
    .filter(t => t.name.startsWith(prefix))
    .sort((a, b) => a.offset - b.offset);
}

/**
 * Absolute byte ranges covering `layers`, merged where they are close enough
 * that one request beats two. Layers are laid out sequentially in GGUF, so a
 * contiguous slice usually collapses to a SINGLE range request.
 */
export function byteRanges(model, layers, {gap = 1 << 20} = {}) {
  const ts = layers.flatMap(l => layerTensors(model, l)).sort((a, b) => a.offset - b.offset);
  const out = [];
  for (const t of ts) {
    const start = model.dataStart + t.offset, end = start + t.bytes;
    const last = out[out.length - 1];
    if (last && start - last.end <= gap) last.end = Math.max(last.end, end);
    else out.push({start, end});
  }
  return out;
}

export function totalBytes(ranges) {
  return ranges.reduce((a, r) => a + (r.end - r.start), 0);
}

/** Fetch one range. Returns an ArrayBuffer of exactly the requested bytes. */
export async function fetchRange(url, {start, end}) {
  const r = await fetch(url, {headers: {Range: `bytes=${start}-${end - 1}`}});
  if (!r.ok && r.status !== 206) throw new Error(`range fetch failed: ${r.status}`);
  return r.arrayBuffer();
}

/** Split `nLayers` starting at `first` across `n` devices, as evenly as possible. */
export function splitLayers(first, last, n) {
  const total = last - first + 1;
  if (n > total) throw new Error(`can't split ${total} layers across ${n} devices`);
  const per = Math.floor(total / n), extra = total % n;
  const out = [];
  let cur = first;
  for (let i = 0; i < n; i++) {
    const k = per + (i < extra ? 1 : 0);
    out.push([cur, cur + k - 1]);
    cur += k;
  }
  return out;
}
