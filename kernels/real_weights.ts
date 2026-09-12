// Range-fetch one real Q8_0 tensor out of Qwen3-0.6B on HuggingFace.
//
// Synthetic weights are uniform random, which is the easy case: every block has
// roughly the same scale and the quants fill the int8 range evenly. Real weights
// do not -- scales vary by orders of magnitude across a row, some blocks are
// nearly all zeros, and outliers push a few blocks to saturation. A kernel can
// pass on synthetic data and fail on the real thing, so every kernel here is
// validated against both.
//
// Only the tensor's own byte range is fetched (1-3MB), not the model.

import { readModel, fetchRange, layerTensors } from "../node/src/gguf.mjs";
import type { LayerWeights } from "./layer.ts";

const MODEL = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf";
const CACHE = new URL("./.cache/", import.meta.url);

export interface RealTensor {
  name: string; rows: number; cols: number; packed: Uint8Array;
}

let modelPromise: Promise<Awaited<ReturnType<typeof readModel>>> | null = null;

/**
 * Fetch a Q8_0 tensor by GGUF name. Cached under kernels/.cache so the test
 * suite stays fast and does not hammer HuggingFace; delete that directory to
 * force a re-fetch.
 *
 * GGUF shapes are [in, out] -- ggml is column-major and lists the fastest
 * varying dimension first -- so shape[0] is the input width (our `cols`,
 * contiguous within a row) and shape[1] is the number of output rows.
 */
export async function realQ8Tensor(name: string): Promise<RealTensor> {
  const file = new URL(encodeURIComponent(name) + ".bin", CACHE);
  const meta = new URL(encodeURIComponent(name) + ".json", CACHE);
  try {
    const dims = JSON.parse(await Deno.readTextFile(meta));
    const packed = await Deno.readFile(file);
    return { name, rows: dims.rows, cols: dims.cols, packed };
  } catch {
    // not cached
  }

  modelPromise ??= readModel(MODEL);
  const model = await modelPromise;
  const t = model.tensors.find((x: { name: string }) => x.name === name);
  if (!t) throw new Error(`tensor ${name} not in ${MODEL}`);
  if (t.dtype !== 8) throw new Error(`${name} is dtype ${t.dtype}, expected Q8_0 (8)`);
  const start = model.dataStart + t.offset;
  const buf = await fetchRange(MODEL, { start, end: start + t.bytes });
  const packed = new Uint8Array(buf);
  const [cols, rows] = t.shape;

  await Deno.mkdir(CACHE, { recursive: true });
  await Deno.writeFile(file, packed);
  await Deno.writeTextFile(meta, JSON.stringify({ rows, cols }));
  return { name, rows, cols, packed };
}

/**
 * Every tensor of one layer, split into the Q8_0 projections and the f32 norm
 * gains, keyed by the suffix after `blk.N.` so callers do not repeat the index.
 *
 * The absolute file offset is `model.dataStart + tensor.offset` -- tensor offsets
 * in GGUF are relative to the start of the data section, not the file. Reading
 * them as absolute silently fetches the wrong bytes, which decode as valid-looking
 * Q8_0 and produce plausible garbage rather than an error.
 *
 * Fetched as ONE range request covering the whole layer (~18 MB for Qwen3-0.6B)
 * and sliced locally, because layers are laid out contiguously and one request
 * beats eleven.
 */
export async function realLayer(layer: number): Promise<LayerWeights> {
  const cacheFile = new URL(`layer${layer}.bin`, CACHE);
  const cacheMeta = new URL(`layer${layer}.json`, CACHE);

  modelPromise ??= readModel(MODEL);
  const model = await modelPromise;
  const ts = layerTensors(model, layer);
  if (!ts.length) throw new Error(`no tensors for layer ${layer}`);

  const first = ts[0].offset;
  const last = ts[ts.length - 1];
  const span = last.offset + last.bytes - first;

  let blob: Uint8Array;
  try {
    blob = await Deno.readFile(cacheFile);
    JSON.parse(await Deno.readTextFile(cacheMeta));
    if (blob.byteLength !== span) throw new Error("stale cache");
  } catch {
    const start = model.dataStart + first;
    blob = new Uint8Array(await fetchRange(MODEL, { start, end: start + span }));
    await Deno.mkdir(CACHE, { recursive: true });
    await Deno.writeFile(cacheFile, blob);
    await Deno.writeTextFile(cacheMeta, JSON.stringify({ span, count: ts.length }));
  }

  const out: LayerWeights = { q8: {}, f32: {} };
  const prefix = `blk.${layer}.`;
  for (const t of ts) {
    const key = t.name.slice(prefix.length);
    const at = t.offset - first;
    const bytes = blob.subarray(at, at + t.bytes);
    if (t.dtype === 8) {
      // GGUF shape is [in, out]: shape[0] is the input width.
      const [cols, rows] = t.shape;
      out.q8[key] = { rows, cols, packed: bytes };
    } else if (t.dtype === 0) {
      // f32 norm gain. The slice may not be 4-byte aligned within the blob, so
      // copy rather than aliasing the buffer.
      out.f32[key] = new Float32Array(bytes.slice().buffer);
    } else {
      throw new Error(`${t.name} has unexpected dtype ${t.dtype}`);
    }
  }
  return out;
}

// ------------------------------------------------------------- the whole model

/** Every tensor of Qwen3-0.6B: 28 layers plus the embedding and the final norm. */
export interface RealModel {
  layers: LayerWeights[];
  /** token_embd.weight -- the embedding table AND, tied, the LM head. */
  embd: RealTensor;
  /** output_norm.weight: the final RMSNorm before the vocab projection. */
  outputNorm: Float32Array;
  eos: number;
  bos: number;
  nLayers: number;
  hidden: number;
  vocab: number;
}

/**
 * Load the whole model, cached as ONE file under kernels/.cache.
 *
 * Deliberately a single whole-file download rather than 30 range requests. The
 * per-layer path above exists because a kernel test wants ~18 MB and not 640 MB,
 * but a full forward pass needs every byte anyway, and one sequential read beats
 * thirty range requests that each pay a round trip. The cache is what makes the
 * test suite usable: the download happens once, and every subsequent run reads
 * from disk.
 *
 * The file is validated by SIZE against the directory the header describes, which
 * is what catches a truncated download -- the failure mode that matters, because a
 * short file's tail decodes as valid-looking Q8_0 rather than erroring.
 */
export async function realModel(): Promise<RealModel> {
  const file = new URL("model.gguf", CACHE);
  modelPromise ??= readModel(MODEL);
  const model = await modelPromise;

  // The file is exactly dataStart + (end of the last tensor) bytes.
  let end = 0;
  for (const t of model.tensors) end = Math.max(end, t.offset + t.bytes);
  const wantBytes = model.dataStart + end;

  let blob: Uint8Array;
  try {
    const stat = await Deno.stat(file);
    if (stat.size !== wantBytes) {
      throw new Error(`cached model.gguf is ${stat.size} bytes, want ${wantBytes}`);
    }
    blob = await Deno.readFile(file);
  } catch {
    console.log(`  fetching ${(wantBytes / 1e6).toFixed(0)} MB model to ${file.pathname} ...`);
    const r = await fetch(MODEL);
    if (!r.ok) throw new Error(`model fetch failed: ${r.status}`);
    blob = new Uint8Array(await r.arrayBuffer());
    if (blob.byteLength !== wantBytes) {
      throw new Error(`downloaded ${blob.byteLength} bytes, want ${wantBytes}`);
    }
    await Deno.mkdir(CACHE, { recursive: true });
    await Deno.writeFile(file, blob);
  }

  // Tensor offsets are relative to dataStart, not to the file. Reading them as
  // absolute fetches wrong bytes that decode as valid-looking Q8_0.
  const at = (t: { offset: number; bytes: number }) =>
    blob.subarray(model.dataStart + t.offset, model.dataStart + t.offset + t.bytes);
  const find = (name: string) => {
    const t = model.tensors.find((x: { name: string }) => x.name === name);
    if (!t) throw new Error(`tensor ${name} not in ${MODEL}`);
    return t;
  };

  const layers: LayerWeights[] = [];
  for (let l = 0; l < model.nLayers; l++) {
    const w: LayerWeights = { q8: {}, f32: {} };
    const prefix = `blk.${l}.`;
    for (const t of layerTensors(model, l)) {
      const key = t.name.slice(prefix.length);
      const bytes = at(t);
      if (t.dtype === 8) {
        const [cols, rows] = t.shape;
        w.q8[key] = { rows, cols, packed: bytes };
      } else if (t.dtype === 0) {
        // slice() first: the subarray's byteOffset is not 4-byte aligned in
        // general, and Float32Array over an unaligned offset throws.
        w.f32[key] = new Float32Array(bytes.slice().buffer);
      } else {
        throw new Error(`${t.name} has unexpected dtype ${t.dtype}`);
      }
    }
    layers.push(w);
  }

  const et = find("token_embd.weight");
  if (et.dtype !== 8) throw new Error(`token_embd is dtype ${et.dtype}, expected Q8_0`);
  const [ecols, erows] = et.shape;
  const nt = find("output_norm.weight");

  return {
    layers,
    embd: { name: et.name, rows: erows, cols: ecols, packed: at(et) },
    outputNorm: new Float32Array(at(nt).slice().buffer),
    eos: Number(model.metadata["tokenizer.ggml.eos_token_id"]),
    bos: Number(model.metadata["tokenizer.ggml.bos_token_id"]),
    nLayers: model.nLayers,
    hidden: Number(model.metadata[`${model.arch}.embedding_length`]),
    vocab: erows,
  };
}
