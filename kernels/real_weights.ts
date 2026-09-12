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

import { readModel, fetchRange } from "../node/src/gguf.mjs";

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
