// Causal grouped-query attention for N query positions at once, with a STREAMING
// (online) softmax so there is no cap on the key count.
//
//   scores[i][t] = dot(q[i,h], k[t,kvh]) / sqrt(head_dim),  t <= pos0 + i
//   out[i][h]    = sum_t softmax(scores[i])[t] * v[t,kvh]
//
// This is the prefill counterpart to attention.wgsl. That kernel is the right
// shape for decode -- one query, the whole cache visible, all scores held in
// workgroup memory so the softmax is one pass -- and the wrong shape here for two
// reasons:
//
//  1. Prefill has N queries, each with a DIFFERENT causal bound. Running the
//     decode kernel N times would work but would recompute nothing useful and
//     would need N dispatches.
//  2. Its scores array is $MAX_KEYS f32 of workgroup memory, which caps context
//     at 2048 and -- per the note in that file -- cannot simply be enlarged,
//     because workgroup memory is an occupancy cliff: 16 KB made it 15x slower.
//
// So this kernel never materializes a score row. It is the flash-attention online
// softmax: walk the keys in tiles of TILE, and maintain a running max `m`, a
// running denominator `l`, and a running unnormalized output accumulator `acc`.
// When a tile's max exceeds the running one, both `l` and `acc` are rescaled by
// exp(m_old - m_new) before the tile is folded in. Workgroup memory is then
// O(TILE), not O(n_keys), and the arithmetic is identical to a two-pass softmax
// up to f32 rounding -- which is what test_prefill.ts checks against attnRef.
//
// ONE WORKGROUP PER (query, head). The alternative -- one workgroup per head
// walking all N queries -- reuses the k/v tile across queries and would be the
// faster choice at large N, but it serializes the queries and prefill's whole
// point is parallelism: a 20-token prompt at 16 heads is 320 workgroups, which is
// what keeps the GPU busy. The k tile is re-read per query from L2, not from
// device memory, so the cost of not sharing it is small.
//
// The causal bound is computed, not masked. Query i sits at absolute position
// pos0 + i, so it attends to keys 0..pos0+i inclusive; the loop bound IS the
// mask, and no -inf mask tensor is built or uploaded. That matters beyond
// tidiness: the ONNX graph materializes an N x (past+N) Trilu mask every prefill,
// which for a 1000-token prompt is 4 MB of traffic to encode a comparison.

struct Dims {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  n_queries: u32,   // N: how many query positions in this batch
  pos0: u32,        // absolute position of query 0 (== past length)
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       q: array<f32>;   // [n_queries][n_heads][head_dim]
@group(0) @binding(1) var<storage, read>       k: array<f32>;   // [n_keys][n_kv_heads][head_dim]
@group(0) @binding(2) var<storage, read>       v: array<f32>;   // same layout as k
@group(0) @binding(3) var<storage, read_write> o: array<f32>;   // [n_queries][n_heads][head_dim]
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 128u;
// One tile of keys per iteration. 64 scores is 256 bytes of workgroup memory, so
// occupancy is not affected at all -- the whole reason for the streaming form.
// It is >= 1 warp so the score computation is not half-idle, and the head_dim=128
// output loop below is WG-wide regardless of it.
const TILE: u32 = 64u;

var<workgroup> tile_s: array<f32, TILE>;    // this tile's scores, then its exp()
var<workgroup> part: array<f32, WG>;        // reduction scratch (max, then sum)
var<workgroup> tile_max: f32;
var<workgroup> tile_sum: f32;
// The running output accumulator, one f32 per output channel. head_dim is 128 for
// Qwen3, so this is 512 bytes. Sized to WG because the output loop is strided by
// WG and head_dim <= WG for every model this engine targets; larger head dims
// would need the loop to walk it in WG-sized chunks, which the guard below makes
// explicit rather than silently wrong.
var<workgroup> acc: array<f32, WG>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let hd = d.head_dim;
  // Workgroup (query, head): x is the head, y the query position.
  let h = wg.x;
  let qi = wg.y;
  if (h >= d.n_heads || qi >= d.n_queries) { return; }
  // head_dim > WG would silently drop channels, so refuse rather than corrupt.
  if (hd > WG) { return; }

  let kvh = h / (d.n_heads / d.n_kv_heads);
  let scale = 1.0 / sqrt(f32(hd));
  let qbase = (qi * d.n_heads + h) * hd;
  // Causal: query qi is at absolute position pos0 + qi and sees keys 0..that.
  let n_keys = d.pos0 + qi + 1u;

  // Running softmax state. `m` and `l` are replicated in every thread (they are
  // derived from workgroup-wide reductions, so they agree by construction);
  // `acc` is shared because output channels are split across threads.
  var m = -3.4028235e38;
  var l = 0.0;
  if (t < hd) { acc[t] = 0.0; }
  workgroupBarrier();

  var base = 0u;
  while (base < n_keys) {
    let n_this = min(TILE, n_keys - base);

    // --- this tile's scores -------------------------------------------------
    for (var j = t; j < n_this; j = j + WG) {
      let kbase = ((base + j) * d.n_kv_heads + kvh) * hd;
      var s = 0.0;
      for (var i = 0u; i < hd; i = i + 1u) {
        s = s + q[qbase + i] * k[kbase + i];
      }
      tile_s[j] = s * scale;
    }
    workgroupBarrier();

    // --- tile max -----------------------------------------------------------
    var tm = -3.4028235e38;
    for (var j = t; j < n_this; j = j + WG) { tm = max(tm, tile_s[j]); }
    part[t] = tm;
    workgroupBarrier();
    var stride = WG / 2u;
    while (stride > 0u) {
      if (t < stride) { part[t] = max(part[t], part[t + stride]); }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    if (t == 0u) { tile_max = part[0]; }
    workgroupBarrier();

    // The new running max, and the factor that rescales everything accumulated
    // so far onto it. When this tile is entirely below the running max (common
    // after the first tile) `corr` is exp(0) == 1 and the rescale is a no-op.
    let m_new = max(m, tile_max);
    let corr = exp(m - m_new);

    // --- exponentiate against m_new, and sum -------------------------------
    var ts = 0.0;
    for (var j = t; j < n_this; j = j + WG) {
      let e = exp(tile_s[j] - m_new);
      tile_s[j] = e;
      ts = ts + e;
    }
    part[t] = ts;
    workgroupBarrier();
    stride = WG / 2u;
    while (stride > 0u) {
      if (t < stride) { part[t] = part[t] + part[t + stride]; }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    if (t == 0u) { tile_sum = part[0]; }
    workgroupBarrier();

    // --- fold the tile into the running output ------------------------------
    // acc <- acc * corr + sum_j e_j * v[base+j]. One thread per output channel,
    // so nothing needs reducing here.
    if (t < hd) {
      var a = acc[t] * corr;
      for (var j = 0u; j < n_this; j = j + 1u) {
        a = a + tile_s[j] * v[((base + j) * d.n_kv_heads + kvh) * hd + t];
      }
      acc[t] = a;
    }
    l = l * corr + tile_sum;
    m = m_new;
    // tile_s is about to be overwritten by the next tile's scores, and the loop
    // above reads it, so the two must not overlap.
    workgroupBarrier();
    base = base + TILE;
  }

  // Normalize once at the end: the running `l` is the full softmax denominator.
  if (t < hd) {
    o[(qi * d.n_heads + h) * hd + t] = acc[t] / l;
  }
}
