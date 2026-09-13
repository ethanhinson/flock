// Sum N partial-sum vectors into one. This is the reduction half of a
// COLUMN-WISE (input-split) sharded matvec.
//
// Column-wise sharding gives shard s the columns [c0_s, c1_s) of W and the
// matching slice of x, so shard s computes a PARTIAL y over its own columns and
// the true y is the sum of the N partials. Row-wise sharding needs no kernel at
// all -- the output slices concatenate -- which is exactly the asymmetry that
// makes row-wise the cheaper split and is measured in bench_shard.ts.
//
// LAYOUT. The partials arrive as one buffer of n_shards * n_rows f32, shard s at
// offset s * n_rows. One contiguous buffer rather than N bindings for two
// reasons: WebGPU's maxStorageBuffersPerShaderStage is 8 on the default limits,
// so N bindings caps N at 6, and a coordinator that gathers partials off the
// wire is writing into one staging buffer anyway. Within a single device the
// partials are written straight there by the matvec, at a per-shard byte offset,
// so the "gather" is free.
//
// SUMMATION ORDER IS THE CONTRACT. The loop below adds shard 0, then shard 1,
// ... in index order, one f32 add each, with no tree and no reassociation. That
// makes the result reproducible and, more importantly, makes it something a CPU
// reference can model EXACTLY: shardReduceRef in shard_ref.ts is the same loop
// with Math.fround. A tree reduction here would be marginally more accurate and
// would still be modellable, but it would also mean the answer depended on
// n_shards in a way the serial order does not -- the serial order's partial sums
// for the first k shards are the same whether there are k shards or k+1.
//
// This is why a column-wise shard is NOT bit-identical to the unsharded kernel
// and cannot be made so: the unsharded kernel's 64 lanes each accumulate over
// blocks {lane, lane+64, ...} spanning the WHOLE row and then tree-reduce, while
// N column shards partition those blocks N ways first. Different f32 addition
// order, therefore a different f32 answer. The claim the tests make instead is
// that the sharded GPU result matches an FMA-modelled strict-f32 CPU reference
// of the SHARDED order bit-for-bit, which is a strictly stronger statement than
// any tolerance against the unsharded result.

struct Dims {
  n_rows: u32,      // length of each partial vector
  n_shards: u32,    // how many partials to sum
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       parts: array<f32>;   // [n_shards][n_rows]
@group(0) @binding(1) var<storage, read_write> o:     array<f32>;   // [n_rows]
@group(0) @binding(2) var<uniform>             d:     Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let row = g.x;
  if (row >= d.n_rows) { return; }
  // Serial, in shard order. See the note above: this order is the contract.
  var acc = parts[row];
  for (var s = 1u; s < d.n_shards; s = s + 1u) {
    acc = acc + parts[s * d.n_rows + row];
  }
  o[row] = acc;
}
