// Argmax over the logit vector, as a two-stage GPU reduction.
//
// Greedy decode needs ONE number out of 151936: the index of the largest logit.
// The alternative is reading the whole vector back and scanning it in JS, and the
// reason not to is measured, not aesthetic -- a map readback on this backend is
// ~24 ms regardless of size, and 151936 floats is 608 KB to copy on top of that,
// while the reduction below is two dispatches of ~20 us each and returns 8 bytes.
// bench_head.ts times both.
//
// Stage 1 (`pass1`): each of `n_groups` workgroups reduces its slice to one
// (value, index) pair. Stage 2 (`pass2`): one workgroup reduces those pairs.
// Two stages rather than one because a reduction across workgroups needs a
// dispatch boundary to synchronize -- WebGPU has no device-wide barrier inside a
// pass, and atomics on a f32 max would need a bitcast trick that buys nothing
// here.
//
// TIE-BREAKING IS PART OF THE CONTRACT. numpy/torch argmax return the LOWEST
// index among equal maxima, and a divergence test against ONNX greedy decode is
// worthless if the two disagree on ties. So every comparison here is strictly
// "greater than, or equal with a smaller index", and the reduction is a tree over
// a fixed layout, which makes the result deterministic rather than merely
// usually-right. Real logit ties are rare but they are not hypothetical: early in
// a prompt several vocab entries can land on the same f32.
//
// Values are carried as f32 in one array and indices as u32 in another, rather
// than as a packed struct, because WGSL workgroup arrays of structs pad to
// 16 bytes per element on some backends and 2 x 4 bytes is the layout that
// actually fits.
//
// Stage 2 writes to DIFFERENT buffers than it reads, even though reducing 594
// partials in place would be the obvious thing. WebGPU forbids binding one buffer
// as both `read` and `read_write` in a single dispatch -- it is a validation
// error, not a data race to be careful about -- so the host ping-pongs: stage 1
// writes (pval, pidx), stage 2 reads those as (x, xidx) and writes (oval, oidx).
// The cost is 8 bytes of extra buffer.

struct Dims {
  n: u32,          // how many values to reduce
  n_groups: u32,   // workgroups in stage 1 (== the length of the stage-2 input)
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       x:    array<f32>;   // logits (pass1) / partial values (pass2)
@group(0) @binding(1) var<storage, read>       xidx: array<u32>;   // pass2 only: the indices those partials came from
@group(0) @binding(2) var<storage, read_write> oval: array<f32>;   // best value out
@group(0) @binding(3) var<storage, read_write> oidx: array<u32>;   // best index out
@group(0) @binding(4) var<uniform>             d:    Dims;

const WG: u32 = 256u;

var<workgroup> wval: array<f32, WG>;
var<workgroup> widx: array<u32, WG>;

/// True if (va, ia) beats (vb, ib): strictly larger, or equal with a lower index.
/// The index half is what makes ties match numpy/torch rather than being racy.
fn better(va: f32, ia: u32, vb: f32, ib: u32) -> bool {
  return va > vb || (va == vb && ia < ib);
}

/// Tree-reduce this workgroup's (wval, widx) into slot 0. Every thread must
/// reach every barrier, so the guard is inside the `if (t < stride)` and not
/// around the loop.
fn reduce_wg(t: u32) {
  var stride = WG / 2u;
  while (stride > 0u) {
    if (t < stride) {
      if (better(wval[t + stride], widx[t + stride], wval[t], widx[t])) {
        wval[t] = wval[t + stride];
        widx[t] = widx[t + stride];
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
}

// Stage 1: n_groups workgroups, each reducing a strided slice of x.
//
// The slice is strided (thread t of group g takes elements g*WG + t, then
// + n_groups*WG, ...) rather than contiguous so the loads coalesce: adjacent
// threads read adjacent addresses on every iteration.
@compute @workgroup_size(256)
fn pass1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let stride = d.n_groups * WG;

  var bv = -3.4028235e38;
  var bi = 0xFFFFFFFFu;                  // sentinel: loses every `better` test
  for (var i = wg.x * WG + t; i < d.n; i = i + stride) {
    if (better(x[i], i, bv, bi)) { bv = x[i]; bi = i; }
  }
  // `xidx` is unused by this stage but is read once so that pass1 and pass2
  // derive the SAME `layout: "auto"` bind group layout and the host can build one
  // bind group per pass shape instead of two. `select` keeps it a no-op: the
  // condition is false for every real index, and WGSL evaluates both arms of
  // select without branching, so the load is one coalesced word per thread.
  bi = select(bi, xidx[0], bi == 0xFFFFFFFEu);
  wval[t] = bv;
  widx[t] = bi;
  workgroupBarrier();
  reduce_wg(t);
  if (t == 0u) {
    oval[wg.x] = wval[0];
    oidx[wg.x] = widx[0];
  }
}

// Stage 2: one workgroup over the n_groups partials. `x` is bound to the stage-1
// value buffer and the ORIGINAL logit indices ride along in `xidx`, so the final
// answer indexes the logits and not the partials.
@compute @workgroup_size(256)
fn pass2(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var bv = -3.4028235e38;
  var bi = 0xFFFFFFFFu;
  for (var i = t; i < d.n_groups; i = i + WG) {
    if (better(x[i], xidx[i], bv, bi)) { bv = x[i]; bi = xidx[i]; }
  }
  wval[t] = bv;
  widx[t] = bi;
  workgroupBarrier();
  reduce_wg(t);
  if (t == 0u) {
    oval[0] = wval[0];
    oidx[0] = widx[0];
  }
}
