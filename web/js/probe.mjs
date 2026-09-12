// web/js/probe.mjs — measure what a device can actually do.
//
// ONE PROBE, TWO CALLERS, and that is the whole reason this is a module rather than
// inline in a page. web/inspect.html (served at /check) is what a human opens on a
// phone to find out whether it can be a bird; web/bird.html calls the same function
// before /join and sends the result as `caps`. The allocator refuses a device any
// layer whose largest tensor exceeds its maxStorageBufferBindingSize, so the number
// it plans against has to be the REAL one -- and two probes would eventually
// disagree about which that is. Whatever a human reads at /check is literally what
// the coordinator was told.
//
// Nothing here loads weights and nothing here needs the coordinator, so it is safe
// on a device nowhere near being able to run a layer -- which is exactly the device
// whose limits matter most.

/**
 * Measure this device. Returns {caps, detail}: `caps` is the small object the
 * coordinator gets, `detail` is everything, for the page.
 *
 * `gpu` and `nav` are injectable so a test can drive the failure branches -- no
 * WebGPU, an adapter that refuses a device, a shader that computes the wrong number
 * -- without owning such a device.
 */
export async function probe(gpu, nav) {
  if (gpu === undefined) gpu = typeof navigator !== 'undefined' ? navigator.gpu : null;
  if (nav === undefined) nav = typeof navigator !== 'undefined' ? navigator : {};
  const detail = {
    gpu: !!gpu,
    cores: nav.hardwareConcurrency || null,
    // deviceMemory is Chrome-only and coarse (a power-of-two GB hint). Reported
    // because it is the only memory number a browser offers at all, and deliberately
    // NOT used as the weight budget: guessing a budget from it is the kind of
    // inference that produced the iPad OOM this path exists to stop repeating.
    deviceMemoryGb: nav.deviceMemory || null,
    secureContext: typeof isSecureContext === 'boolean' ? isSecureContext : null,
    platform: nav.platform || null,
    ua: nav.userAgent || null,
    limits: null, adapter: null, computeOk: null, computeMs: null, error: null,
  };
  if (!gpu) {
    detail.error = 'no navigator.gpu: this browser has no WebGPU at all';
    return {caps: capsOf(detail), detail};
  }
  let dev = null;
  try {
    const ad = await gpu.requestAdapter();
    if (!ad) throw new Error('requestAdapter() returned null — WebGPU is present ' +
                             'but no adapter was granted');
    const L = ad.limits || {};
    detail.limits = {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize ?? null,
      maxBufferSize: L.maxBufferSize ?? null,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize ?? null,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup ?? null,
    };
    detail.adapter = {vendor: ad.info?.vendor || null,
                      architecture: ad.info?.architecture || null,
                      device: ad.info?.device || null};
    // Ask for the adapter's REAL limits, not the defaults. Overflowing
    // maxStorageBufferBindingSize is a SILENT wrong answer on this backend rather
    // than a throw (see kernels/lib.ts getDevice), so the number the allocator
    // plans against must be the one a device would actually be created with.
    dev = await ad.requestDevice({requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
    }});
    const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
    detail.computeOk = await runShader(dev);
    detail.computeMs = +((typeof performance !== 'undefined'
      ? performance.now() : 0) - t0).toFixed(1);
  } catch (e) {
    detail.error = String(e?.message || e);
    detail.computeOk = false;
  } finally {
    // Destroy it: the bird page creates its OWN device for the real work, and two
    // live devices on a phone is memory the weights are about to need.
    try { dev?.destroy?.(); } catch {}
  }
  return {caps: capsOf(detail), detail};
}

/** The small object /join carries: only what the allocator uses, plus enough
 *  identity to make a /status row readable by a human. */
export function capsOf(d) {
  return {
    gpu: d.gpu,
    maxStorageBufferBindingSize: d.limits?.maxStorageBufferBindingSize ?? null,
    maxBufferSize: d.limits?.maxBufferSize ?? null,
    maxComputeWorkgroupStorageSize: d.limits?.maxComputeWorkgroupStorageSize ?? null,
    cores: d.cores,
    vendor: d.adapter?.vendor ?? null,
    secureContext: d.secureContext,
    computeOk: d.computeOk,
  };
}

/**
 * Dispatch one real compute pass and check the number it returns.
 *
 * 64 invocations each write i*i, and element 63 must read back 3969. A limits table
 * proves nothing about whether a shader RUNS here -- the interesting devices are the
 * ones that advertise generous limits and then fail to create a pipeline.
 */
async function runShader(dev) {
  const N = 64;
  const mod = dev.createShaderModule({code: `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < ${N}u) { out[gid.x] = gid.x * gid.x; }
}`});
  const pipe = dev.createComputePipeline({layout: 'auto',
    compute: {module: mod, entryPoint: 'main'}});
  const buf = dev.createBuffer({size: N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const read = dev.createBuffer({size: N * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const bind = dev.createBindGroup({layout: pipe.getBindGroupLayout(0),
    entries: [{binding: 0, resource: {buffer: buf}}]});
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(buf, 0, read, 0, N * 4);
  dev.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const got = new Uint32Array(read.getMappedRange().slice(0))[N - 1];
  read.unmap();
  buf.destroy(); read.destroy();
  return got === (N - 1) * (N - 1);
}

/**
 * Can this device be a bird, and if not, why in one sentence?
 *
 * Deliberately does NOT decide how many layers it gets: that is the allocator's job
 * on the coordinator, which is the only side that knows the model. This answers only
 * the question a human standing in front of the device is asking.
 */
export function verdict(d) {
  const mb = n => n == null ? 'unknown' : `${(n / 1e6).toFixed(0)} MB`;
  if (!d.gpu) {
    return {ok: false, headline: 'cannot be a bird',
            why: 'This browser has no WebGPU. A bird runs transformer layers as ' +
                 'WebGPU compute shaders, so there is nothing to fall back to.\n' +
                 'On iOS, WebGPU needs iOS 17 or newer.'};
  }
  if (d.error || d.computeOk === false) {
    return {ok: false, headline: 'cannot be a bird',
            why: `WebGPU is present but did not work here.\n${d.error ||
                  'a real compute shader returned the wrong number'}`};
  }
  const bind = d.limits?.maxStorageBufferBindingSize || 0;
  const DEFAULT = 128 * 1024 * 1024;
  return {ok: true, headline: 'can be a bird',
    why: `Largest single tensor this device can hold: ${mb(bind)}.\n` +
      (bind <= DEFAULT
        ? "That is WebGPU's default binding limit. A layer whose biggest tensor is " +
          'larger than this cannot go on this device at any split — one tensor ' +
          'cannot be divided across devices.'
        : "Comfortably above WebGPU's 128 MiB default, so per-tensor size is " +
          'unlikely to be what stops this device.') +
      '\nThe coordinator decides how many layers that buys; it is the only side ' +
      'that knows the model.'};
}
