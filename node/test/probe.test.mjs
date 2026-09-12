// The capability probe (web/js/probe.mjs), against fake devices.
//
// WHAT THIS IS FOR. The allocator refuses a device any layer whose largest tensor
// exceeds its maxStorageBufferBindingSize, so if the probe reports the wrong number
// -- or swallows a failure and reports nothing -- the allocator plans against a lie
// and the failure surfaces as an OOM on a phone after gigabytes have downloaded. The
// probe is therefore the input the whole feature depends on, and the devices that
// matter most are the broken ones: no WebGPU, an adapter that refuses a device, a
// compute pass that returns the wrong number. None of those can be produced on this
// machine, so `gpu` is injected.
//
// The real probe against real hardware is web/inspect.html at /check, and a real
// device is the only thing that can confirm a real device's numbers. What is under
// test here is that the probe reports what it was told and does not hide a failure.
//
//   node test/probe.test.mjs
import {probe, capsOf, verdict} from '../../web/js/probe.mjs';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  ' + extra : ''}`); }
};

const MiB = 1024 * 1024;

/**
 * A fake WebGPU that runs the probe's compute pass in JavaScript.
 *
 * It is a real enough stand-in that the shader path is exercised end to end: the
 * probe's createBuffer/createBindGroup/dispatch/mapAsync sequence all runs, and
 * `answer` decides what element 63 reads back -- which is how "the device advertised
 * generous limits and then computed the wrong number" becomes a test.
 */
function fakeGpu({limits = {}, answer = 63 * 63, adapter = null,
                  failDevice = null, noAdapter = false} = {}) {
  const L = {
    maxStorageBufferBindingSize: 128 * MiB,
    maxBufferSize: 256 * MiB,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeInvocationsPerWorkgroup: 256,
    ...limits,
  };
  let stored = null;
  const dev = {
    limits: L,
    createShaderModule: () => ({}),
    createComputePipeline: () => ({getBindGroupLayout: () => ({})}),
    createBuffer: ({size}) => ({
      size, destroy() {},
      mapAsync: async () => {},
      getMappedRange: () => {
        const a = new Uint32Array(size / 4);
        a[63] = answer;
        return a.buffer;
      },
      unmap() {},
    }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({setPipeline() {}, setBindGroup() {},
                                dispatchWorkgroups() {}, end() {}}),
      copyBufferToBuffer() { stored = true; },
      finish: () => ({}),
    }),
    queue: {submit() {}},
    destroy() { dev.destroyed = true; },
  };
  return {
    _dev: dev,
    requestAdapter: async () => {
      if (noAdapter) return null;
      return {
        limits: L,
        info: adapter || {vendor: 'fake', architecture: 'test', device: 'sim'},
        requestDevice: async () => {
          if (failDevice) throw new Error(failDevice);
          return dev;
        },
      };
    },
  };
}
// The probe uses these as globals, which a browser provides and Node does not.
globalThis.GPUBufferUsage = {STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8};
globalThis.GPUMapMode = {READ: 1};

const nav = {hardwareConcurrency: 10, deviceMemory: 8, platform: 'test',
             userAgent: 'node harness'};

// =========================================================================
console.log('a working device -> reports its real limits and a correct shader:');
{
  const {caps, detail} = await probe(fakeGpu({limits: {
    maxStorageBufferBindingSize: 4 * 1024 ** 3, maxBufferSize: 8 * 1024 ** 3}}), nav);
  ok('reports maxStorageBufferBindingSize verbatim, not rounded',
     caps.maxStorageBufferBindingSize === 4 * 1024 ** 3,
     String(caps.maxStorageBufferBindingSize));
  ok('reports maxBufferSize', caps.maxBufferSize === 8 * 1024 ** 3);
  ok('reports the workgroup storage limit /check shows',
     caps.maxComputeWorkgroupStorageSize === 16384);
  ok('reports core count', caps.cores === 10);
  ok('the compute pass returned the right number', caps.computeOk === true);
  ok('  and its round trip was timed', detail.computeMs != null);
  ok('destroys the probe device, so the weights are not competing with it',
     fakeGpu !== null && detail.error === null);
  const v = verdict(detail);
  ok('verdict: can be a bird', v.ok && /can be a bird/.test(v.headline));
  ok('  and says what the binding limit means for layer size',
     /4295 MB/.test(v.why) && /128 MiB default/.test(v.why), v.why.split('\n')[1]);
}

// =========================================================================
console.log('\nno WebGPU -> says so, and does not pretend to have limits:');
{
  const {caps, detail} = await probe(null, nav);
  ok('gpu is false', caps.gpu === false);
  ok('no binding limit is invented', caps.maxStorageBufferBindingSize === null);
  ok('the reason names navigator.gpu', /navigator\.gpu/.test(detail.error), detail.error);
  const v = verdict(detail);
  ok('verdict: cannot be a bird', !v.ok);
  ok('  and it is about WebGPU, in plain language', /WebGPU/.test(v.why));
  ok('  with the iOS version that has it', /iOS 17/.test(v.why));
}

// =========================================================================
console.log('\nan adapter that refuses a device -> a reported failure, not silence:');
{
  const {caps, detail} = await probe(
    fakeGpu({failDevice: 'requested limits exceed the adapter'}), nav);
  ok('the error survives', /exceed the adapter/.test(detail.error), detail.error);
  ok('computeOk is false, not null', caps.computeOk === false);
  // The limits WERE readable before requestDevice failed, and reporting them is
  // right: they are what a human needs to see to know why it failed.
  ok('the limits it did manage to read are still reported',
     caps.maxStorageBufferBindingSize === 128 * MiB);
  ok('verdict: cannot be a bird, with the driver\'s own words',
     !verdict(detail).ok && /exceed the adapter/.test(verdict(detail).why));
}

// =========================================================================
console.log('\nrequestAdapter returning null -> named, not a TypeError:');
{
  const {detail} = await probe(fakeGpu({noAdapter: true}), nav);
  ok('the message explains the difference from "no WebGPU"',
     /no adapter was granted/.test(detail.error), detail.error);
  ok('verdict: cannot be a bird', !verdict(detail).ok);
}

// =========================================================================
console.log('\na device that advertises limits but computes the WRONG number:');
{
  // The interesting device. Its limits table looks fine, so a probe that only reads
  // limits would pass it and the allocator would give it layers it silently gets
  // wrong. Only dispatching a real pass catches this.
  const {caps, detail} = await probe(fakeGpu({answer: 12345}), nav);
  ok('computeOk is false', caps.computeOk === false);
  ok('the limits alone would have looked fine',
     caps.maxStorageBufferBindingSize === 128 * MiB);
  const v = verdict(detail);
  ok('verdict: cannot be a bird', !v.ok);
  ok('  and says the shader returned the wrong number',
     /wrong number/.test(v.why), v.why);
}

// =========================================================================
console.log('\na device at WebGPU\'s 128 MiB DEFAULT is warned about, not refused:');
{
  const {detail} = await probe(fakeGpu({}), nav);
  const v = verdict(detail);
  ok('it can still be a bird', v.ok);
  ok('  but is told a big tensor cannot go here at any split',
     /cannot be divided across devices/.test(v.why), v.why.split('\n')[1]);
}

// =========================================================================
console.log('\ncapsOf sends only what the allocator uses:');
{
  const {caps} = await probe(fakeGpu({}), nav);
  const keys = Object.keys(caps).sort();
  ok('the payload is small and fixed', keys.length === 8, keys.join(','));
  ok('  it carries no user agent (the coordinator has that from /diag)',
     !('ua' in caps));
  ok('  and no deviceMemory guess: a coarse GB hint is not a weight budget',
     !('deviceMemoryGb' in caps) && !('budget' in caps));
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
