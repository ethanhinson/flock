"""
export_phone_shard.py — carve N transformer layers out of a model and write an
ONNX file the PHONE can run in its browser via ONNX Runtime Web + WebGPU.

Why ONNX and not MLX: MLX is Apple-native and can't run in Safari. ONNX Runtime
Web ships a WebGPU backend, so the phone's GPU executes the same math your Mac
does -- without us hand-writing a single WGSL kernel (which is the part of
swarmllm.ai that costs thousands of lines).

KV CACHE (--cache, default on):
An ONNX graph is static, so a cache can't live inside it as hidden state. We
make it explicit: past K/V come in as inputs, new K/V go out as outputs, and
the caller (the phone JS) holds them between steps. That turns each decode step
from "re-run the whole sequence" into "run one token", which is the difference
between O(n^2) and O(n) work over a generation.

Run:  python3 export_phone_shard.py --start 24 --end 27
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--start", type=int, default=24)
p.add_argument("--end", type=int, default=27)
p.add_argument("--out", default="web/phone_shard.onnx")
p.add_argument("--no-cache", dest="cache", action="store_false",
               help="export the simple stateless graph (slower, easier to read)")
args = p.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
print(f"loading {args.model} (cpu, float32) ...")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
cfg = model.config
layers = model.model.layers[args.start : args.end + 1]
is_last = args.end == cfg.num_hidden_layers - 1
n_my = len(layers)
print(f"carving layers {args.start}-{args.end} of {cfg.num_hidden_layers} "
      f"(is_last={is_last}, kv_cache={args.cache})")


class OneLayerCache:
    """Minimal stand-in for HF's DynamicCache, for ONE layer.

    Qwen3Attention calls `past_key_values.update(k, v, layer_idx)` and expects
    the full (past + new) tensors back. That's the entire contract, so we can
    satisfy it with plain tensor concatenation -- which traces cleanly to ONNX,
    unlike HF's cache classes.
    """
    def __init__(self, past_k, past_v):
        self.past_k, self.past_v = past_k, past_v
        self.new_k = self.new_v = None

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if self.past_k is not None and self.past_k.shape[2] > 0:
            k = torch.cat([self.past_k, k], dim=2)
            v = torch.cat([self.past_v, v], dim=2)
        self.new_k, self.new_v = k, v
        return k, v


class PhoneShard(torch.nn.Module):
    """Stateless slice: hidden state in, hidden state out.

    With --cache the past/new K/V tensors ride along as explicit graph inputs
    and outputs, so the phone keeps the conversation state for its own layers.
    Nobody holds the whole model's cache -- it's sharded exactly like the
    weights are.
    """
    def __init__(self, layers, rotary, norm, is_last, use_cache):
        super().__init__()
        self.layers, self.rotary, self.norm = layers, rotary, norm
        self.is_last, self.use_cache = is_last, use_cache

    def forward(self, hidden, position_ids, *past):
        pos = self.rotary(hidden, position_ids)
        q_len = hidden.shape[1]
        kv_len = q_len + (past[0].shape[2] if self.use_cache and past else 0)

        # Causal mask over (query x key): with a cache, q_len is 1 and every
        # past key is visible, so the mask is all-zeros; during prefill it's
        # the usual upper-triangular -inf.
        m = torch.full((q_len, kv_len), torch.finfo(hidden.dtype).min)
        m = torch.triu(m, diagonal=1 + (kv_len - q_len))[None, None]

        outs = []
        for i, lyr in enumerate(self.layers):
            if self.use_cache:
                c = OneLayerCache(past[2 * i], past[2 * i + 1])
                out = lyr(hidden, attention_mask=m, position_ids=position_ids,
                          past_key_values=c, use_cache=True, position_embeddings=pos)
                outs += [c.new_k, c.new_v]
            else:
                out = lyr(hidden, attention_mask=m, position_ids=position_ids,
                          position_embeddings=pos)
            hidden = out[0] if isinstance(out, tuple) else out

        # NOTE: we deliberately stop before the vocab projection. lm_head is
        # 151936x1024 (~600MB fp32) and is TIED to the embedding matrix the Mac
        # already holds -- shipping it to the phone would quadruple the download
        # to buy nothing. The phone returns a hidden state; the Mac projects it.
        hidden = self.norm(hidden) if self.is_last else hidden
        return (hidden, *outs) if self.use_cache else hidden


shard = PhoneShard(layers, model.model.rotary_emb, model.model.norm,
                   is_last, args.cache).eval()

H, KVH, HD = cfg.hidden_size, cfg.num_key_value_heads, cfg.head_dim
# Trace the DECODE shape: 1 new token attending to some existing past. Tracing
# with an empty past would let the exporter fold the concat away and drop the
# cache inputs from the graph entirely.
PAST = 3
dummy_h = torch.randn(1, 1 if args.cache else 4, H)
dummy_p = torch.tensor([[PAST]]) if args.cache else torch.arange(4)[None, :]
dummy_past = tuple(torch.randn(1, KVH, PAST, HD) for _ in range(2 * n_my)) if args.cache else ()

in_names = ["hidden", "position_ids"]
out_names = ["output"]
dyn = {"hidden": {1: "seq"}, "position_ids": {1: "seq"}, "output": {1: "seq"}}
if args.cache:
    for i in range(n_my):
        for nm, tag in ((f"past_k{i}", "past"), (f"past_v{i}", "past")):
            in_names.append(nm); dyn[nm] = {2: "past_seq"}
        for nm in (f"new_k{i}", f"new_v{i}"):
            out_names.append(nm); dyn[nm] = {2: "total_seq"}

with torch.no_grad():
    ref = shard(dummy_h, dummy_p, *dummy_past)
ref_h = ref[0] if isinstance(ref, tuple) else ref
print("torch output:", tuple(ref_h.shape), f"(+{len(out_names)-1} cache tensors)")

torch.onnx.export(
    shard, (dummy_h, dummy_p, *dummy_past), args.out,
    input_names=in_names, output_names=out_names, dynamic_axes=dyn,
    opset_version=17, do_constant_folding=True, dynamo=False,
)

meta = {"start": args.start, "end": args.end, "hidden": H, "n_layers": n_my,
        "kv_heads": KVH, "head_dim": HD, "kv_cache": args.cache,
        "needs_lm_head": is_last, "is_last": is_last,
        "n_total": cfg.num_hidden_layers, "model": args.model}
json.dump(meta, open("web/phone_shard.json", "w"), indent=2)

# The legacy tracer embeds weights inline; split them into the .data sidecar so
# the graph file stays small and the phone can stream weights separately.
import onnx as _onnx
_m = _onnx.load(args.out)
_onnx.save(_m, args.out, save_as_external_data=True, all_tensors_to_one_file=True,
           location=os.path.basename(args.out) + ".data", size_threshold=1024)

mb = (os.path.getsize(args.out) +
      (os.path.getsize(args.out + ".data") if os.path.exists(args.out + ".data") else 0)) / 1e6
print(f"wrote {args.out}  ({mb:.0f} MB total -> this is your phone download)")
print(f"wrote web/phone_shard.json  {meta}")

# --- verify the ONNX graph matches torch, so the phone can't be silently wrong
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
feed = {"hidden": dummy_h.numpy(), "position_ids": dummy_p.numpy().astype(np.int64)}
if args.cache:
    for i in range(n_my):
        feed[f"past_k{i}"] = dummy_past[2 * i].numpy()
        feed[f"past_v{i}"] = dummy_past[2 * i + 1].numpy()
got = sess.run(None, feed)[0]
err = float(np.abs(got - ref_h.numpy()).max())
print(f"onnx vs torch max abs err: {err:.2e}  {'OK' if err < 1e-3 else 'FAIL'}")
