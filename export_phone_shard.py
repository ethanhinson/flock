"""
export_phone_shard.py — carve N transformer layers out of a model and write an
ONNX file the PHONE can run in its browser via ONNX Runtime Web + WebGPU.

Why ONNX and not MLX: MLX is Apple-native and can't run in Safari. ONNX Runtime
Web ships a WebGPU backend, so the phone's GPU executes the same math your Mac
does -- without us hand-writing a single WGSL kernel (which is the part of
swarmllm.ai that costs thousands of lines).

Run:  python3 export_phone_shard.py --start 24 --end 27
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoConfig

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--start", type=int, default=24)
p.add_argument("--end", type=int, default=27)
p.add_argument("--out", default="web/phone_shard.onnx")
args = p.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
print(f"loading {args.model} (cpu, float32) ...")
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
model.eval()
cfg = model.config
layers = model.model.layers[args.start : args.end + 1]
is_last = args.end == cfg.num_hidden_layers - 1
print(f"carving layers {args.start}-{args.end} of {cfg.num_hidden_layers} (is_last={is_last})")


class PhoneShard(torch.nn.Module):
    """Stateless slice: hidden state in, hidden state out.

    No KV cache. Every step re-runs the full sequence, which is wasteful but
    keeps the phone side tiny and dependency-free. Real systems cache; this
    trades speed for ~150 lines of JS you can actually read.
    """
    def __init__(self, layers, rotary, norm, head, is_last, head_dim):
        super().__init__()
        self.layers, self.rotary = layers, rotary
        self.norm, self.head, self.is_last = norm, head, is_last
        self.head_dim = head_dim

    def forward(self, hidden, position_ids):
        pos = self.rotary(hidden, position_ids)
        seq = hidden.shape[1]
        # causal mask so token i can't see i+1
        m = torch.full((seq, seq), torch.finfo(hidden.dtype).min)
        m = torch.triu(m, diagonal=1)[None, None, :, :]
        for lyr in self.layers:
            out = lyr(hidden, attention_mask=m, position_ids=position_ids,
                      position_embeddings=pos)
            hidden = out[0] if isinstance(out, tuple) else out
        # NOTE: we deliberately stop before the vocab projection. lm_head is
        # 151936x1024 (~600MB fp32) and is TIED to the embedding matrix the Mac
        # already holds -- shipping it to the phone would quadruple the download
        # to buy nothing. The phone returns a hidden state; the Mac projects it.
        return self.norm(hidden) if self.is_last else hidden


shard = PhoneShard(layers, model.model.rotary_emb, model.model.norm,
                   None, is_last, cfg.head_dim).eval()

H = cfg.hidden_size
dummy_h = torch.randn(1, 4, H)
dummy_p = torch.arange(4)[None, :]

with torch.no_grad():
    ref = shard(dummy_h, dummy_p)
print("torch output:", tuple(ref.shape))

torch.onnx.export(
    shard, (dummy_h, dummy_p), args.out,
    input_names=["hidden", "position_ids"],
    output_names=["output"],
    dynamic_axes={"hidden": {1: "seq"}, "position_ids": {1: "seq"},
                  "output": {1: "seq"}},
    opset_version=17, do_constant_folding=True,
)

meta = {"start": args.start, "end": args.end, "hidden": H, "needs_lm_head": is_last,
        "is_last": is_last, "n_total": cfg.num_hidden_layers,
        "model": args.model}
json.dump(meta, open("web/phone_shard.json", "w"), indent=2)

mb = (os.path.getsize(args.out) +
      (os.path.getsize(args.out + ".data") if os.path.exists(args.out + ".data") else 0)) / 1e6
print(f"wrote {args.out}  ({mb:.0f} MB total -> this is your phone download)")
print(f"wrote web/phone_shard.json  {meta}")

# --- verify the ONNX graph matches torch, so the phone can't be silently wrong
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
got = sess.run(None, {"hidden": dummy_h.numpy(), "position_ids": dummy_p.numpy().astype(np.int64)})[0]
err = float(np.abs(got - ref.numpy()).max())
print(f"onnx vs torch max abs err: {err:.2e}  {'OK' if err < 1e-3 else 'FAIL'}")
