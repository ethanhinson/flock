"""
flock_export_coordinator.py — export the COORDINATOR's half to ONNX.

The coordinator holds the embedding, layers 0..cut-1, and the tied vocab
projection. Exporting it lets the whole flock run on Node, so every link in the
chain can be a WebRTC data channel instead of a websocket.

Three graphs, because they run at different points in a lap:
    embed.onnx   token ids   -> hidden state      (start of the lap)
    layers.onnx  hidden      -> hidden (+ K/V)    (our slice)
    head.onnx    hidden      -> next token id     (end of the lap)

This file (and flock_export.py) is the ONLY remaining Python: PyTorch stays the
reference implementation that every ONNX graph is checked against, so a bad
export can't be silently wrong.

Run:  python3 flock_export_coordinator.py --cut 24
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--cut", type=int, default=24, help="coordinator owns layers 0..cut-1")
p.add_argument("--outdir", default="web/coord")
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
print(f"loading {args.model} ...")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
cfg = model.config
H, KVH, HD = cfg.hidden_size, cfg.num_key_value_heads, cfg.head_dim
layers = model.model.layers[:args.cut]
n_my = len(layers)
print(f"coordinator: embedding + layers 0-{args.cut-1} + vocab projection")


def save(m, path, args_, in_names, out_names, dyn):
    torch.onnx.export(m, args_, path, input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, opset_version=17,
                      do_constant_folding=True, dynamo=False)
    import onnx
    g = onnx.load(path)
    onnx.save(g, path, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(path) + ".data", size_threshold=1024)


class Embed(torch.nn.Module):
    def __init__(self, e): super().__init__(); self.e = e
    def forward(self, ids): return self.e(ids)


class Head(torch.nn.Module):
    """Final norm + tied vocab projection -> argmax token id."""
    def __init__(self, norm, head): super().__init__(); self.norm, self.head = norm, head
    def forward(self, hidden):
        return self.head(hidden).argmax(-1)


class OneLayerCache:
    """Same ONNX-friendly cache contract flock_export.py uses."""
    def __init__(self, pk, pv):
        self.past_k, self.past_v = pk, pv
        self.new_k = self.new_v = None

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if self.past_k is not None and self.past_k.shape[2] > 0:
            k = torch.cat([self.past_k, k], dim=2)
            v = torch.cat([self.past_v, v], dim=2)
        self.new_k, self.new_v = k, v
        return k, v


class Layers(torch.nn.Module):
    def __init__(self, layers, rotary):
        super().__init__(); self.layers, self.rotary = layers, rotary

    def forward(self, hidden, position_ids, *past):
        pos = self.rotary(hidden, position_ids)
        q = hidden.shape[1]
        kv = q + (past[0].shape[2] if past else 0)
        m = torch.full((q, kv), torch.finfo(hidden.dtype).min)
        m = torch.triu(m, diagonal=1 + (kv - q))[None, None]
        outs = []
        for i, lyr in enumerate(self.layers):
            c = OneLayerCache(past[2 * i], past[2 * i + 1])
            out = lyr(hidden, attention_mask=m, position_ids=position_ids,
                      past_key_values=c, use_cache=True, position_embeddings=pos)
            hidden = out[0] if isinstance(out, tuple) else out
            outs += [c.new_k, c.new_v]
        return (hidden, *outs)


# --- embed ---------------------------------------------------------------
ids = torch.tensor([[1, 2, 3, 4]])
save(Embed(model.model.embed_tokens), f"{args.outdir}/embed.onnx", (ids,),
     ["ids"], ["hidden"], {"ids": {1: "seq"}, "hidden": {1: "seq"}})
print("wrote embed.onnx")

# --- layers (traced at decode shape, with a NON-EMPTY past) ---------------
PAST = 3
dh = torch.randn(1, 1, H)
dp = torch.tensor([[PAST]])
dpast = tuple(torch.randn(1, KVH, PAST, HD) for _ in range(2 * n_my))
lay = Layers(layers, model.model.rotary_emb).eval()
in_names = ["hidden", "position_ids"]
out_names = ["output"]
dyn = {"hidden": {1: "seq"}, "position_ids": {1: "seq"}, "output": {1: "seq"}}
for i in range(n_my):
    for nm in (f"past_k{i}", f"past_v{i}"):
        in_names.append(nm); dyn[nm] = {2: "past_seq"}
    for nm in (f"new_k{i}", f"new_v{i}"):
        out_names.append(nm); dyn[nm] = {2: "total_seq"}
with torch.no_grad():
    ref = lay(dh, dp, *dpast)
save(lay, f"{args.outdir}/layers.onnx", (dh, dp, *dpast), in_names, out_names, dyn)
print(f"wrote layers.onnx ({n_my} layers)")

# --- head ----------------------------------------------------------------
hd = torch.randn(1, 1, H)
head = Head(model.model.norm, model.lm_head).eval()
with torch.no_grad():
    ref_tok = head(hd)
save(head, f"{args.outdir}/head.onnx", (hd,), ["hidden"], ["token"],
     {"hidden": {1: "seq"}, "token": {1: "seq"}})
print("wrote head.onnx")

json.dump({"model": args.model, "cut": args.cut, "hidden": H, "kv_heads": KVH,
           "head_dim": HD, "n_layers": n_my, "n_total": cfg.num_hidden_layers,
           "eos": cfg.eos_token_id if isinstance(cfg.eos_token_id, int)
                  else (cfg.eos_token_id or [151645])[0]},
          open(f"{args.outdir}/coord.json", "w"), indent=2)

# --- verify every graph against PyTorch ----------------------------------
import onnxruntime as ort
print("\nverifying against PyTorch:")
s = ort.InferenceSession(f"{args.outdir}/layers.onnx", providers=["CPUExecutionProvider"])
feed = {"hidden": dh.numpy(), "position_ids": dp.numpy().astype(np.int64)}
for i in range(n_my):
    feed[f"past_k{i}"] = dpast[2 * i].numpy()
    feed[f"past_v{i}"] = dpast[2 * i + 1].numpy()
err = float(np.abs(s.run(None, feed)[0] - ref[0].numpy()).max())
print(f"  layers  max abs err {err:.2e}  {'OK' if err < 1e-3 else 'FAIL'}")

s2 = ort.InferenceSession(f"{args.outdir}/head.onnx", providers=["CPUExecutionProvider"])
got = s2.run(None, {"hidden": hd.numpy()})[0]
print(f"  head    argmax match {'OK' if (got == ref_tok.numpy()).all() else 'FAIL'}")

tot = sum(os.path.getsize(os.path.join(args.outdir, f))
          for f in os.listdir(args.outdir)) / 1e6
print(f"\ncoordinator ONNX total: {tot:.0f} MB (stays on this machine)")
