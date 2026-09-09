"""
build_shards.py — export one ONNX shard per phone.

`export_phone_shard.py` carves a single layer range. This splits a range across
N devices and names the outputs web/shard0.onnx, shard1.onnx, ... plus a
web/swarm.json manifest the coordinator reads to know the topology.

Run:  python3 build_shards.py --start 20 --end 27 --peers 2
"""
import argparse, json, subprocess, sys, os

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--start", type=int, default=24)
p.add_argument("--end", type=int, default=27)
p.add_argument("--peers", type=int, default=1)
a = p.parse_args()

total = a.end - a.start + 1
if a.peers > total:
    sys.exit(f"can't split {total} layers across {a.peers} peers")

per, extra = divmod(total, a.peers)
ranges, cur = [], a.start
for i in range(a.peers):
    n = per + (1 if i < extra else 0)     # spread the remainder over the first peers
    ranges.append((cur, cur + n - 1))
    cur += n

print(f"splitting layers {a.start}-{a.end} across {a.peers} peer(s): {ranges}")
os.makedirs("web", exist_ok=True)
for i, (s, e) in enumerate(ranges):
    out = f"web/shard{i}.onnx"
    print(f"\n=== peer {i}: layers {s}-{e} -> {out} ===")
    r = subprocess.run([sys.executable, "export_phone_shard.py", "--model", a.model,
                        "--start", str(s), "--end", str(e), "--out", out])
    if r.returncode:
        sys.exit(f"export failed for peer {i}")

meta = json.load(open(f"web/shard{len(ranges)-1}.json"))
manifest = {"model": a.model, "n_total": meta["n_total"], "hidden": meta["hidden"],
            "kv_heads": meta["kv_heads"], "head_dim": meta["head_dim"],
            "kv_cache": meta["kv_cache"], "mac_layers": [0, a.start - 1],
            "peers": [{"slot": i, "start": s, "end": e} for i, (s, e) in enumerate(ranges)]}
json.dump(manifest, open("web/swarm.json", "w"), indent=2)
print(f"\nwrote web/swarm.json: Mac 0-{a.start-1}, peers {ranges}")
