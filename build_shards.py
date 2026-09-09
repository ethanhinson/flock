"""
build_shards.py — export one ONNX shard per bird in the flock.

`flock_export.py` carves a single layer range. This splits a range across
N devices and names the outputs web/shard0.onnx, shard1.onnx, ... plus a
web/flock.json manifest the coordinator reads to know the topology.

Run:  python3 build_shards.py --start 20 --end 27 --birds 2
"""
import argparse, json, subprocess, sys, os

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--start", type=int, default=24)
p.add_argument("--end", type=int, default=27)
p.add_argument("--birds", type=int, default=1)
a = p.parse_args()

total = a.end - a.start + 1
if a.birds > total:
    sys.exit(f"can't split {total} layers across {a.birds} birds")

per, extra = divmod(total, a.birds)
ranges, cur = [], a.start
for i in range(a.birds):
    n = per + (1 if i < extra else 0)     # spread the remainder over the first birds
    ranges.append((cur, cur + n - 1))
    cur += n

print(f"splitting layers {a.start}-{a.end} across {a.birds} bird(s): {ranges}")
os.makedirs("web", exist_ok=True)
for i, (s, e) in enumerate(ranges):
    out = f"web/shard{i}.onnx"
    print(f"\n=== bird {i}: layers {s}-{e} -> {out} ===")
    r = subprocess.run([sys.executable, "flock_export.py", "--model", a.model,
                        "--start", str(s), "--end", str(e), "--out", out])
    if r.returncode:
        sys.exit(f"export failed for peer {i}")

meta = json.load(open(f"web/shard{len(ranges)-1}.json"))
manifest = {"model": a.model, "n_total": meta["n_total"], "hidden": meta["hidden"],
            "kv_heads": meta["kv_heads"], "head_dim": meta["head_dim"],
            "kv_cache": meta["kv_cache"], "coord_layers": [0, a.start - 1],
            "birds": [{"slot": i, "start": s, "end": e} for i, (s, e) in enumerate(ranges)]}
json.dump(manifest, open("web/flock.json", "w"), indent=2)
print(f"\nwrote web/flock.json: coordinator 0-{a.start-1}, birds {ranges}")
