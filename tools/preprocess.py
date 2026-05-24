"""Convert references.json (decompressed) into references_vec.npy + references_lbl.npy."""
import argparse
import os
import numpy as np
import orjson

ap = argparse.ArgumentParser()
ap.add_argument("refs_json", help="decompressed references.json")
ap.add_argument("--out-dir", default=".")
args = ap.parse_args()

out_vec = os.path.join(args.out_dir, "references_vec.npy")
out_lbl = os.path.join(args.out_dir, "references_lbl.npy")

with open(args.refs_json, "rb") as f:
    data = orjson.loads(f.read())
print(f"loaded {len(data)} records")

vecs = np.empty((len(data), 14), dtype=np.float32)
lbls = np.empty(len(data), dtype=np.uint8)
for i, r in enumerate(data):
    vecs[i] = r["vector"]
    lbls[i] = 1 if r["label"] == "fraud" else 0

np.save(out_vec, vecs)
np.save(out_lbl, lbls)
print(f"fraud rate: {lbls.mean():.4f}")
print(f"wrote {out_vec} + {out_lbl}")
