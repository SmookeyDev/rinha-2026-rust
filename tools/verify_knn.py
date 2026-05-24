"""Run brute-force k-NN k=5 (FAISS) and check that the normalization matches
the official test-data.json labels bit for bit.
"""
import argparse
import numpy as np
import orjson
import faiss

from normalize import normalize, round4

ap = argparse.ArgumentParser()
ap.add_argument("refs_npy")
ap.add_argument("lbls_npy")
ap.add_argument("test_data_json", help="upstream/test/test-data.json")
args = ap.parse_args()

refs = np.load(args.refs_npy)
lbls = np.load(args.lbls_npy)

with open(args.test_data_json, "rb") as f:
    entries = orjson.loads(f.read())["entries"]

queries = np.empty((len(entries), 14), dtype=np.float32)
expected = np.empty(len(entries), dtype=bool)
for i, e in enumerate(entries):
    queries[i] = [round4(x) for x in normalize(e["request"])]
    expected[i] = e["expected_approved"]

index = faiss.IndexFlatL2(14)
index.add(refs)
_, I = index.search(queries, 5)
approved = (lbls[I].sum(axis=1) / 5.0) < 0.6

matches = (approved == expected).sum()
total = len(entries)
print(f"accuracy: {matches}/{total} = {100*matches/total:.4f}%")
print(f"mismatches: {total - matches}")
