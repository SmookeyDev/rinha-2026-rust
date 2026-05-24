"""Build the IVF index (ivf_int16.bin) consumed by the Rust runtime.

Pipeline:
  references.npy + labels.npy  ->  ivf_int16.bin

FAISS trains an IVF with nlist=4096 clusters over the 3M references and
exports everything as a custom binary blob, with vectors quantised to int16
(scale=10000) and laid out in SoA panels of 8 vectors for AVX2.

Binary layout (little-endian):
    [0..32)               header
        u32 magic     0x484E4952
        u32 version   3
        u32 n_total
        u32 dim       (14)
        u32 n_clusters
        u32 scale     (10000)
        u32 _pad[2]
    centroids[K*dim]      f32 row-major
    cluster_sizes[K]      u32
    cluster_offsets[K+1]  u32   prefix sum in vector units
    panel_offsets[K+1]    u32   prefix sum in i16 units
    vectors[N*dim]        i16   SoA panels of 8 + AoS tail per cluster
    labels[N]             u8    0=legit, 1=fraud
"""
import argparse
import os
import struct
import numpy as np
import faiss

DIM = 14
SCALE = 10000


def build(refs_npy: str, lbls_npy: str, out_bin: str, nlist: int = 4096) -> None:
    refs = np.load(refs_npy).astype(np.float32)
    lbls = np.load(lbls_npy)
    n_total = refs.shape[0]
    assert refs.shape[1] == DIM

    # Pre-quantise so the offline training matches the runtime distances.
    refs_int16 = np.round(refs * SCALE).astype(np.int32)
    refs_q = refs_int16.astype(np.float32) / SCALE

    print(f"training IVF nlist={nlist} on {n_total} vectors")
    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, nlist)
    index.train(refs_q[: min(500_000, n_total)])
    index.add(refs_q)

    centroids = index.quantizer.reconstruct_n(0, nlist).astype(np.float32)
    inv = index.invlists

    # Reorder vectors so each cluster occupies a contiguous range.
    cluster_sizes = np.zeros(nlist, dtype=np.uint32)
    cluster_offsets = np.zeros(nlist + 1, dtype=np.uint32)
    ordered_ids = np.empty(n_total, dtype=np.int64)
    pos = 0
    for cid in range(nlist):
        sz = inv.list_size(cid)
        cluster_sizes[cid] = sz
        cluster_offsets[cid] = pos
        ids = faiss.rev_swig_ptr(inv.get_ids(cid), sz).copy().astype(np.int64)
        ordered_ids[pos:pos + sz] = ids
        pos += sz
    cluster_offsets[nlist] = pos
    assert pos == n_total

    ordered_lbls = lbls[ordered_ids].astype(np.uint8)
    ordered_int16 = refs_int16[ordered_ids].astype(np.int16)
    print(f"clusters min/avg/max: {cluster_sizes.min()}/{cluster_sizes.mean():.0f}/{cluster_sizes.max()}")

    # Pack into SoA panels + AoS tail per cluster.
    out = np.empty(n_total * DIM, dtype=np.int16)
    panel_offsets = np.zeros(nlist + 1, dtype=np.uint32)
    wp = 0
    for cid in range(nlist):
        start = cluster_offsets[cid]
        sz = cluster_sizes[cid]
        panel_offsets[cid] = wp
        vecs = ordered_int16[start:start + sz]
        n_full = sz // 8
        for p in range(n_full):
            panel = vecs[p * 8:(p + 1) * 8]          # (8, DIM)
            for d in range(DIM):
                out[wp + d * 8:wp + d * 8 + 8] = panel[:, d]
            wp += 8 * DIM
        for v in range(sz % 8):
            out[wp:wp + DIM] = vecs[n_full * 8 + v]
            wp += DIM
    panel_offsets[nlist] = wp
    assert wp == n_total * DIM

    with open(out_bin, "wb") as f:
        f.write(struct.pack("<IIIIIIII", 0x484E4952, 3, n_total, DIM, nlist, SCALE, 0, 0))
        f.write(centroids.tobytes())
        f.write(cluster_sizes.tobytes())
        f.write(cluster_offsets.tobytes())
        f.write(panel_offsets.tobytes())
        f.write(out.tobytes())
        f.write(ordered_lbls.tobytes())

    print(f"wrote {out_bin} ({os.path.getsize(out_bin) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("refs_npy", help="references_vec.npy from preprocess.py")
    ap.add_argument("lbls_npy", help="references_lbl.npy from preprocess.py")
    ap.add_argument("-o", "--out", default="ivf_int16.bin")
    ap.add_argument("--nlist", type=int, default=4096)
    args = ap.parse_args()
    build(args.refs_npy, args.lbls_npy, args.out, args.nlist)
