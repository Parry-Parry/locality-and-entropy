"""
Metric–measure locality utilities
---------------------------------
* robust_diameter : quantile of pairwise cosine distances  (Δ_q)
* query_density   : TwoNN + k-NN density estimate          (ρ_q)

Assumes:
    • FlexIndex returns float32 document embeddings (size d, already on CPU)
    • cosine distance is computed on L2-normalised vectors
    • FAISS is available (`pip install faiss-cpu` or `faiss-gpu`)
"""
import json, argparse, random, math
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import faiss                      #  <-- NEW
from scipy.spatial.distance import pdist

# -----------------------   metric helpers   -------------------------
def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)

def robust_diameter(emb, alpha: float = 0.99) -> float:
    dists = pdist(emb, metric="cosine")
    return float(np.quantile(dists, alpha))

# ----------  TwoNN intrinsic dimension & k-NN density  --------------
def twonn_dimension(r1: np.ndarray, r2: np.ndarray) -> float:
    """Facco et al. (2017) two-NN intrinsic-dimension estimator."""
    # avoid log(0) by clipping
    eps = 1e-12
    ratios = (r2 + eps) / (r1 + eps)
    return 1.0 / np.mean(np.log(ratios))

def knn_density(r_k: np.ndarray, k: int, N: int, d_hat: float) -> np.ndarray:
    """
    r_k : (N,)   distance to k-th NN (cosine in [0,2])
    Returns per-point density up to the unit-ball constant.
    """
    # Volume constant V_d cancels if you only compare densities, but keep it for completeness
    V_d = (math.pi ** (d_hat / 2)) / math.gamma(d_hat / 2 + 1)
    return k / (N * V_d * np.power(r_k, d_hat).clip(min=1e-12))

# ---------------------------  dataset  ------------------------------
# (identical to your TrainingDataset except for doc-vector normalisation)
class TrainingDataset(...):
    ...
    def _get_doc(self, doc_id):
        vec = self.d_vecs[self.d_idx.inv[doc_id]]
        return vec                      # already float32

# ----------------------  per-collection pre-compute  ----------------
def build_faiss_index(all_vectors: np.ndarray, use_gpu: bool = False):
    all_vectors = l2_normalise(all_vectors.astype("float32"))
    index = faiss.IndexFlatIP(all_vectors.shape[1])       # inner-product for cosine
    if use_gpu:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(all_vectors)
    return index

def compute_global_knn(index, batch_vectors, k: int = 10):
    batch_vectors = l2_normalise(batch_vectors.astype("float32"))
    # faiss returns similarities; convert to cosine distances
    sim, _ = index.search(batch_vectors, k + 1)           # +1 for self-match
    sim = sim[:, 1:]                                      # drop self
    r_k  = 1.0 - sim[:, -1]                               # cosine distance
    r1   = 1.0 - sim[:, 0]
    r2   = 1.0 - sim[:, 1]
    return r1, r2, r_k

# --------------------------  main loop  -----------------------------
TRAIN_JSONL = [
    "data/bm25.16.jsonl",
    "data/crossencoder.16.jsonl",
    "data/random.16.jsonl",
    "data/ensemble.16.jsonl",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output.json")
    parser.add_argument("--k", type=int, default=10, help="k for k-NN density")
    parser.add_argument("--gpu", action="store_true", help="use faiss-gpu if available")
    args = parser.parse_args()

    # 1 ─── load the whole embedding matrix once
    flex = FlexIndex("data/doc_embeddings")
    d_idx, d_vecs, _ = flex.payload()
    all_vecs = d_vecs.astype("float32")        # (N_docs, d)
    faiss_idx = build_faiss_index(all_vecs, use_gpu=args.gpu)
    N_docs = all_vecs.shape[0]

    output = {}
    for data in tqdm(TRAIN_JSONL, desc="Runs"):
        name = data.split("/")[-1].split(".")[0]
        ds   = TrainingDataset(data, flex, group_size=16, no_positive=False)
        dl   = DataLoader(ds, batch_size=1024, shuffle=False,
                          collate_fn=TrainingCollator(None))

        diam_q, rho_q, qids = [], [], []
        for batch in tqdm(dl, desc=f"{name}"):
            qids.extend(batch["query_id"])
            emb  = batch["docs_batch"]                     # (B*16, d)
            emb  = emb.reshape(len(batch["query_id"]), 16, -1)

            # --- robust diameter Δ̂_q ---------------------
            for e in emb:
                diam_q.append(robust_diameter(e, alpha=0.99))

            # --- density ρ̂_q via global k-NN -------------
            flat = emb.reshape(-1, emb.shape[-1])          # flatten for FAISS
            r1, r2, r_k = compute_global_knn(faiss_idx, flat, k=args.k)
            d_hat = twonn_dimension(r1, r2)
            rho   = knn_density(r_k, args.k, N_docs, d_hat)
            rho   = rho.reshape(len(batch["query_id"]), 16)
            rho_q.extend(np.median(rho, axis=1))

        output[name] = {
            "diameter_0.99":        float(np.quantile(diam_q, 0.99)),
            "density_median":       float(np.median(rho_q)),
            "diameters_per_query":  dict(zip(qids, map(float, diam_q))),
            "density_per_query":    dict(zip(qids, map(float, rho_q))),
            "num_queries":          len(ds),
            "num_docs":             len(ds) * 16,
            "k_nn":                 args.k,
        }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
