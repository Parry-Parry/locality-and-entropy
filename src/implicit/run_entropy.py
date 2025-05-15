from collections import defaultdict
from fire import Fire
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import json
import random

if not pt.started():
    pt.init()
import logging
import os
import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid
from implicit.util import save_json, load_json


def entropy(prob_dist):
    """Compute Shannon entropy efficiently."""
    return -np.sum(prob_dist * np.log2(prob_dist + 1e-12))  # Avoid log(0)

def pairwise_entropy(scores):
    """
    scores : (k,) 1-D array of raw (unnormalised) scores for one query.
    Returns the average binary entropy over all i<j pairs (bits).
    """
    # pair-wise logit differences
    diff = scores[:, None] - scores[None, :]          # shape (k, k)
    p    = sigmoid(diff)                              # Bernoulli probs

    # binary Shannon entropy, add tiny ε for numerical safety
    eps  = 1e-12
    h    = -(p * np.log2(p + eps) + (1.0 - p) * np.log2(1.0 - p + eps))

    # use only upper-triangular part (i < j) to avoid double-counting
    k = len(scores)
    return h[np.triu_indices(k, k=1)].mean()       

def mine(
    file,
    teacher_file: str,
    dataset: str,
    out_dir: str,
    model_name_or_path: str = None,
    batch_size: int = 512,
    group_size: int = 16,
    chunk_batches: int = 10000,
    name_override : str = None
):
    chunk_size = chunk_batches * batch_size
    dataset = irds.load(dataset)
    lookup = defaultdict(dict)
    name = model_name_or_path.split('/')[-1] if name_override is None else name_override
    out_file = out_dir + f"/{name}.{group_size}.entropy.json.gz"
    if os.path.exists(out_file):
        logging.info(f"File already exists at {out_file}")
        return f"File already exists at {out_file}"
    if os.path.exists(teacher_file):
        lookup = load_json(teacher_file)
    else:
        raise FileNotFoundError(teacher_file)

    def pivot_triples(triples):
        frame = {
            "qid": [],
            "docno": [],
            "score": [],
        }
        for row in tqdm(triples, desc="Pivoting triples"):
            qid = str(row['query_id'])
            doc_id_a = str(row['doc_id_a'])
            doc_id_b = (
                [str(x) for x in row['doc_id_b']]
                if type(row['doc_id_b']) is list
                else [str(row['doc_id_b'])]
            )
            doc_id_b = random.sample(doc_id_b, k=group_size - 1) if len(doc_id_b) != group_size - 1 else doc_id_b

            if qid not in lookup:
                continue

            frame["qid"].append(qid)
            frame["docno"].append(doc_id_a)

            for id in doc_id_b:
                frame["qid"].append(qid)
                frame["docno"].append(id)

            frame["score"].extend([lookup[qid].get(docno, 0.0) for docno in [doc_id_a] + doc_id_b])
        return pd.DataFrame(frame)


    entropy_lookup = {}
    # read json lines by line in chunks using a buffer
    with open(file, "r") as f:
        buffer = []
        for line in f:
            buffer.append(json.loads(line))
            if len(buffer) == chunk_size:
                frame = pivot_triples(buffer)
                if len(frame) == 0:
                    buffer = []
                    continue

                for qid, group in tqdm(frame.groupby("qid")):
                    scores = group["score"].values.astype(np.float64)
                    entropy_lookup[qid] = pairwise_entropy(scores)

                buffer = []
    
    save_json(entropy_lookup, out_file)

    return f"Successfully saved to {out_dir}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(mine)
