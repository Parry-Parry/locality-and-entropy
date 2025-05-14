import json
import gzip
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from rankers import DotDataCollator, TrainingDataset
import ir_datasets as irds
import torch
from torch.utils.data import DataLoader
import json
from pyterrier_dr import FlexIndex
import random
from scipy.spatial.distance import pdist

def robust_diameter(embeddings, alpha=0.99):
    """
    embeddings : (k, d) float32/float64 ndarray
        k document vectors for a single query (k = 16 here).
    alpha      : float in (0,1)
        Quantile used as robust upper bound; default 0.99.

    Returns
    -------
    Δ_hat : float
        Robust empirical diameter for this query:
        the alpha-quantile of all pairwise cosine distances.
    """
    # pairwise cosine distances (size k*(k-1)/2 = 120 for k=16)
    dists = pdist(embeddings, metric="cosine")
    Δ_hat = np.quantile(dists, alpha)
    return float(Δ_hat)


# Training / score file pairs  (extend or modify as needed)
TRAIN_JSONL=[
  "data/bm25.16.jsonl",
  "data/crossencoder.16.jsonl",
  "data/random.16.jsonl",
  "data/ensemble.16.jsonl",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Output file to save the results",
    )
    # Define constant
    TOTAL_DOCS=12000000
    BASE_BATCH_SIZE=64
    GROUP_SIZE=16
    # how many steps to get, TOTAL_DOCS / (BATCH_SIZE * GROUP_SIZE)
    PER_BATCH_DOCS=BASE_BATCH_SIZE * GROUP_SIZE
    total_steps=TOTAL_DOCS // PER_BATCH_DOCS

    print(f"Total steps: {total_steps} for {TOTAL_DOCS} docs")

    bert = AutoModel.from_pretrained("bert-base-uncased").cuda()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    output = {}
    for data in tqdm(TRAIN_JSONL, desc="Processing All"):
        name = data.split("/")[-1].split(".")[0]
        dataset = TrainingDataset(
            data,
            group_size=16,
            corpus=irds.load("msmarco-passage/train/triples-small"),
            no_positive=False,
            teacher_file=None,
            lazy_load_text=True,
        )
        # make dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=BASE_BATCH_SIZE,
            shuffle=False,
            collate_fn=DotDataCollator(tokenizer),
            num_workers=4,
        )
        deltas = []
        for i, batch in tqdm(enumerate(dataloader), desc="Processing Batches", total=total_steps):
            if i >= total_steps:
                break
            docs = batch["docs_batch"]
            # Move the tensors to GPU
            docs = {k: v.cuda() for k, v in docs.items()}
            with torch.no_grad():
                # Get the CLS token representations
                cls_vectors_numpy = bert(**docs).last_hidden_state[:, 0, :].cpu().numpy()
            
            # reshape for group size
            cls_vectors_numpy = cls_vectors_numpy.reshape(
                (cls_vectors_numpy.shape[0] // GROUP_SIZE, GROUP_SIZE, -1)
            )
            deltas.extend(
                robust_diameter(
                    cls_vectors_numpy[i],
                    alpha=0.99,
                )
                for i in range(cls_vectors_numpy.shape[0])
            )

        overall_diameter = np.quantile(deltas, 0.99)
        output[name] = {
            "diameter": overall_diameter,
            "diameters": deltas,
            "num_docs": TOTAL_DOCS,
            "num_queries": BASE_BATCH_SIZE*total_steps,
        }
    with open(parser.output, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
