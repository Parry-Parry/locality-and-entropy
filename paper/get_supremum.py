import json
import gzip
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
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


class TrainingDataset(Dataset):
    def __init__(
        self,
        training_dataset_file: str,
        doc_embeddings: FlexIndex,
        group_size: int = 16,
        no_positive: bool = False,
        text_field: str = "text",
        query_field: str = "text",
    ) -> None:
        assert training_dataset_file.endswith(
            "jsonl"
        ), "Training dataset should be a JSONL file and should not be compressed"

        self.training_dataset_file = training_dataset_file
        self.doc_embeddings = doc_embeddings
        self.group_size = group_size
        self.no_positive = no_positive
        self.n_neg = self.group_size - 1 if not self.no_positive else self.group_size
        self.text_field = text_field
        self.query_field = query_field

        self.d_idx, self.d_vecs, _ = self.doc_embeddings.payload()

        self.line_offsets = self._get_line_offsets()
        super().__init__()
        self.__post_init__()

    def _get_line_offsets(self):
        """Store byte offsets for each line in an uncompressed JSONL file."""
        offsets = []
        with open(self.training_dataset_file, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(offset)
        return offsets

    def _get_line_by_index(self, idx):
        """Retrieve a line by index, using offsets for uncompressed files."""
        with open(self.training_dataset_file, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[idx])
            return json.loads(f.readline())

    def __post_init__(self):
        # Use _get_line_by_index to check multi-negative configuration
        first_entry = self._get_line_by_index(0)
        self.multi_negatives = isinstance(first_entry["doc_id_b"], list)
        total_negs = len(first_entry["doc_id_b"]) if self.multi_negatives else 1
        assert (
            self.n_neg <= total_negs
        ), f"Only found {total_negs} negatives, cannot take {self.n_neg} negatives"

    def _get_doc(self, doc_id):
        return self.d_vecs[self.d_idx.inv[doc_id]]

    def _get(self, item):
        query_id = item["query_id"]
        doc_id_a = item["doc_id_a"]
        doc_id_b = item["doc_id_b"]

        doc_id_a_text = [self._get_doc(str(doc_id_a))]
        doc_id_b_text = (
            [self._get_doc(str(doc_id)) for doc_id in doc_id_b]
            if self.multi_negatives
            else [self._get_doc(str(doc_id_b))]
        )

        return None, query_id, doc_id_a, doc_id_a_text, doc_id_b, doc_id_b_text

    def __len__(self):
        # Length based on line offsets for uncompressed, or generator count for compressed
        return (
            len(self.line_offsets)
            if self.line_offsets
            else sum(1 for _ in self._data_generator())
        )

    def __getitem__(self, idx):
        # Retrieve the line corresponding to idx
        item = self._get_line_by_index(idx)

        _, query_id, _, doc_id_a_text, _, doc_id_b_text = self._get(
            item
        )

        if len(doc_id_b_text) > (self.n_neg):
            doc_id_b_text = random.sample(doc_id_b_text, self.n_neg)
        return (query_id, doc_id_a_text + doc_id_b_text)


class TrainingCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch) -> dict:
        qidx, doc_embedding, label = [], [], []

        for q, d, *l in batch:
            qidx.append(q)
            doc_embedding.extend(d)
            if l:
                label.extend(l[0])
        # Convert lists of NumPy arrays to single NumPy arrays before converting to tensors
        doc_embedding = np.array(doc_embedding)
        label = torch.tensor(np.array(label)) if label else None
        return {
            "query_id": qidx,
            "docs_batch": doc_embedding,
            "labels": label,
        }


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
    TOTAL_DOCS = 12000000
    N_QUERIES = TOTAL_DOCS / 16
    output = {}
    for data in tqdm(TRAIN_JSONL, desc="Processing All"):
        name = data.split("/")[-1].split(".")[0]
        dataset = TrainingDataset(
            data,
            FlexIndex("data/doc_embeddings"),
            group_size=16,
            no_positive=False,
        )
        # make dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=TrainingCollator(None),
        )
        deltas, qids = [], []
        for i, batch in tqdm(enumerate(dataloader), desc="Processing Batches", total=len(dataloader)):
            docs = batch["docs_batch"]
            qids = batch["query_id"]
            deltas.append(
                robust_diameter(
                    docs,
                    alpha=0.99,
                )
            )
        
        diameters = {qid: delta for qid, delta in zip(qids, deltas)}

        overall_diameter = np.quantile(deltas, 0.99)
        output[name] = {
            "diameter": overall_diameter,
            "diameters": diameters,
            "num_queries": len(dataset),
            "num_docs": len(dataset)*16,
        }
    with open(parser.output, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
