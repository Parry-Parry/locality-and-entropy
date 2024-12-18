from fire import Fire
import ir_datasets as irds
import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init()
import logging
import os
import random
import pyterrier_alpha as pta

def bm25(index_dir: str, k1: float = 1.2, b: float = 0.75, threads: int = 1, **kwargs):
    import pyterrier as pt

    if not pt.started():
        pt.init()
    from pyterrier_pisa import PisaIndex

    if os.path.exists(index_dir):
        logging.info(f"Loading index from {index_dir}")
        return PisaIndex(index_dir, threads=threads, **kwargs).bm25(
            k1=k1, b=b, verbose=True
        )
    else:
        logging.info("assuming pre-built index")
        return PisaIndex.from_dataset(index_dir, threads=threads, **kwargs).bm25(
            k1=k1, b=b, verbose=True
        )


def mine(
    file,
    dataset: str,
    out_dir: str,
    index_path: str = 'macavaney/msmarco-passage.corpusgraph.bm25.1024',
    subset_depth: int = 100,
    n_neg: int = 15,
):
    logging.info(f"Index: {index_path}")
    logging.info(f"Output Directory: {out_dir}")
    logging.info("Loading model...")
    if index_path is None:
        index_path = "msmarco_passage"
    name = "bm25-graph"
    graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.1024').to_limit_k(subset_depth)

    dataset = irds.load(dataset)
    logging.info("Loading dataset...")
    docs = pd.DataFrame(dataset.docs_iter())
    docs = docs.doc_id.to_list()
    triples = pd.read_json(file, orient="records", lines=True)

    def get_negatives(doc_id_a):
        candidates = graph.neighbours(doc_id_a)
        length = len(candidates)
        if length < n_neg:
            return candidates + random.sample(docs, k=n_neg - length)
        return random.sample(candidates, k=n_neg)

    triples["doc_id_b"] = triples["doc_id_a"].map(get_negatives)
    group_size = n_neg + 1
    out_file = out_dir + f"/{name}.{group_size}.jsonl.gz"
    triples.to_json(out_file, orient="records", lines=True, compression="gzip")
    return f"Successfully saved to {out_file}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(mine)
