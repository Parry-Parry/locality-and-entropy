from collections import defaultdict
from fire import Fire
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
from tqdm import tqdm

if not pt.started():
    pt.init()
import logging
import os
import random
from pyterrier_caching import ScorerCache
from implicit.util import run2lookup, save_json


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


def load_crossencoder(
    model_name_or_path: str,
    batch_size: int = 512,
    verbose: bool = True,
    cache: str = None,
):
    from rankers import CatTransformer

    model = CatTransformer.from_pretrained(
        model_name_or_path, batch_size=batch_size, verbose=verbose
    )
    if cache is not None:
        cached_scorer = ScorerCache(cache, model)
        if not cached_scorer.built():
            dataset = pt.get_dataset("irds:msmarco-passage")
            cached_scorer.build(dataset.get_corpus_iter())
        return cached_scorer
    return model


def mine(
    file,
    dataset: str,
    out_dir: str,
    model_name_or_path: str = None,
    batch_size: int = 512,
    n_neg: int = None,
    n_negs: list = [15],
    cache: str = None,
):
    logging.info(f"Output Directory: {out_dir}")
    logging.info("Loading model...")

    dataset = irds.load(dataset)
    logging.info("Loading dataset...")
    docs = pd.DataFrame(dataset.docs_iter())
    docs_lookup = docs.set_index("doc_id")["text"].to_dict()
    docs = docs.doc_id.to_list()
    query_lookup = (
        pd.DataFrame(dataset.queries_iter()).set_index("query_id")["text"].to_dict()
    )
    triples = pd.read_json(file, orient="records", lines=True)
    doc_id_a_lookup = triples.set_index("query_id").doc_id_a.to_dict()
    doc_id_a_lookup = {str(k): v for k, v in doc_id_a_lookup.items()}

    def pivot_negs(negs):
        frame = {
            "qid": [],
            "docno": [],
            "query": [],
            "text": [],
        }
        for row in negs.itertuples():
            frame["qid"].append(row.query_id)
            frame["docno"].append(row.doc_id_a)
            frame["query"].append(query_lookup[row.query_id])
            frame["text"].append(docs_lookup[row.doc_id_a])
            for doc_id_b in row.doc_id_b:
                frame["qid"].append(row.query_id)
                frame["docno"].append(doc_id_b)
                frame["query"].append(query_lookup[row.query_id])
                frame["text"].append(docs_lookup[doc_id_b])
        return pd.DataFrame(frame)

    crossencoder = load_crossencoder(model_name_or_path, batch_size=batch_size, cache=cache)
    lookup = defaultdict(dict)
    for n_neg in n_negs:
        group_size = n_neg + 1
        tmp_triples = triples.copy()
        tmp_triples["doc_id_b"] = [random.sample(docs, k=n_neg) for _ in range(len(triples))]
        frame = pivot_negs(tmp_triples)
        res = crossencoder.transform(frame)
        for row in tqdm(res.itertuples()):
            lookup[row.qid][row.docno] = row.score
        tmp_triples.to_json(
            out_dir + f"/random.{group_size}.jsonl", orient="records", lines=True
        )

    save_json(lookup, out_dir + f"/random.scores.json.gz")

    return f"Successfully saved to {out_dir}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(mine)
