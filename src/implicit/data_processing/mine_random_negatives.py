from collections import defaultdict
from fire import Fire
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
import tqdm
import json
if not pt.started():
    pt.init()
import logging
import os
import random
from pyterrier_caching import ScorerCache
from implicit.util import save_json, load_json
from more_itertools import chunked


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
    verbose: bool = False,
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
    batch_size: int = 768,
    n_neg: int = None,
    n_negs: list = [15],
    depth : int = 50,
    cache: str = None,
):
    logging.info(f"Output Directory: {out_dir}")
    logging.info("Loading model...")

    dataset = irds.load(dataset)
    logging.info("Loading dataset...")
    docs = pd.DataFrame(dataset.docs_iter())
    docs_lookup = docs.set_index("doc_id")["text"].to_dict()
    docs = docs.doc_id.to_list()
    queries = pd.DataFrame(dataset.queries_iter())
    query_lookup = (
        queries.set_index("query_id")["text"].to_dict()
    )
    triples = load_json(file)

    query_pos_lookup = {}
    for row in triples:
        if row["query_id"] not in query_pos_lookup:
            query_pos_lookup[row["query_id"]] = []
        query_pos_lookup[row["query_id"]].append(row["doc_id_a"])

    negatives = [random.sample(docs, k=depth) for _ in range(len(query_pos_lookup))]
    crossencoder = load_crossencoder(model_name_or_path, batch_size=batch_size, cache=cache)
    lookup = defaultdict(dict)
    n_neg = n_neg or n_negs[0]
    group_size = n_negs[0] + 1
    out_file = out_dir + f"/random.{group_size}.jsonl"
    for batch in tqdm(chunked(zip(query_pos_lookup.items(), negatives), 100)):
        frame = {
            "qid": [],
            "docno": [],
            "query": [],
            "text": [],
        }
        for (query_id, pos), negs in batch:
            query_text = query_lookup[str(query_id)]
            for doc_id in pos:
                frame["qid"].append(query_id)
                frame["docno"].append(doc_id)
                frame["query"].append(query_text)
                frame["text"].append(docs_lookup[str(doc_id)])
            for doc_id in negs:
                frame["qid"].append(query_id)
                frame["docno"].append(doc_id)
                frame["query"].append(query_text)
                frame["text"].append(docs_lookup[str(doc_id)])
        frame = pd.DataFrame(frame).drop_duplicates()
        res = crossencoder.transform(frame)
        for row in res.itertuples():
            lookup[row.qid][row.docno] = row.score
    
    with open(out_file, "w") as f:
        for (query_id, pos), negs in tqdm.tqdm(zip(query_pos_lookup.items(), negatives)):
            for doc_id in pos:
                f.write(json.dumps({"query_id": query_id, "doc_id_a": doc_id, "doc_id_b": [x for x in random.sample(docs, k=n_neg)]}) + "\n")

    save_json(lookup, out_dir + f"/random.scores.json.gz")

    return f"Successfully saved to {out_dir}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(mine)
