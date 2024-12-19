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
from implicit.util import run2lookup, save_json, load_json


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
    cache: str = None,
    chunk_batches: int = 10000,
    name_override : str = None
):
    chunk_size = chunk_batches * batch_size
    dataset = irds.load(dataset)
    queries = pd.DataFrame(dataset.queries_iter()).set_index("query_id").text.to_dict()
    docs = pd.DataFrame(dataset.docs_iter()).set_index("doc_id").text.to_dict()
    lookup = defaultdict(dict)
    name = model_name_or_path.replace("/", "-") if name_override is None else name_override
    if os.path.exists(out_dir + f"/{name}.scores.json.gz"):
        lookup = load_json(out_dir + f"/{name}.scores.json.gz")
        print(type(next(iter(lookup.items()))[0]))

    def pivot_triples(triples):
        frame = {
            "qid": [],
            "docno": [],
            "query": [],
            "text": [],
        }
        for row in tqdm(triples.itertuples(), desc="Pivoting triples"):
            qid = str(row.query_id)
            doc_id_a = str(row.doc_id_a)
            doc_id_b = (
                [str(x) for x in row.doc_id_b]
                if type(row.doc_id_b) == list
                else [str(row.doc_id_b)]
            )
            query_text = queries[qid]
            if qid not in lookup:
                lookup[qid] = {}
            if doc_id_a not in lookup[qid]:
                frame["qid"].append(qid)
                frame["docno"].append(doc_id_a)
                frame["text"].append(docs[doc_id_a])
                frame["query"].append(query_text)

            for id in doc_id_b:
                if id not in lookup[qid]:
                    frame["qid"].append(qid)
                    frame["docno"].append(id)
                    frame["text"].append(docs[id])
                    frame["query"].append(query_text)

        frame["score"] = [0.0] * len(frame["qid"])
        return pd.DataFrame(frame)

    triples = pd.read_json(file, lines=True, orient="records", chunksize=chunk_size)
    logging.info("Loading crossencoder...")
    crossencoder = load_crossencoder(
        model_name_or_path, batch_size=batch_size, cache=cache
    )

    for group in triples:
        frame = pivot_triples(group)
        # filter if we already have scores for a qid-docno pair

        res = crossencoder.transform(frame)
        for row in tqdm(res.itertuples()):
            lookup[row.qid][row.docno] = row.score
    if name_override:
        name = name_override
    save_json(lookup, out_dir + f"/{name}.scores.json.gz")

    return f"Successfully saved to {out_dir}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(mine)
