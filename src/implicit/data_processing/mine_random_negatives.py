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
from implicit.util import save_json


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
    triples = pd.read_json(file, orient="records", lines=True, chunksize=100*batch_size)

    # get the jsonl linecount without ingesting the file
    line_count = 0
    with open(file, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            line_count += 1
            if not line:
                break
    print(f"Line count: {line_count}")

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
            frame["query"].append(query_lookup[str(row.query_id)])
            frame["text"].append(docs_lookup[str(row.doc_id_a)])
            for doc_id_b in row.doc_id_b:
                frame["qid"].append(row.query_id)
                frame["docno"].append(doc_id_b)
                frame["query"].append(query_lookup[str(row.query_id)])
                frame["text"].append(docs_lookup[str(doc_id_b)])
        return pd.DataFrame(frame).drop_duplicates()

    crossencoder = load_crossencoder(model_name_or_path, batch_size=batch_size, cache=cache)
    lookup = defaultdict(dict)
    n_neg = n_neg or n_negs[0]
    group_size = n_negs[0] + 1
    out_file = out_dir + f"/random.{group_size}.jsonl"

    progress_bar = tqdm.tqdm(total=line_count)
    with open(out_file, "a") as f:
        for chunk in triples:
            progress_bar.update(len(chunk))
            chunk["doc_id_b"] = [random.sample(docs, k=n_neg) for _ in range(len(chunk))]
            frame = pivot_negs(chunk)
            res = crossencoder.transform(frame)
            for row in res.itertuples():
                lookup[row.qid][row.docno] = row.score
            for row in chunk.itertuples():
                f.write(json.dumps({"query_id": row.query_id, "doc_id_a": row.doc_id_a, "doc_id_b": row.doc_id_b}) + "\n")
                f.flush()

    save_json(lookup, out_dir + f"/random.scores.json.gz")

    return f"Successfully saved to {out_dir}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(mine)
