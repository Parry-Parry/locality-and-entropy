from collections import defaultdict
from fire import Fire
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import json
import numpy as np
if not pt.started():
    pt.init()
import logging
import os
import gzip
from implicit.util import save_json, load_json


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
    return model


def mine(
    file,
    dataset: str,
    out_dir: str,
    model_name_or_path: str = None,
    N_DOCS: int = 12000000,
    runtime_batch_size: int = 8,
    runtime_group_size: int = 16,
    batch_size: int = 512,
    cache_every: int = 10,
    cache: str = None,
    chunk_batches: int = 10,
    name_override : str = 'ensemble.all'
):
    num_steps = N_DOCS // (runtime_batch_size * runtime_group_size)
    num_queries = num_steps * runtime_batch_size
    chunk_size = chunk_batches * batch_size
    dataset = irds.load(dataset)
    queries = pd.DataFrame(dataset.queries_iter()).set_index("query_id").text.to_dict()
    docs = pd.DataFrame(dataset.docs_iter()).set_index("doc_id").text.to_dict()

    relevant_pairs = set()
    relevant_queries = set()
    
    with open('data/triples.jsonl') as f:
        for line in tqdm(f):
            line = json.loads(line)
            qid = line['query_id']
            docid = line['doc_id_a']
            relevant_pairs.add((qid, docid))
            relevant_queries.add(qid)

    name = name_override

    out_file = out_dir + f"/{name}.scores.json.gz"
    if os.path.exists(out_file):
        lookup = load_json(out_file)
    else:
        lookup = defaultdict(dict)

    def pivot_triples(triples):
        frame = {
            "qid": [],
            "docno": [],
            "query": [],
            "text": [],
        }
        #print("Pivoting triples")
        for row in tqdm(triples, desc="Pivoting triples"):
            #print("Pivoting row")
            qid = str(row['qid'])
            if qid not in relevant_queries:
                continue
            if type(row['pos']) is list and len(row['pos']) > 0:
                doc_id_a = [str(x) for x in row['pos']]
            elif type(row['pos']) is str:
                doc_id_a = row['pos']
            elif type(row['pos']) is int:
                doc_id_a = str(row['pos'])
            else:
                continue

            negs = row['neg']
            doc_id_b = set()
            for _, idx in negs.items():
                doc_id_b.update([str(x) for x in idx])

            query_text = queries[qid]
            if qid not in lookup:
                lookup[qid] = {}
            if type(doc_id_a) is list:
                for id in doc_id_a:
                    if (qid, id) not in relevant_pairs:
                        continue
                    if id not in lookup[qid]:
                        frame["qid"].append(qid)
                        frame["docno"].append(id)
                        frame["text"].append(docs[id])
                        frame["query"].append(query_text)
            else:
                if (qid, doc_id_a) not in relevant_pairs:
                    continue
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
        #print("Pivoting done")
        frame["score"] = [0.0] * len(frame["qid"])
        return pd.DataFrame(frame)

    logging.info("Loading crossencoder...")
    crossencoder = load_crossencoder(
        model_name_or_path, batch_size=batch_size, cache=cache
    )

    # read json lines by line in chunks using a buffer
    N_CHUNKS = 0
    with gzip.open(file, "rt") as f:
        total_lines = sum(1 for _ in f)
        remaining_lines = total_lines
        # random num queries to read
        f.seek(0)
        print(f"reading file with chunk size {chunk_size}, total lines {remaining_lines}")
        buffer = []
        for i, line in enumerate(f):
            # print("Reading line")
            buffer.append(json.loads(line))
            buffer_len = len(buffer)
            # print(f"Buffer length: {buffer_len}")
            if buffer_len >= chunk_size or remaining_lines < chunk_size:
                N_CHUNKS += 1
                remaining_lines -= buffer_len
                frame = pivot_triples(buffer)
                if len(frame) == 0:
                    buffer = []
                    continue
                res = crossencoder.transform(frame)
                #print(res.head())
                for row in tqdm(res.itertuples()):
                    lookup[row.qid][row.docno] = row.score
                buffer = []
                if N_CHUNKS % cache_every == 0 and N_CHUNKS > 0:
                    save_json(lookup, out_dir + f"/{name}.scores.json.gz")
    save_json(lookup, out_dir + f"/{name}.scores.json.gz")

    return f"Successfully saved to {out_dir}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(mine)
