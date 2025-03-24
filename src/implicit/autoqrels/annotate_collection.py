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
from implicit.util import save_json, load_json

def load_labeller(
    dataset: str,
    backbone: str = 'google/flan-t5-xl',
    batch_size: int = 8,
    verbose: bool = False,
    query_id_field: str = 'query_id',
    doc_id_field: str = 'doc_id',
    query_field: str = 'text',
    doc_field: str = 'text',
    max_src_len: int = 330,
    cache_path: str = None
):
    from .label_runner import LabelRunner

    return LabelRunner(
        dataset,
        backbone=backbone,
        batch_size=batch_size,
        verbose=verbose,
        query_id_field=query_id_field,
        doc_id_field=doc_id_field,
        query_field=query_field,
        doc_field=doc_field,
        max_src_len=max_src_len,
        cache_path=cache_path
    )


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
    lookup = defaultdict(dict)
    name = model_name_or_path.replace("/", "-") if name_override is None else name_override
    if os.path.exists(out_dir + f"/{name}.scores.json.gz"):
        lookup = load_json(out_dir + f"/{name}.scores.json.gz")
        print(type(next(iter(lookup.items()))[0]))

    triples = pd.read_json(file, lines=True, orient="records", chunksize=chunk_size)

    def triples_iterator():
        for chunk in triples:
            # if any qid in chunk is in lookup, skip
            out = []
            list_chunk = chunk.to_dict(orient="records")

            for element in list_chunk:
                qid = element["query_id"]
                if qid not in lookup:
                    out.append(element)
            yield out

    logging.info("Loading crossencoder...")
    labeller = load_labeller(
        dataset,
        model_name_or_path,
        batch_size,
        cache=cache,
        verbose=True
    )

    for group in tqdm(triples_iterator()):

        res = labeller.transform(group)
        lookup.update(res)
    if name_override:
        name = name_override
    save_json(lookup, out_dir + f"/{name}.scores.json.gz")

    return f"Successfully saved to {out_dir}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(mine)
