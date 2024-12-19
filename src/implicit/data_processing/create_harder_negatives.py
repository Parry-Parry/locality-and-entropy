import gzip
import json
from tqdm import tqdm
import os
import logging
import pickle
import sys
import requests
from fire import Fire
import pandas as pd
import ir_datasets as irds
import random

def http_get(url: str, path: str) -> None:
    """
    Downloads a URL to a given path on disk.

    Args:
        url (str): The URL to download.
        path (str): The path to save the downloaded file.

    Raises:
        requests.HTTPError: If the HTTP request returns a non-200 status code.

    Returns:
        None
    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print(f"Exception when trying to download {url}. Response {req.status_code}", file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


def get_negatives(triples_file : str, num_negs_per_system=5, ce_score_margin=3.0, data_folder="data", n_neg=15):
    all_docs = pd.DataFrame(irds.load('msmarco-passage').docs_iter()).doc_id.to_list()
    ce_scores_file = os.path.join(data_folder, "cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz")
    if not os.path.exists(ce_scores_file):
        logging.info("Download cross-encoder scores file")
        http_get(
            "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
            ce_scores_file,
        )
    triples = pd.read_json(triples_file, lines=True, orient="records", chunksize=100000)
    logging.info("Load CrossEncoder scores dict")
    with gzip.open(ce_scores_file, "rb") as fIn:
        ce_scores = pickle.load(fIn)

    # As training data we use hard-negatives that have been mined using various systems
    train_file_path = os.path.join(data_folder, "msmarco-hard-negatives.jsonl.gz")
    if not os.path.exists(train_file_path):
        logging.info("Download cross-encoder scores file")
        http_get(
            "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz",
            train_file_path,
        )

    negs_to_use = None
    lookup = {}

    with gzip.open(train_file_path, 'rt') as fIn:
        for line in tqdm(fIn):
            data = json.loads(line)

            # Get the positive passage ids
            qid = data["qid"]
            pos_pids = data["pos"]

            if len(pos_pids) == 0:  # Skip entries without positives passages
                continue

            pos_min_ce_score = min([ce_scores[qid][pid] for pid in data["pos"]])
            ce_score_threshold = pos_min_ce_score - ce_score_margin

            # Get the hard negatives
            neg_pids = set()
            if negs_to_use is None:
                negs_to_use = list(data["neg"].keys())

            for system_name in negs_to_use:
                if system_name not in data["neg"]:
                    continue

                system_negs = data["neg"][system_name]
                negs_added = 0
                for pid in system_negs:
                    if ce_scores[qid][pid] > ce_score_threshold:
                        continue

                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break
        
            neg_pids = list(neg_pids)
            if len(neg_pids) < n_neg:
                neg_pids = neg_pids + random.sample(all_docs, n_neg - len(neg_pids))
            lookup[data['qid']] = neg_pids
    group_size = n_neg + 1
    out_file = os.path.join(data_folder, f"ensemble.{group_size}.jsonl")
    with open(out_file, "w") as f:
        for batch in triples:
            for row in batch.itertuples():
                try:
                    doc_id_b = lookup[str(row.query_id)]
                except KeyError:
                    print(f"Query ID {row.query_id} not found")
                    continue
                f.write(json.dumps({"query_id": row.query_id, "doc_id_a": doc_id_b, "doc_id_b": row.doc_id_b}) + "\n")

    return out_file


if __name__ == "__main__":
    Fire(get_negatives)
