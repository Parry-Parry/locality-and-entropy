#!/usr/bin/env python3
"""
compute_batch_coverage.py

Compute query-agnostic test-document coverage
for every SCORE × RUN combination.

Coverage = |S ∩ R| / |R|
    S  : unique docids in the score file
    R  : unique docids in the run file (optionally top-k)

Score files (gzip JSON):
    data/bm25.scores.json.gz
    data/crossencoder.scores.json.gz
    data/random.scores.json.gz
    data/ensemble.all.scores.json.gz

Run files (gzip TREC 6-column):
    msmarco-passage-trec-dl-2019-judged-bm25.run.gz
    msmarco-passage-trec-dl-2020-judged-bm25.run.gz
"""

import gzip, json, argparse, sys, csv, itertools, pathlib
import pyterrier as pt
from tqdm import tqdm

SCORE_GZ = [
    "data/bm25.scores.json.gz",
    "data/crossencoder.scores.json.gz",
    "data/random.scores.json.gz",
    "data/ensemble.all.scores.json.gz",
]

RUN_GZ = [
    "data/msmarco-passage-trec-dl-2019-judged-bm25.run.gz",
    "data/msmarco-passage-trec-dl-2020-judged-bm25.run.gz",
]

def read_score_docs(path):
    docs = set()
    with gzip.open(path, "rt") as f:
        data = json.load(f)

    #  {qid: {docid: score}}
    for qdict in data.values():
        keys = [str(k) for k in qdict.keys()]
        docs.update(keys)

    return docs

def read_run_docs(path, topk=None):
    results = pt.io.read_results(path)
    if topk is not None:
        results = results.sort_values(["qid", "score"], ascending=[True, False]).groupby("qid").head(topk)
    unique_docs = set(results.docno.unique().tolist())
    unique_docs = {str(d) for d in unique_docs}

    return unique_docs

def main():
    ap = argparse.ArgumentParser(
        description="Compute coverage for all score/run combinations")
    ap.add_argument("--topk", type=int, default=None,
                    help="restrict R to top-k docs per query")
    ap.add_argument("--csv", type=str,
                    help="optional output CSV file")
    args = ap.parse_args()

    # cache loaded sets to avoid double I/O
    score_sets = {p: read_score_docs(p) for p in SCORE_GZ}
    run_sets   = {p: read_run_docs(p, args.topk) for p in RUN_GZ}

    rows = []
    for score_path, run_path in tqdm(itertools.product(SCORE_GZ, RUN_GZ)):
        S = score_sets[score_path]
        R = run_sets[run_path]
        if not R:
            cov = float("nan")
        else:
            cov = len(S & R) / len(R)
        rows.append([score_path, run_path,
                     len(S), len(R), len(S & R), f"{cov:.4f}"])

    header = ["score_file", "run_file",
              "|S|", "|R|", "|S∩R|", "coverage"]
    print("\t".join(header))
    for r in rows:
        print("\t".join(map(str, r)))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
