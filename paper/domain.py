#!/usr/bin/env python3
"""
compute_batch_coverage.py

Compute query‑agnostic test‑document coverage
for every SCORE × RUN combination.

Coverage = |S ∩ R| / |R|
    S  : unique docids in the score file
    R  : unique docids in the run file (optionally top‑k)

In addition to raw coverage, we also report a normalised
metric that expresses each coverage value relative to the
coverage obtained with the *random* score file for the same
run (``coverage_rel_random``):

    coverage_rel_random = coverage / coverage_random

where ``coverage_random`` is the coverage produced by
``data/random.scores.json.gz`` for that particular run.

Output columns (TSV / optional CSV):
    score_file, run_file, |S|, |R|, |S∩R|, coverage, coverage_rel_random

Score files (gzip JSON):
    data/bm25.scores.json.gz
    data/crossencoder.scores.json.gz
    data/random.scores.json.gz
    data/ensemble.all.scores.json.gz

Run files (gzip TREC 6‑column):
    msmarco-passage-trec-dl-2019-judged-bm25.run.gz
    msmarco-passage-trec-dl-2020-judged-bm25.run.gz
"""

import gzip
import json
import argparse
import csv
import itertools
import math
import pathlib
import sys

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

def read_score_docs(path: str) -> set[str]:
    """Return the set of *all* document IDs present in ``path``."""
    docs: set[str] = set()
    with gzip.open(path, "rt") as f:
        data = json.load(f)  # {qid: {docid: score}}

    for qdict in data.values():
        docs.update([str(k) for k in qdict.keys()])

    return docs

def read_run_docs(path: str, topk: int | None = None) -> set[str]:
    """Return the set of *unique* document IDs appearing in ``path``.

    If *topk* is given, only the top‑k documents per query (by score)
    are considered.
    """
    results = pt.io.read_results(path)
    if topk is not None:
        results = (
            results.sort_values(["qid", "score"], ascending=[True, False])
            .groupby("qid")
            .head(topk)
        )
    return {str(d) for d in results.docno.unique().tolist()}

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute coverage for all score/run combinations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=None,
        help="restrict R to top‑k docs per query",
    )
    ap.add_argument(
        "--csv",
        type=str,
        help="optional output CSV file",
    )
    args = ap.parse_args()

    # --- Load all required document sets once (to avoid repeated I/O) --------
    score_sets = {p: read_score_docs(p) for p in SCORE_GZ}
    run_sets: dict[str, set[str]] = {
        p: read_run_docs(p, args.topk) for p in RUN_GZ
    }

    # First pass: compute *all* coverage values, memorising those for the
    # random score file so that we can normalise later.
    rows: list[list] = []  # one entry per SCORE×RUN combination
    random_cov: dict[str, float] = {}  # run_path -> coverage for random

    for score_path, run_path in tqdm(itertools.product(SCORE_GZ, RUN_GZ)):
        S = score_sets[score_path]
        R = run_sets[run_path]

        cov = float("nan") if not R else len(S & R) / len(R)

        # Remember the coverage of the *random* domain for this run.
        if "random" in pathlib.Path(score_path).stem:
            random_cov[run_path] = cov

        rows.append(
            [
                score_path,
                run_path,
                len(S),
                len(R),
                len(S & R),
                cov,  # keep as float for now; format later
            ]
        )

    # Second pass: attach coverage_rel_random to every row.
    for r in rows:
        run_path = r[1]
        cov = r[5]
        rand_cov = random_cov.get(run_path, math.nan)

        if not rand_cov or math.isnan(rand_cov):
            norm = math.nan
        else:
            norm = cov / rand_cov if rand_cov else math.nan

        r.append(norm)  # index 6

    # --- Output ----------------------------------------------------------------
    header = [
        "score_file",
        "run_file",
        "|S|",
        "|R|",
        "|S∩R|",
        "coverage",
        "coverage_rel_random",
    ]

    # TSV to stdout
    print("\t".join(header))
    for r in rows:
        print(
            "\t".join(
                map(
                    str,
                    [
                        r[0],  # score_file
                        r[1],  # run_file
                        r[2],  # |S|
                        r[3],  # |R|
                        r[4],  # |S∩R|
                        f"{r[5]:.4f}",  # coverage
                        f"{r[6]:.4f}",  # coverage_rel_random
                    ],
                )
            )
        )

    # Optional CSV file (comma‑separated, numeric columns unformatted)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows([
                [
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                ]
                for r in rows
            ])
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
