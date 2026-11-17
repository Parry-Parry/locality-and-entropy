#!/usr/bin/env python3
"""P2 item 1: pairwise teacher entropy (Bradley-Terry).

For each training query, sample up to N pairs of candidates and compute

    h( sigma( g(x) - g(x') ) )          in nats

where g is the teacher score, sigma the logistic function and h the binary
entropy h(p) = -p log p - (1-p) log(1-p). Averaged over pairs and queries.

This measures how decisively the teacher separates the pairs a miner actually
trains on: h -> 0 when the teacher is certain which of the two is better,
h -> log 2 = 0.693 when it cannot tell them apart.

ARM A (per miner, implemented here): pairs drawn from that miner's actual
training negatives, read from data/<miner>.16.jsonl.gz and scored from the
precomputed data/<miner>.scores.json.gz. No GPU and no teacher inference is
needed -- the teacher scores already exist.

NOTE ON THE TRAINING FILES. Each query's group is repeated roughly 90 to 110
times in <miner>.16.jsonl.gz, with an identical positive and an identical set
of negatives each time (verified: for qid 400296, 90 instances whose negative
sets union to exactly the 15 negatives). The unit of analysis here is the
QUERY, so only the first instance per query is used. Averaging over raw lines
instead would weight each query by its (arbitrary) repeat count and understate
the standard error by a factor of about sqrt(100).

ARM B (uniform from BM25's top 1,000): NOT computed. The repository mines BM25
candidates on the fly from a PISA index over MSMARCO v1 passage
(src/implicit/data_processing/mine_negatives.py), and no such index and no
stored BM25 top-1000 run over the 400k training queries is available; the only
PISA index on the cluster is msmarco-passage-v2-dedup, a different corpus.
Building a v1 index is the outstanding dependency for arm B.

Note on the teacher scores: they are log-softmax outputs (all negative, roughly
-14 to -39), i.e. log P(relevant). Their differences are therefore log-odds
differences, which is exactly the quantity the Bradley-Terry model expects, so
sigma(g(x) - g(x')) is well defined.

Usage:
    python recompute/theory/pairwise_entropy.py --miner random
    python recompute/theory/pairwise_entropy.py --all
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEG = os.path.join(HERE, "negatives")

MINERS = ["random", "bm25", "crossencoder", "ensemble"]

# data/<miner>.scores.json.gz, except ensemble which get_kappa.py reads from
# ensemble.all.scores.json.gz
SCORE_FILE = {
    "random": "random.scores.json.gz",
    "bm25": "bm25.scores.json.gz",
    "crossencoder": "crossencoder.scores.json.gz",
    "ensemble": "ensemble.all.scores.json.gz",
}
TRIPLE_FILE = {m: f"{m}.16.jsonl.gz" for m in MINERS}

MAX_PAIRS = 200
SEED = 20260101
LOG2 = math.log(2.0)


def binary_entropy_nats(p: np.ndarray) -> np.ndarray:
    """h(p) in nats, with the 0 log 0 = 0 convention."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def pair_entropy(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """h(sigma(g_i - g_j)) over up to MAX_PAIRS distinct unordered pairs."""
    k = scores.size
    if k < 2:
        return np.empty(0)

    n_all = k * (k - 1) // 2
    i_all, j_all = np.triu_indices(k, k=1)
    if n_all > MAX_PAIRS:
        sel = rng.choice(n_all, size=MAX_PAIRS, replace=False)
        i_all, j_all = i_all[sel], j_all[sel]

    diff = scores[i_all] - scores[j_all]
    return binary_entropy_nats(1.0 / (1.0 + np.exp(-diff)))


def run_miner(miner: str) -> dict:
    score_path = os.path.join(NEG, SCORE_FILE[miner])
    triple_path = os.path.join(NEG, TRIPLE_FILE[miner])
    for p in (score_path, triple_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- fetch it first")

    print(f"[{miner}] loading teacher scores ...")
    with gzip.open(score_path, "rt", encoding="utf-8") as fh:
        scores = json.load(fh)

    rng = np.random.default_rng(SEED)
    per_query_mean = []
    n_skipped = 0
    n_lines = 0
    seen = set()

    print(f"[{miner}] streaming training negatives ...")
    with gzip.open(triple_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            obj = json.loads(line)
            qid = str(obj["query_id"])
            if qid in seen:          # groups repeat ~100x; one per query
                continue
            seen.add(qid)
            lookup = scores.get(qid)
            if lookup is None:
                n_skipped += 1
                continue

            # NEGATIVES ONLY: doc_id_a is the positive and is excluded
            negs = [str(d) for d in obj.get("doc_id_b", [])]
            vals = [lookup[d] for d in negs if d in lookup]
            if len(vals) < 2:
                n_skipped += 1
                continue

            h = pair_entropy(np.asarray(vals, dtype=float), rng)
            if h.size:
                per_query_mean.append(h.mean())

    arr = np.asarray(per_query_mean, dtype=float)
    return {
        "miner": miner,
        "n_queries": arr.size,
        "n_lines": n_lines,
        "repeat_factor": n_lines / max(len(seen), 1),
        "n_skipped": n_skipped,
        "mean_entropy_nats": arr.mean(),
        "se": arr.std(ddof=1) / math.sqrt(arr.size),
        "sd": arr.std(ddof=1),
        "median": float(np.median(arr)),
        "frac_of_log2": arr.mean() / LOG2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--miner", choices=MINERS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    targets = MINERS if args.all else [args.miner]
    if not targets or targets == [None]:
        raise SystemExit("pass --miner <name> or --all")

    out_csv = os.path.join(HERE, "pairwise_entropy.csv")
    rows = []
    if os.path.exists(out_csv):
        rows = pd.read_csv(out_csv).to_dict("records")

    for miner in targets:
        res = run_miner(miner)
        rows = [r for r in rows if r["miner"] != miner] + [res]
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(
            f"[{miner}] mean h = {res['mean_entropy_nats']:.4f} nats "
            f"+/- {res['se']:.5f} (se), n = {res['n_queries']}, "
            f"{100 * res['frac_of_log2']:.1f}% of log 2"
        )

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nmaximum possible entropy: log 2 = {LOG2:.4f} nats")
    print(f"wrote {os.path.relpath(out_csv, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()
