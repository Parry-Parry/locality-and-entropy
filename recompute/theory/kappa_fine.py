#!/usr/bin/env python3
"""A finer estimate of kappa = sup_x dmu/dpi(x), the coverage constant.

The four-bin estimate in rank_bins.py is a LOWER bound: averaging the density
ratio inside a bin can only shrink its supremum. This refines it two ways.

1. Finer partitions. The same ratio computed over 10, 50, 100 and 1000 rank
   cells. As the partition refines, the estimate increases towards the true
   supremum, so the sequence shows how much the coarse estimate understates.

2. The evaluation measure the experiments really use. mu is uniform over BM25's
   top 100 (the rerank depth), so for a partition cell C,

       mu(C) = |C and ranks 1..100| / 100,     pi(C) = miner's share in C.

   A cell with mu > 0 and pi = 0 gives kappa = infinity: the miner never
   samples where evaluation looks, and the coverage assumption of the
   corrected theory fails outright.

CPU only; PISA retrieval is multithreaded but needs no GPU.

Usage:
    python kappa_fine.py --n-queries 2000
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
import pandas as pd

REPO = "/mnt/primary/implicit/sigir25-implicit-diff"
DATA = f"{REPO}/data"
OUT = f"{REPO}/recompute/theory"
PISA_INDEX = "/mnt/indices/msmarco-passage.pisa"

MINERS = ["random", "bm25", "crossencoder", "ensemble"]
SCORE_FILE = {m: f"{DATA}/{m}.scores.json.gz" for m in MINERS}
SCORE_FILE["bm25"] = f"{DATA}/bm25.crossencoder.scores.json.gz"
SCORE_FILE["ensemble"] = f"{DATA}/ensemble.all.scores.json.gz"

SEED = 20260101
DEPTH = 1000
EVAL_DEPTH = 100
# number of equal-width rank cells over 1..DEPTH, plus one cell for "absent"
GRANULARITIES = [4, 10, 50, 100, 1000]


def cells_for(n_cells: int) -> np.ndarray:
    """Upper rank edge of each cell; last cell is 'beyond DEPTH'."""
    if n_cells == 4:                      # the original bins
        return np.array([10, 100, 1000])
    return np.linspace(DEPTH / n_cells, DEPTH, n_cells)


def cell_index(ranks: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Index of the cell each rank falls in; len(edges) means absent."""
    return np.searchsorted(edges, ranks, side="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=2000)
    args = ap.parse_args()

    import ir_datasets as irds
    import pandas as pd
    from pyterrier_pisa import PisaIndex

    rng = np.random.default_rng(SEED)

    print("selecting query sample ...")
    common = None
    for m in MINERS:
        with gzip.open(SCORE_FILE[m], "rt") as fh:
            qids = set(json.load(fh).keys())
        common = qids if common is None else (common & qids)
    sample = sorted(common)
    if len(sample) > args.n_queries:
        idx = rng.choice(len(sample), size=args.n_queries, replace=False)
        sample = [sample[i] for i in sorted(idx)]
    wanted = set(sample)
    print(f"{len(sample)} queries from {len(common)} common")

    ds = irds.load("msmarco-passage/train/triples-small")
    qtext = {}
    for q in ds.queries_iter():
        if q.query_id in wanted:
            qtext[q.query_id] = q.text
            if len(qtext) == len(wanted):
                break

    print("BM25 top-1000 ...")
    bm25 = PisaIndex(PISA_INDEX, threads=16).bm25(
        k1=1.2, b=0.75, num_results=DEPTH, verbose=True
    )
    res = bm25.transform(
        pd.DataFrame([{"qid": q, "query": t} for q, t in sorted(qtext.items())])
    )
    ranks = {}
    for qid, grp in res.groupby("qid"):
        ranks[qid] = dict(zip(grp["docno"].astype(str), grp["rank"] + 1))

    print("loading miner negatives ...")
    negs = {}
    for m in MINERS:
        g = {}
        with gzip.open(f"{DATA}/{m}.16.jsonl.gz", "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                q = str(o["query_id"])
                if q in wanted and q not in g:
                    g[q] = [str(d) for d in o.get("doc_id_b", [])]
                    if len(g) == len(wanted):
                        break
        negs[m] = g
        print(f"  {m}: {len(g)} queries")

    rows = []
    for n_cells in GRANULARITIES:
        edges = cells_for(n_cells)
        n_total = len(edges) + 1          # +1 for the 'absent' cell

        # mu: uniform over BM25 top-EVAL_DEPTH, spread across the cells
        eval_ranks = np.arange(1, EVAL_DEPTH + 1)
        mu_counts = np.bincount(cell_index(eval_ranks, edges), minlength=n_total)
        mu = mu_counts / mu_counts.sum()

        for m in MINERS:
            pi_counts = np.zeros(n_total, dtype=np.int64)
            for q, docs in negs[m].items():
                rmap = ranks.get(q, {})
                r = np.array([rmap.get(d, DEPTH + 1) for d in docs])
                pi_counts += np.bincount(cell_index(r, edges), minlength=n_total)
            pi = pi_counts / pi_counts.sum()

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(pi > 0, mu / np.where(pi > 0, pi, 1.0), np.inf)
            ratio = np.where((mu == 0) & (pi == 0), 0.0, ratio)
            k = float(np.max(ratio))
            uncovered = float(mu[(pi == 0) & (mu > 0)].sum())
            rows.append({
                "miner": m,
                "n_cells": n_total,
                "kappa": k,
                "tv": float(0.5 * np.abs(mu - pi).sum()),
                "mu_mass_uncovered": uncovered,
                "bias_over_B": (float("inf") if not np.isfinite(k)
                                else 2.0 * (1.0 - 1.0 / k**2)),
            })
        print(f"  partition with {n_total} cells done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/kappa_fine.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== kappa against mu = uniform over BM25 top-100 ===")
    print(df.pivot(index="miner", columns="n_cells", values="kappa")
            .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n=== mu mass the miner never samples (coverage failure) ===")
    print(df.pivot(index="miner", columns="n_cells",
                   values="mu_mass_uncovered")
            .to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {OUT}/kappa_fine.csv")
    print("\nkappa should increase with the number of cells; the coarse")
    print("four-bin figure is a lower bound on the true supremum.")


if __name__ == "__main__":
    main()
