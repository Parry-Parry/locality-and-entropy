#!/usr/bin/env python3
"""P2 item 2: rank-bin mismatch between each miner's negatives and BM25.

For each miner, take its training negatives, find each negative's rank in
BM25's top 1,000 for that query (1001+ if absent), bin as 1-10, 11-100,
101-1000, >1000, and report:

    kappa = max over bins of mu_bin / pi_bin
    TV    = 1/2 sum |mu_bin - pi_bin|

for two target distributions mu: uniform over the four bins, and top-weighted
(0.4, 0.3, 0.2, 0.1). pi is the miner's empirical bin distribution.

BM25 is the same index and parameters the mining used
(src/implicit/data_processing/mine_negatives.py: PisaIndex ... .bm25(k1=1.2,
b=0.75)), so the ranks are the ones the original pipeline saw.

Queries are sampled (--n-queries) because retrieving top-1000 for all 398k
training queries is unnecessary for a distributional estimate; the sample is
drawn with a fixed seed and the count is reported.

Usage:
    python rank_bins.py --n-queries 2000
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
BINS = ["1-10", "11-100", "101-1000", ">1000"]
MU = {
    "uniform": np.array([0.25, 0.25, 0.25, 0.25]),
    "top_weighted": np.array([0.4, 0.3, 0.2, 0.1]),
}
SEED = 20260101
DEPTH = 1000


def bin_of(rank: int) -> int:
    if rank <= 10:
        return 0
    if rank <= 100:
        return 1
    if rank <= 1000:
        return 2
    return 3


def load_miner_negatives(miner: str, wanted: set[str]) -> dict[str, list[str]]:
    """qid -> negatives, one group per query (groups repeat ~100x in file)."""
    path = f"{DATA}/{miner}.16.jsonl.gz"
    out: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj["query_id"])
            if qid not in wanted or qid in out:
                continue
            out[qid] = [str(d) for d in obj.get("doc_id_b", [])]
            if len(out) == len(wanted):
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=2000)
    args = ap.parse_args()

    import ir_datasets as irds
    import pandas as pd
    from pyterrier_pisa import PisaIndex

    # ---- choose the query sample -------------------------------------
    # queries common to all miners, so the comparison is like-for-like
    print("selecting query sample ...")
    common: set[str] | None = None
    for miner in MINERS:
        with gzip.open(f"{DATA}/{miner}.scores.json.gz"
                       if miner != "ensemble"
                       else f"{DATA}/ensemble.all.scores.json.gz",
                       "rt", encoding="utf-8") as fh:
            qids = set(json.load(fh).keys())
        common = qids if common is None else (common & qids)
        print(f"  {miner}: {len(qids)} queries, common so far {len(common)}")

    rng = np.random.default_rng(SEED)
    sample = sorted(common)
    if len(sample) > args.n_queries:
        idx = rng.choice(len(sample), size=args.n_queries, replace=False)
        sample = [sample[i] for i in sorted(idx)]
    wanted = set(sample)
    print(f"sampled {len(sample)} queries (seed {SEED})")

    # ---- query text --------------------------------------------------
    ds = irds.load("msmarco-passage/train/triples-small")
    qtext = {}
    for q in ds.queries_iter():
        if q.query_id in wanted:
            qtext[q.query_id] = q.text
            if len(qtext) == len(wanted):
                break
    print(f"resolved {len(qtext)} query texts")

    # ---- BM25 top-1000 ------------------------------------------------
    print(f"retrieving BM25 top-{DEPTH} from {PISA_INDEX} ...")
    index = PisaIndex(PISA_INDEX, threads=16)
    bm25 = index.bm25(k1=1.2, b=0.75, num_results=DEPTH, verbose=True)
    topics = pd.DataFrame(
        [{"qid": q, "query": t} for q, t in sorted(qtext.items())]
    )
    res = bm25.transform(topics)
    print(f"retrieved {len(res)} rows for {res.qid.nunique()} queries")

    # qid -> {docno: rank}; pyterrier ranks are 0-based
    ranks: dict[str, dict[str, int]] = {}
    for qid, grp in res.groupby("qid"):
        ranks[qid] = dict(zip(grp["docno"].astype(str), grp["rank"] + 1))

    # ---- bin each miner's negatives ----------------------------------
    rows, dists = [], {}
    for miner in MINERS:
        print(f"[{miner}] loading negatives ...")
        negs = load_miner_negatives(miner, wanted)
        counts = np.zeros(4, dtype=np.int64)
        n_negs = 0
        for qid, docs in negs.items():
            rmap = ranks.get(qid, {})
            for d in docs:
                counts[bin_of(rmap.get(d, DEPTH + 1))] += 1
                n_negs += 1
        pi = counts / counts.sum()
        dists[miner] = pi

        row = {
            "miner": miner,
            "n_queries": len(negs),
            "n_negatives": n_negs,
        }
        for b, c, p in zip(BINS, counts, pi):
            row[f"count_{b}"] = int(c)
            row[f"frac_{b}"] = p
        for name, mu in MU.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(pi > 0, mu / pi, np.inf)
            row[f"kappa_{name}"] = float(np.max(ratio))
            row[f"tv_{name}"] = float(0.5 * np.abs(mu - pi).sum())
        rows.append(row)
        print(f"[{miner}] bins {dict(zip(BINS, counts))}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/rank_bins_bm25.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== bin distribution (fraction of negatives) ===")
    print(df[["miner", "n_queries", "n_negatives"]
             + [f"frac_{b}" for b in BINS]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== divergence from target distributions ===")
    print(df[["miner", "kappa_uniform", "tv_uniform",
              "kappa_top_weighted", "tv_top_weighted"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {OUT}/rank_bins_bm25.csv")


if __name__ == "__main__":
    main()
