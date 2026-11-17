#!/usr/bin/env python3
"""P2 item 1 arm B: pairwise teacher entropy under UNIFORM sampling from BM25 top-1000.

Arm A (pairwise_entropy.py) used each miner's own negatives with the teacher
scores already stored in data/<miner>.scores.json.gz. Arm B needs teacher
scores for documents nobody scored before, so it runs the teacher itself.

Method
------
* Same 2,000-query sample as rank_bins.py (seed 20260101, queries common to all
  four miners), so the two items are directly comparable.
* BM25 top-1000 from the index and parameters the mining used:
  PisaIndex(/mnt/indices/msmarco-passage.pisa).bm25(k1=1.2, b=0.75).
* Per query, sample K documents uniformly without replacement from the top
  1,000, score them with the teacher, and take up to 200 of the pairs among
  them.

  Sampling K docs then pairing is equivalent to sampling pairs directly: every
  unordered pair has the same inclusion probability K(K-1)/(1000*999), so the
  pairs are a uniform sample of all C(1000,2) pairs, and subsampling 200 of
  them uniformly preserves that. It costs K teacher scorings per query instead
  of up to 400.

* h(sigma(g(x) - g(x'))) in nats, averaged within a query then across queries.

The teacher is scored through rankers' CatTransformer, the same class the
original pipeline used, so the scores are on the same log_softmax(logits)[:, 1]
scale as the stored ones. --validate re-scores documents that DO have stored
scores and reports the agreement, which checks that claim rather than assuming
it.

Usage:
    python entropy_armb.py --n-queries 2000 --k-docs 64 --validate
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os

import numpy as np
import pandas as pd

REPO = "/mnt/primary/implicit/sigir25-implicit-diff"
DATA = f"{REPO}/data"
OUT = f"{REPO}/recompute/theory"
PISA_INDEX = "/mnt/indices/msmarco-passage.pisa"
TEACHER = f"{REPO}/checkpoints/crossencoder"

MINERS = ["random", "bm25", "crossencoder", "ensemble"]
SCORE_FILE = {m: f"{DATA}/{m}.scores.json.gz" for m in MINERS}
SCORE_FILE["ensemble"] = f"{DATA}/ensemble.all.scores.json.gz"

SEED = 20260101
DEPTH = 1000
MAX_PAIRS = 200
LOG2 = math.log(2.0)


def binary_entropy_nats(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--k-docs", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    import ir_datasets as irds
    import pyterrier as pt
    from pyterrier_pisa import PisaIndex
    from rankers import CatTransformer

    rng = np.random.default_rng(SEED)

    # ---- same query sample as item 2 ---------------------------------
    print("selecting query sample ...")
    common = None
    for m in MINERS:
        with gzip.open(SCORE_FILE[m], "rt", encoding="utf-8") as fh:
            qids = set(json.load(fh).keys())
        common = qids if common is None else (common & qids)
    sample = sorted(common)
    if len(sample) > args.n_queries:
        idx = rng.choice(len(sample), size=args.n_queries, replace=False)
        sample = [sample[i] for i in sorted(idx)]
    wanted = set(sample)
    print(f"sampled {len(sample)} of {len(common)} common queries (seed {SEED})")

    # ---- query text ---------------------------------------------------
    ds = irds.load("msmarco-passage/train/triples-small")
    qtext = {}
    for q in ds.queries_iter():
        if q.query_id in wanted:
            qtext[q.query_id] = q.text
            if len(qtext) == len(wanted):
                break
    print(f"resolved {len(qtext)} query texts")

    # ---- BM25 top-1000 -------------------------------------------------
    print("retrieving BM25 top-1000 ...")
    bm25 = PisaIndex(PISA_INDEX, threads=16).bm25(
        k1=1.2, b=0.75, num_results=DEPTH, verbose=True
    )
    res = bm25.transform(
        pd.DataFrame([{"qid": q, "query": t} for q, t in sorted(qtext.items())])
    )
    print(f"retrieved {len(res)} rows / {res.qid.nunique()} queries")

    # ---- sample K docs per query ---------------------------------------
    picks: dict[str, list[str]] = {}
    for qid, grp in res.groupby("qid"):
        docs = grp.sort_values("rank")["docno"].astype(str).tolist()
        k = min(args.k_docs, len(docs))
        if k < 2:
            continue
        sel = rng.choice(len(docs), size=k, replace=False)
        picks[qid] = [docs[i] for i in sel]
    print(f"{len(picks)} queries x up to {args.k_docs} docs to score")

    # ---- document text --------------------------------------------------
    need = {d for ds_ in picks.values() for d in ds_}
    if args.validate:
        with gzip.open(SCORE_FILE["bm25"], "rt", encoding="utf-8") as fh:
            stored_all = json.load(fh)
        val_q = sample[:200]
        stored = {q: stored_all[q] for q in val_q if q in stored_all}
        for q, lk in stored.items():
            need |= set(lk.keys())
        print(f"validation adds {len(stored)} queries' stored docs")
    else:
        stored = {}

    print(f"resolving text for {len(need)} documents ...")
    corpus = irds.load("msmarco-passage")
    texts = {}
    for doc in corpus.docs_iter():
        if doc.doc_id in need:
            texts[doc.doc_id] = doc.text
            if len(texts) == len(need):
                break
    print(f"resolved {len(texts)} / {len(need)}")

    # ---- teacher ---------------------------------------------------------
    print(f"loading teacher from {TEACHER} ...")
    teacher = CatTransformer.from_pretrained(
        TEACHER, batch_size=args.batch_size, verbose=True
    )

    def score(pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
        """pairs of (qid, docno, -) -> scored frame."""
        frame = pd.DataFrame(
            [
                {"qid": q, "docno": d, "query": qtext[q],
                 "text": texts.get(d, "")}
                for q, d in pairs
            ]
        )
        return teacher.transform(frame)

    # ---- validation: do we reproduce the stored teacher scores? ---------
    if stored:
        pairs = [(q, d) for q, lk in stored.items() for d in lk]
        pairs = [(q, d) for q, d in pairs if d in texts]
        print(f"\nvalidating on {len(pairs)} stored (query, doc) scores ...")
        sc = score(pairs)
        got = dict(zip(zip(sc.qid, sc.docno), sc.score))
        diffs = [
            abs(got[(q, d)] - stored[q][d])
            for q, d in pairs
            if (q, d) in got
        ]
        diffs = np.asarray(diffs)
        print(f"  n={diffs.size} mean|diff|={diffs.mean():.6f} "
              f"max|diff|={diffs.max():.6f} "
              f"frac>0.01={np.mean(diffs > 0.01):.4f}")
        pd.DataFrame({"abs_diff": diffs}).to_csv(
            f"{OUT}/armb_teacher_validation.csv", index=False)

    # ---- arm B ----------------------------------------------------------
    print("\nscoring BM25 top-1000 samples ...")
    all_pairs = [(q, d) for q, docs in picks.items() for d in docs]
    sc = score(all_pairs)
    by_q: dict[str, list[float]] = {}
    for q, s in zip(sc.qid, sc.score):
        by_q.setdefault(q, []).append(float(s))

    per_query = []
    for q, vals in by_q.items():
        a = np.asarray(vals, dtype=float)
        k = a.size
        if k < 2:
            continue
        i, j = np.triu_indices(k, k=1)
        n_all = i.size
        if n_all > MAX_PAIRS:
            sel = rng.choice(n_all, size=MAX_PAIRS, replace=False)
            i, j = i[sel], j[sel]
        h = binary_entropy_nats(1.0 / (1.0 + np.exp(-(a[i] - a[j]))))
        per_query.append(h.mean())

    arr = np.asarray(per_query, dtype=float)
    row = {
        "arm": "B_uniform_bm25_top1000",
        "n_queries": arr.size,
        "k_docs_per_query": args.k_docs,
        "mean_entropy_nats": arr.mean(),
        "se": arr.std(ddof=1) / math.sqrt(arr.size),
        "sd": arr.std(ddof=1),
        "median": float(np.median(arr)),
        "frac_of_log2": arr.mean() / LOG2,
    }
    pd.DataFrame([row]).to_csv(f"{OUT}/pairwise_entropy_armB.csv", index=False)

    print("\n=== ARM B: uniform pairs from BM25 top-1000 ===")
    for k, v in row.items():
        print(f"  {k:22s} {v}")
    print(f"\nmax possible = log 2 = {LOG2:.4f} nats")
    print(f"wrote {OUT}/pairwise_entropy_armB.csv")


if __name__ == "__main__":
    main()
