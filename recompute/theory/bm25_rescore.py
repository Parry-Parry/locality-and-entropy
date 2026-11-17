#!/usr/bin/env python3
"""Item 1 arm A for the BM25 miner, re-scored with the SURVIVING teacher.

data/bm25.scores.json.gz does not come from any surviving teacher checkpoint
(teacher_matrix.csv: mean |diff| 46.6 against all of them, while random,
crossencoder and ensemble reproduce exactly), and the disagreement is not a
constant offset (bm25_offset.csv: within-query Spearman median 0.41).

So the arm A entropy figure for BM25 (0.4292 nats) is computed from a
different teacher than the other three miners, and the cross-miner comparison
is confounded at exactly the point H1.2 cares about.

This re-scores the BM25 miner's own training negatives with the surviving
teacher and recomputes the entropy, putting all four miners on one teacher.

Two numbers result, and they answer different questions:

  as-stored   what the BM25 students were actually distilled from
  re-scored   what the comparison with the other three miners should use

Usage:
    python bm25_rescore.py --n-queries 20000
"""

from __future__ import annotations

import argparse
import gzip
import json
import math

import numpy as np
import pandas as pd

REPO = "/mnt/primary/implicit/sigir25-implicit-diff"
OUT = f"{REPO}/recompute/theory"
TEACHER = f"{REPO}/checkpoints/crossencoder"
SEED = 20260101
MAX_PAIRS = 200
LOG2 = math.log(2.0)


def binary_entropy_nats(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def entropy_of(vals: np.ndarray, rng) -> float:
    k = vals.size
    if k < 2:
        return float("nan")
    i, j = np.triu_indices(k, k=1)
    if i.size > MAX_PAIRS:
        sel = rng.choice(i.size, size=MAX_PAIRS, replace=False)
        i, j = i[sel], j[sel]
    return float(
        binary_entropy_nats(1.0 / (1.0 + np.exp(-(vals[i] - vals[j])))).mean()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import ir_datasets as irds
    from rankers import CatTransformer

    rng = np.random.default_rng(SEED)

    # one group per query (groups repeat ~100x in the file)
    print("reading BM25 training negatives ...")
    groups: dict[str, list[str]] = {}
    with gzip.open(f"{REPO}/data/bm25.16.jsonl.gz", "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            q = str(o["query_id"])
            if q not in groups:
                groups[q] = [str(d) for d in o.get("doc_id_b", [])]
    print(f"{len(groups)} distinct queries")

    qids = sorted(groups)
    if len(qids) > args.n_queries:
        idx = rng.choice(len(qids), size=args.n_queries, replace=False)
        qids = [qids[i] for i in sorted(idx)]
    groups = {q: groups[q] for q in qids}
    print(f"sampled {len(groups)} queries (seed {SEED})")

    with gzip.open(f"{REPO}/data/bm25.scores.json.gz", "rt") as fh:
        stored = json.load(fh)

    ds = irds.load("msmarco-passage/train/triples-small")
    want_q = set(groups)
    qtext = {}
    for q in ds.queries_iter():
        if q.query_id in want_q:
            qtext[q.query_id] = q.text
            if len(qtext) == len(want_q):
                break
    want_d = {d for v in groups.values() for d in v}
    print(f"resolving {len(want_d)} document texts ...")
    dtext = {}
    for doc in irds.load("msmarco-passage").docs_iter():
        if doc.doc_id in want_d:
            dtext[doc.doc_id] = doc.text
            if len(dtext) == len(want_d):
                break
    print(f"resolved {len(dtext)}/{len(want_d)}")

    model = CatTransformer.from_pretrained(
        TEACHER, batch_size=args.batch_size, verbose=True
    )
    frame = pd.DataFrame(
        [{"qid": q, "docno": d, "query": qtext[q], "text": dtext.get(d, "")}
         for q, docs in groups.items() for d in docs
         if q in qtext and d in dtext]
    )
    print(f"scoring {len(frame)} (query, doc) pairs with the surviving teacher ...")
    res = model.transform(frame)
    got: dict[str, dict[str, float]] = {}
    for q, d, s in zip(res.qid, res.docno, res.score):
        got.setdefault(q, {})[d] = float(s)

    rows = []
    for q, docs in groups.items():
        new = np.array([got[q][d] for d in docs if q in got and d in got.get(q, {})],
                       dtype=float)
        old = np.array([stored[q][d] for d in docs
                        if q in stored and d in stored.get(q, {})], dtype=float)
        if new.size < 2 or old.size < 2:
            continue
        rows.append({
            "qid": q,
            "h_rescored": entropy_of(new, rng),
            "h_as_stored": entropy_of(old, rng),
        })

    df = pd.DataFrame(rows).dropna()
    df.to_csv(f"{OUT}/bm25_rescore_perquery.csv", index=False)

    out = []
    for col, label in [("h_as_stored", "as stored (lost teacher)"),
                       ("h_rescored", "re-scored (surviving teacher)")]:
        a = df[col].to_numpy()
        out.append({
            "variant": label,
            "n_queries": a.size,
            "mean_entropy_nats": a.mean(),
            "se": a.std(ddof=1) / math.sqrt(a.size),
            "sd": a.std(ddof=1),
            "median": float(np.median(a)),
            "frac_of_log2": a.mean() / LOG2,
        })
    res_df = pd.DataFrame(out)
    res_df.to_csv(f"{OUT}/bm25_rescore.csv", index=False)

    print("\n=== BM25 miner, arm A, both teachers ===")
    print(res_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nfor reference, the other three miners (surviving teacher):")
    print("  random 0.6596   crossencoder 0.2167   ensemble 0.1794")
    print(f"\nwrote {OUT}/bm25_rescore.csv")


if __name__ == "__main__":
    main()
