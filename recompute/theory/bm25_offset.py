#!/usr/bin/env python3
"""Is data/bm25.scores.json.gz a constant offset from the surviving teacher?

The teacher matrix shows bm25.scores.json.gz disagrees with every surviving
teacher by a mean absolute difference of ~46.6, while random, crossencoder and
ensemble agree exactly. Whether that matters depends entirely on the STRUCTURE
of the disagreement:

  * If the difference is a constant (global, or per query), then MarginMSE
    targets (which use residuals s_pos - s_neg) and KL targets (softmax over
    the group) are UNCHANGED, because both cancel additive constants. The BM25
    arm would then be comparable with the others after all.

  * If the difference varies within a query, the targets genuinely differ and
    the BM25 arm of Table 4.1 was distilled from different supervision than
    the other three arms.

Also reports rank correlation within each query, which bounds how much the
ordering itself changed.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
from scipy import stats

REPO = "/mnt/primary/implicit/sigir25-implicit-diff"
OUT = f"{REPO}/recompute/theory"
TEACHER = f"{REPO}/checkpoints/crossencoder"
N_QUERIES = 300


def main() -> None:
    import ir_datasets as irds
    from rankers import CatTransformer

    with gzip.open(f"{REPO}/data/bm25.scores.json.gz", "rt") as fh:
        stored = json.load(fh)
    qids = sorted(stored.keys())[:N_QUERIES]
    trips = [(q, d, stored[q][d]) for q in qids for d in stored[q]]
    print(f"{len(trips)} pairs from {len(qids)} queries")

    ds = irds.load("msmarco-passage/train/triples-small")
    want_q = {q for q, _, _ in trips}
    qtext = {}
    for q in ds.queries_iter():
        if q.query_id in want_q:
            qtext[q.query_id] = q.text
            if len(qtext) == len(want_q):
                break
    want_d = {d for _, d, _ in trips}
    dtext = {}
    for doc in irds.load("msmarco-passage").docs_iter():
        if doc.doc_id in want_d:
            dtext[doc.doc_id] = doc.text
            if len(dtext) == len(want_d):
                break
    trips = [(q, d, s) for q, d, s in trips if q in qtext and d in dtext]
    print(f"resolved text for {len(trips)} pairs")

    model = CatTransformer.from_pretrained(TEACHER, batch_size=256, verbose=True)
    frame = pd.DataFrame(
        [{"qid": q, "docno": d, "query": qtext[q], "text": dtext[d]}
         for q, d, _ in trips]
    )
    res = model.transform(frame)
    got = dict(zip(zip(res.qid, res.docno), res.score))

    rows = []
    for q in sorted({q for q, _, _ in trips}):
        items = [(d, s) for qq, d, s in trips if qq == q]
        if len(items) < 3:
            continue
        old = np.array([s for _, s in items], dtype=float)
        new = np.array([got[(q, d)] for d, _ in items], dtype=float)
        diff = old - new
        rows.append({
            "qid": q,
            "n_docs": len(items),
            "diff_mean": diff.mean(),
            "diff_sd": diff.std(ddof=1),
            "diff_range": diff.max() - diff.min(),
            "spearman": stats.spearmanr(old, new).statistic,
            "pearson": stats.pearsonr(old, new).statistic,
            # do residuals (MarginMSE targets) survive?
            "resid_max_abs_diff": float(
                np.abs((old - old[0]) - (new - new[0])).max()
            ),
            # does the softmax (KL target) survive?
            "softmax_max_abs_diff": float(
                np.abs(
                    np.exp(old - old.max()) / np.exp(old - old.max()).sum()
                    - np.exp(new - new.max()) / np.exp(new - new.max()).sum()
                ).max()
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/bm25_offset.csv", index=False)

    print("\n=== structure of the disagreement ===")
    print(f"per-query offset (old - new): mean {df.diff_mean.mean():.4f}, "
          f"sd across queries {df.diff_mean.std(ddof=1):.4f}")
    print(f"WITHIN-query sd of the offset: median {df.diff_sd.median():.6f}, "
          f"max {df.diff_sd.max():.6f}")
    print(f"within-query range of offset : median {df.diff_range.median():.6f}, "
          f"max {df.diff_range.max():.6f}")
    print()
    print(f"Spearman(old, new) within query: median {df.spearman.median():.6f}, "
          f"min {df.spearman.min():.6f}")
    print(f"Pearson (old, new) within query: median {df.pearson.median():.6f}, "
          f"min {df.pearson.min():.6f}")
    print()
    print("Do the training targets actually change?")
    print(f"  MarginMSE residuals, max |diff|: median "
          f"{df.resid_max_abs_diff.median():.6f}, max "
          f"{df.resid_max_abs_diff.max():.6f}")
    print(f"  KL softmax probs,    max |diff|: median "
          f"{df.softmax_max_abs_diff.median():.6f}, max "
          f"{df.softmax_max_abs_diff.max():.6f}")

    constant = df.diff_sd.max() < 1e-4
    print()
    print("VERDICT: " + (
        "constant offset -- MarginMSE and KL targets are UNAFFECTED"
        if constant else
        "NOT a constant offset -- the targets genuinely differ"
    ))
    print(f"wrote {OUT}/bm25_offset.csv")


if __name__ == "__main__":
    main()
