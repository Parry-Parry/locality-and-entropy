#!/usr/bin/env python3
"""Which model produced data/*.scores.json.gz?

Scoring data/bm25.scores.json.gz's own (query, doc) pairs with
checkpoints/crossencoder gives a mean absolute difference of ~50, so the stored
scores did not come from that model as loaded. This tries each candidate and
reports which, if any, reproduces them.

Candidates: the teacher's top-level save, each of its intermediate
checkpoint-* directories, and the public cross-encoder whose name appears in
data/ (cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz).
"""

from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

REPO = "/mnt/primary/implicit/sigir25-implicit-diff"
TEACHER = f"{REPO}/checkpoints/crossencoder"
N_QUERIES = 25


def main() -> None:
    import ir_datasets as irds
    from rankers import CatTransformer

    with gzip.open(f"{REPO}/data/bm25.scores.json.gz", "rt") as fh:
        stored = json.load(fh)

    qids = sorted(stored.keys())[:N_QUERIES]
    pairs = [(q, d, stored[q][d]) for q in qids for d in stored[q]]
    print(f"{len(pairs)} stored (query, doc, score) triples from {len(qids)} queries")
    arr = np.array([p[2] for p in pairs])
    print(f"stored score range: {arr.min():.4f} to {arr.max():.4f}, "
          f"mean {arr.mean():.4f}")

    ds = irds.load("msmarco-passage/train/triples-small")
    want_q = {q for q, _, _ in pairs}
    qtext = {}
    for q in ds.queries_iter():
        if q.query_id in want_q:
            qtext[q.query_id] = q.text
            if len(qtext) == len(want_q):
                break

    want_d = {d for _, d, _ in pairs}
    corpus = irds.load("msmarco-passage")
    dtext = {}
    for doc in corpus.docs_iter():
        if doc.doc_id in want_d:
            dtext[doc.doc_id] = doc.text
            if len(dtext) == len(want_d):
                break
    print(f"resolved {len(qtext)} queries, {len(dtext)} docs")

    frame = pd.DataFrame(
        [{"qid": q, "docno": d, "query": qtext[q], "text": dtext.get(d, "")}
         for q, d, _ in pairs]
    )
    truth = {(q, d): s for q, d, s in pairs}

    candidates = [("top-level", TEACHER)]
    for p in sorted(glob.glob(f"{TEACHER}/checkpoint-*")):
        if os.path.exists(f"{p}/config.json"):
            candidates.append((os.path.basename(p), p))
    candidates.append(
        ("public MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    rows = []
    for label, path in candidates:
        try:
            model = CatTransformer.from_pretrained(
                path, batch_size=256, verbose=False
            )
            res = model.transform(frame.copy())
            got = dict(zip(zip(res.qid, res.docno), res.score))
            diffs = np.array([
                abs(got[(q, d)] - truth[(q, d)])
                for q, d, _ in pairs if (q, d) in got
            ])
            vals = np.array([got[(q, d)] for q, d, _ in pairs if (q, d) in got])
            rows.append({
                "candidate": label,
                "n": diffs.size,
                "mean_abs_diff": diffs.mean(),
                "max_abs_diff": diffs.max(),
                "frac_within_0.01": float(np.mean(diffs < 0.01)),
                "score_min": vals.min(),
                "score_max": vals.max(),
            })
            print(f"  {label:24s} mean|diff|={diffs.mean():10.5f} "
                  f"range {vals.min():8.3f}..{vals.max():8.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:24s} FAILED: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{REPO}/recompute/theory/which_teacher.csv", index=False)
    print()
    print(df.to_string(index=False))
    if len(df):
        best = df.loc[df["mean_abs_diff"].idxmin()]
        print(f"\nclosest: {best['candidate']} "
              f"(mean|diff| {best['mean_abs_diff']:.6f})")


if __name__ == "__main__":
    main()
