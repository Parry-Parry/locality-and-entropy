#!/usr/bin/env python3
"""Does data/bm25.crossencoder.scores.json.gz -- the file BM25 training ACTUALLY
used (scripts/train_cross_bm25.sh: TEACHER_FILE=...) -- come from the surviving
teacher?

An earlier check used data/bm25.scores.json.gz, which the training scripts do
not reference; that file has ~200 docs/query and is the mining pool, not the
distillation targets.
"""
import gzip, json
import numpy as np, pandas as pd
import ir_datasets as irds
from rankers import CatTransformer

REPO = "/mnt/primary/implicit/sigir25-implicit-diff"
N = 40

with gzip.open(f"{REPO}/data/bm25.crossencoder.scores.json.gz", "rt") as fh:
    stored = json.load(fh)
qids = sorted(stored)[:N]
trips = [(q, d, stored[q][d]) for q in qids for d in stored[q]]
print(f"{len(trips)} pairs / {len(qids)} queries")

ds = irds.load("msmarco-passage/train/triples-small")
wq = {q for q, _, _ in trips}
qt = {}
for q in ds.queries_iter():
    if q.query_id in wq:
        qt[q.query_id] = q.text
        if len(qt) == len(wq):
            break
wd = {d for _, d, _ in trips}
dt = {}
for doc in irds.load("msmarco-passage").docs_iter():
    if doc.doc_id in wd:
        dt[doc.doc_id] = doc.text
        if len(dt) == len(wd):
            break
trips = [(q, d, s) for q, d, s in trips if q in qt and d in dt]
print(f"resolved {len(trips)}")

m = CatTransformer.from_pretrained(f"{REPO}/checkpoints/crossencoder",
                                   batch_size=256, verbose=False)
res = m.transform(pd.DataFrame(
    [{"qid": q, "docno": d, "query": qt[q], "text": dt[d]} for q, d, _ in trips]))
got = dict(zip(zip(res.qid, res.docno), res.score))
diff = np.array([abs(got[(q, d)] - s) for q, d, s in trips if (q, d) in got])
print(f"\nn={diff.size}  mean|diff|={diff.mean():.6f}  max={diff.max():.6f}  "
      f"frac<0.01={np.mean(diff < 0.01):.4f}")
print("VERDICT: " + ("MATCH -- training targets came from the surviving teacher"
                     if diff.mean() < 0.01 else
                     "MISMATCH -- training targets came from a different model"))
