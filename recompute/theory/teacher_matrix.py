#!/usr/bin/env python3
"""Which teacher checkpoint produced each data/<miner>.scores.json.gz?

Scoring bm25.scores.json.gz's own pairs with checkpoints/crossencoder gives a
mean absolute difference of ~50 on a scale where the stored values run -13 to
-39, so at least one score file was not produced by the current teacher.

The score files were written over five months:

    bm25          2024-11-07     <- predates both surviving teachers
    random        2024-11-29         (checkpoints_old is 2024-11-28)
    crossencoder  2025-03-26         (checkpoints    is 2025-03-25)
    ensemble.all  2025-04-14

This scores a sample of each file's own (query, doc) pairs with every surviving
teacher checkpoint and reports the agreement, so the pairing is measured rather
than inferred from timestamps.

It matters because item 1 arm A compares teacher entropy ACROSS miners. If
different miners' scores came from different teachers, that comparison
confounds "different negatives" with "different teacher", and so do the
MarginMSE and KL distillation targets in Table 4.1.
"""

from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

REPO = "/mnt/primary/implicit/sigir25-implicit-diff"
OUT = f"{REPO}/recompute/theory"
N_QUERIES = 20

SCORE_FILES = {
    "bm25": f"{REPO}/data/bm25.scores.json.gz",
    "random": f"{REPO}/data/random.scores.json.gz",
    "crossencoder": f"{REPO}/data/crossencoder.scores.json.gz",
    "ensemble": f"{REPO}/data/ensemble.all.scores.json.gz",
}


def teachers() -> list[tuple[str, str]]:
    out = [("checkpoints/crossencoder", f"{REPO}/checkpoints/crossencoder")]
    for p in sorted(glob.glob(f"{REPO}/checkpoints/crossencoder/checkpoint-*")):
        if os.path.exists(f"{p}/config.json"):
            out.append((f"current/{os.path.basename(p)}", p))
    # checkpoints_old is gzipped in place; decompress to scratch if present
    old = f"{REPO}/checkpoints_old/crossencoder"
    if os.path.exists(f"{old}/model.safetensors.gz"):
        dest = "/root/teacher_old"
        if not os.path.exists(f"{dest}/model.safetensors"):
            os.makedirs(dest, exist_ok=True)
            import shutil
            for g in glob.glob(f"{old}/*.gz"):
                name = os.path.basename(g)[:-3]
                with gzip.open(g, "rb") as fin, open(f"{dest}/{name}", "wb") as fo:
                    shutil.copyfileobj(fin, fo)
        out.append(("checkpoints_old/crossencoder", dest))
    return out


def main() -> None:
    import ir_datasets as irds
    from rankers import CatTransformer

    # ---- gather sample pairs from every score file --------------------
    samples = {}
    for miner, path in SCORE_FILES.items():
        with gzip.open(path, "rt") as fh:
            stored = json.load(fh)
        qids = sorted(stored.keys())[:N_QUERIES]
        samples[miner] = [(q, d, stored[q][d]) for q in qids for d in stored[q]]
        arr = np.array([s for _, _, s in samples[miner]])
        print(f"{miner:14s} {len(samples[miner]):6d} pairs, "
              f"range {arr.min():8.3f}..{arr.max():8.3f}")

    want_q = {q for s in samples.values() for q, _, _ in s}
    want_d = {d for s in samples.values() for _, d, _ in s}

    ds = irds.load("msmarco-passage/train/triples-small")
    qtext = {}
    for q in ds.queries_iter():
        if q.query_id in want_q:
            qtext[q.query_id] = q.text
            if len(qtext) == len(want_q):
                break
    corpus = irds.load("msmarco-passage")
    dtext = {}
    for doc in corpus.docs_iter():
        if doc.doc_id in want_d:
            dtext[doc.doc_id] = doc.text
            if len(dtext) == len(want_d):
                break
    print(f"\nresolved {len(qtext)}/{len(want_q)} queries, "
          f"{len(dtext)}/{len(want_d)} docs")

    rows = []
    for tname, tpath in teachers():
        print(f"\n--- teacher: {tname} ---")
        try:
            model = CatTransformer.from_pretrained(
                tpath, batch_size=256, verbose=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to load: {exc}")
            continue
        for miner, trips in samples.items():
            trips = [(q, d, s) for q, d, s in trips
                     if q in qtext and d in dtext]
            if not trips:
                continue
            frame = pd.DataFrame(
                [{"qid": q, "docno": d, "query": qtext[q], "text": dtext[d]}
                 for q, d, _ in trips]
            )
            res = model.transform(frame)
            got = dict(zip(zip(res.qid, res.docno), res.score))
            diffs = np.array([abs(got[(q, d)] - s) for q, d, s in trips
                              if (q, d) in got])
            rows.append({
                "teacher": tname,
                "score_file": miner,
                "n": int(diffs.size),
                "mean_abs_diff": float(diffs.mean()),
                "max_abs_diff": float(diffs.max()),
                "frac_within_0.01": float(np.mean(diffs < 0.01)),
            })
            flag = "  <== MATCH" if diffs.mean() < 0.01 else ""
            print(f"  vs {miner:14s} mean|diff|={diffs.mean():10.5f}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/teacher_matrix.csv", index=False)
    print("\n=== agreement matrix (mean |diff|) ===")
    print(df.pivot(index="teacher", columns="score_file",
                   values="mean_abs_diff").to_string(
        float_format=lambda v: f"{v:.5f}"))
    print(f"\nwrote {OUT}/teacher_matrix.csv")


if __name__ == "__main__":
    main()
