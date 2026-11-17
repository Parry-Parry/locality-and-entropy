#!/usr/bin/env python3
"""Measure the corrected pairwise risk R(f) of each trained student, and the
quantities the corrected bound needs, on real evaluation data.

The corrected risk (Definition "Corrected pairwise risk") is

    c_f(x,x') = (1-p) 1{df>0} + p 1{df<0} + 1/2 1{df=0},   p = sigma(g(x)-g(x'))
    R(f)      = E_{mu x mu} [ c_f ]

with g the teacher and f the student. Its floor is
R* = E[min(p, 1-p)] (Lemma "The floor and who attains it"), attained by the
teacher itself, so the excess risk R(f) - R* is what the calibration theorems
bound.

mu here is uniform over the BM25 top-100 candidates for each evaluation query,
which is exactly what the experiments rerank. All pairs within a query are
used (C(100,2) = 4950 per query), so no sampling noise enters.

Measured per student:
    R(f)        corrected risk
    R*          teacher floor, same pairs
    excess      R(f) - R*
    H(g)        teacher entropy on the EVALUATION pairs
    floor_HR    H(g)/(2 log 2), the Hellman-Raviv upper bound on R*
    eta         the Pinsker lower bound on R*
    E[l_f]      the surrogate loss (MarginMSE and RankNet-KL forms)
    bound       floor_HR + zeta sqrt(E[l_f]), the calibration bound

The calibration theorems say excess <= zeta sqrt(E[l_f]). That is a per-student
statement with no sampling or capacity terms, so it can be checked exactly
here: if it ever fails, the theory is wrong.

Usage:
    python measure_risk.py --collection dl19
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

REPO = "/mnt/primary/implicit/sigir25-implicit-diff"
OUT = f"{REPO}/recompute/theory"
TEACHER = f"{REPO}/checkpoints/crossencoder"
LOG2 = math.log(2.0)
DEPTH = 100

IRDS = {
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
}
ZETA = {"RankNet": math.sqrt(2.0), "MarginMSE": 0.5}


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="dl19", choices=list(IRDS))
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import ir_datasets as irds
    import pyterrier as pt
    from rankers import CatTransformer

    coll = args.collection
    ds = irds.load(IRDS[coll])

    # ---- candidate pool: the BM25 run the experiments rerank -------------
    cand_path = f"{REPO}/data/run.msmarco-v1-passage.bm25-flat.{coll}.txt"
    run = pt.io.read_results(cand_path)
    run = (run.sort_values(["qid", "score"], ascending=[True, False])
              .groupby("qid").head(DEPTH))
    print(f"{coll}: {run.qid.nunique()} queries, {len(run)} candidates")

    qtext = {q.query_id: q.text for q in ds.queries_iter()}
    need = set(run["docno"].astype(str))
    corpus = irds.load("msmarco-passage")
    dtext = {}
    for doc in corpus.docs_iter():
        if doc.doc_id in need:
            dtext[doc.doc_id] = doc.text
            if len(dtext) == len(need):
                break
    print(f"resolved {len(dtext)}/{len(need)} documents")

    # ---- teacher scores on the evaluation candidates ---------------------
    print("scoring candidates with the teacher ...")
    teacher = CatTransformer.from_pretrained(
        TEACHER, batch_size=args.batch_size, verbose=True
    )
    frame = pd.DataFrame([
        {"qid": q, "docno": d, "query": qtext[q], "text": dtext.get(d, "")}
        for q, d in zip(run["qid"].astype(str), run["docno"].astype(str))
        if q in qtext and d in dtext
    ])
    tres = teacher.transform(frame)
    g = {}
    for q, d, s in zip(tres.qid, tres.docno, tres.score):
        g.setdefault(str(q), {})[str(d)] = float(s)

    # ---- the student runs ------------------------------------------------
    # explicit map: chained str.replace is unsafe here because "dl_2019"
    # itself contains the substring "dl_20"
    year = {"dl19": "2019", "dl20": "2020"}[coll]
    pattern = f"{REPO}/main_results/msmarco_passage_trec_dl_{year}_judged_*.res.gz"
    students = sorted(glob.glob(pattern))
    students = [s for s in students
                if "graph" not in s and "INT" not in s]
    print(f"{len(students)} student runs")

    rows = []
    for path in students:
        name = os.path.basename(path).replace(".res.gz", "")
        sr = pt.io.read_results(path)
        f = {}
        for q, d, s in zip(sr["qid"].astype(str), sr["docno"].astype(str),
                           sr["score"]):
            f.setdefault(q, {})[d] = float(s)

        risk = floor = ent = n_pairs = 0.0
        mse = kl = 0.0
        for q, gq in g.items():
            fq = f.get(q)
            if not fq:
                continue
            docs = [d for d in gq if d in fq]
            if len(docs) < 2:
                continue
            gv = np.array([gq[d] for d in docs])
            fv = np.array([fq[d] for d in docs])
            i, j = np.triu_indices(len(docs), k=1)
            dg = gv[i] - gv[j]
            df = fv[i] - fv[j]
            p = 1.0 / (1.0 + np.exp(-dg))
            qf = 1.0 / (1.0 + np.exp(-df))

            c = np.where(df > 0, 1 - p, np.where(df < 0, p, 0.5))
            risk += c.sum()
            floor += np.minimum(p, 1 - p).sum()
            ent += binary_entropy(p).sum()
            mse += ((df - dg) ** 2).sum()
            pc, qc = np.clip(p, 1e-12, 1 - 1e-12), np.clip(qf, 1e-12, 1 - 1e-12)
            kl += (pc * np.log(pc / qc) + (1 - pc) * np.log((1 - pc) / (1 - qc))).sum()
            n_pairs += i.size

        if not n_pairs:
            continue
        R, Rs = risk / n_pairs, floor / n_pairs
        H = ent / n_pairs
        loss = {"MarginMSE": mse / n_pairs, "RankNet": kl / n_pairs}
        hr = H / (2 * LOG2)
        row = {
            "run": name, "collection": coll, "n_pairs": int(n_pairs),
            "R_f": R, "R_star": Rs, "excess": R - Rs,
            "H_eval": H, "floor_HellmanRaviv": hr,
            "eta_lower": max(0.0, 0.5 - math.sqrt(max((LOG2 - H) / 2, 0.0))),
            "loss_MarginMSE": loss["MarginMSE"], "loss_RankNet_KL": loss["RankNet"],
        }
        for lname, z in ZETA.items():
            row[f"calib_rhs_{lname}"] = z * math.sqrt(loss[lname])
            row[f"calib_holds_{lname}"] = bool(
                (R - Rs) <= z * math.sqrt(loss[lname]) + 1e-9)
        rows.append(row)
        print(f"  {name[:58]:58s} R={R:.4f} R*={Rs:.4f} excess={R-Rs:.4f}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no student runs matched {pattern}")
    df.to_csv(f"{OUT}/measured_risk_{coll}.csv", index=False)
    print(f"\n=== {coll}: floor check ===")
    print(f"  R* (measured)            : {df.R_star.iloc[0]:.4f}")
    print(f"  eta(H) lower bound       : {df.eta_lower.iloc[0]:.4f}")
    print(f"  H/(2log2) upper bound    : {df.floor_HellmanRaviv.iloc[0]:.4f}")
    ok = (df.eta_lower.iloc[0] <= df.R_star.iloc[0] + 1e-9 <=
          df.floor_HellmanRaviv.iloc[0] + 1e-9)
    print(f"  sandwich holds           : {ok}")
    print(f"\n=== calibration check (per student, exact) ===")
    for lname in ZETA:
        print(f"  {lname:10s} holds for {int(df[f'calib_holds_{lname}'].sum())}"
              f"/{len(df)} students")
    print(f"\nwrote {OUT}/measured_risk_{coll}.csv")


if __name__ == "__main__":
    main()
