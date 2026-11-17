#!/usr/bin/env python3
"""P1 Step 6.2: the determinism gate.

Compares regenerated nfcorpus runs against both the committed original run file
and the value printed in the thesis. A difference above 0.001 on any of them is
a stop condition: it means this inference stack no longer reproduces the
original, and nothing regenerated on it can be trusted.

Usage:
    python recompute/runs_p1/verify_determinism.py
"""

from __future__ import annotations

import os

import ir_datasets
import ir_measures
import pandas as pd
from ir_measures import nDCG

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")

THRESHOLD = 0.001

# checkpoint -> published nfcorpus nDCG@10 (Table 4.8, Ensemble rows)
TARGETS = {
    "cat-LCE-ensemble.16-16": 0.327,
    "cat-RankNet-ensemble.16-16": 0.342,
    "cat-KL_Divergence-ensemble.16-16": 0.341,
}


def score(path: str, qrels) -> tuple[float, int]:
    run = list(ir_measures.read_trec_run(path))
    agg = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)
    return float(agg[nDCG @ 10]), len(run)


def main() -> None:
    qrels = list(ir_datasets.load("beir/nfcorpus/test").qrels_iter())

    rows = []
    for name, published in TARGETS.items():
        fname = f"beir_nfcorpus_{name}_bm25-flat.res.gz"
        regen = os.path.join(OUT, fname)
        orig = os.path.join(REPO, "main_results", fname)
        if not os.path.exists(regen):
            print(f"MISSING regenerated run: {fname}")
            continue

        regen_score, regen_n = score(regen, qrels)
        orig_score, orig_n = (
            score(orig, qrels) if os.path.exists(orig) else (float("nan"), 0)
        )
        rows.append({
            "checkpoint": name,
            "published": published,
            "original_run": orig_score,
            "regenerated": regen_score,
            "diff_vs_original": abs(regen_score - orig_score),
            "diff_vs_published": abs(regen_score - published),
            "n_lines_match": regen_n == orig_n,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "determinism_check.csv"), index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    worst_o = df["diff_vs_original"].max()
    worst_p = df["diff_vs_published"].max()
    print(f"\nmax |regenerated - original|  : {worst_o:.6f}")
    print(f"max |regenerated - published| : {worst_p:.6f}")

    # published values are printed to 3dp, so allow half a unit in the last
    # place on top of the threshold when comparing against them
    ok = worst_o <= THRESHOLD and worst_p <= THRESHOLD + 0.0005
    print(f"\nDETERMINISM CHECK: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(
            "stop condition (plan Step 6.2): regenerated runs do not match. "
            "Do not generate the missing runs on this stack."
        )


if __name__ == "__main__":
    main()
