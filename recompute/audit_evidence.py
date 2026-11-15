#!/usr/bin/env python3
"""Evidence for AUDIT.md: quantify the effect of the inverted TOST criterion.

Reads ONLY the committed TOST outputs (tost_*.tsv) produced by the original
analysis and recounts how many comparisons the original code labelled
"equivalent" versus how many a correctly-oriented TOST would label equivalent
using those same stored p-values.

This isolates the direction-of-test bug. It does NOT correct the margin or the
paired/unpaired bug -- those p-values cannot be repaired after the fact and are
recomputed from per-query scores in Step 4.

Usage:
    python recompute/audit_evidence.py
"""

from __future__ import annotations

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# alpha as used by the original code (generate_*.py default)
ORIGINAL_ALPHA = 0.10

SOURCES = [
    ("main_intermediate", "tost_dl19.tsv"),
    ("main_intermediate", "tost_dl20.tsv"),
    ("main_intermediate", "tost_beir.tsv"),
    ("ent_intermediate", "tost_cat_dl19.tsv"),
    ("ent_intermediate", "tost_cat_dl20.tsv"),
    ("ent_intermediate", "tost_cat_beir.tsv"),
]


def audit_one(path: str, alpha: float = ORIGINAL_ALPHA) -> dict:
    """Count equivalence labels under the original and the correct criterion."""
    df = pd.read_csv(path, sep="\t")
    if "p_lower" not in df or "p_upper" not in df:
        raise ValueError(f"{path}: missing p_lower/p_upper")

    # Original code: edge added when BOTH one-sided p-values are LARGE.
    #   generate_main.py:26-27, generate_beir.py:40-41, generate_entropy.py:96-97
    as_coded = ((df["p_lower"] > alpha) & (df["p_upper"] > alpha)).sum()

    # Correct TOST: equivalence iff BOTH one-sided tests REJECT, i.e.
    #   max(p_lower, p_upper) < alpha
    as_correct = (df[["p_lower", "p_upper"]].max(axis=1) < alpha).sum()

    return {
        "file": os.path.relpath(path, REPO),
        "n_comparisons": len(df),
        "equivalent_as_coded": int(as_coded),
        "equivalent_as_correct": int(as_correct),
        "median_p_lower": float(df["p_lower"].median()),
        "median_p_upper": float(df["p_upper"].median()),
    }


def main() -> None:
    rows = []
    for subdir, fname in SOURCES:
        path = os.path.join(REPO, subdir, fname)
        if not os.path.exists(path):
            print(f"MISSING: {os.path.relpath(path, REPO)}")
            continue
        rows.append(audit_one(path))

    out = pd.DataFrame(rows)
    print(f"alpha = {ORIGINAL_ALPHA} (as used by the original code)\n")
    print(out.to_string(index=False))

    print(
        f"\nTOTAL comparisons: {out['n_comparisons'].sum()}"
        f"\nLabelled equivalent by original code: {out['equivalent_as_coded'].sum()}"
        f"\nEquivalent under correct criterion (same p-values):"
        f" {out['equivalent_as_correct'].sum()}"
    )

    os.makedirs(os.path.join(REPO, "recompute", "results"), exist_ok=True)
    dest = os.path.join(REPO, "recompute", "results", "audit_evidence.csv")
    out.to_csv(dest, index=False)
    print(f"\nwrote {os.path.relpath(dest, REPO)}")


if __name__ == "__main__":
    main()
