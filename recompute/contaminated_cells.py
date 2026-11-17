#!/usr/bin/env python3
"""The marginMSE/BM25 cross-encoder cells pool two different runs.

`parse_run_meta` in tables/evaluate_main.py assigns the sampling domain by
substring match:

    for tag, name in [("random","Random"), ("ensemble","Ensemble"),
                      ("crossencoder","Cross-Encoder"), ("bm25","BM25")]:
        if tag in base.lower():

`msmarco_passage_trec_dl_20XX_judged_cat-MarginMSE-bm25-graph-16-INT` contains
"bm25", so it is labelled domain=BM25 and averaged into the BM25 cell alongside
the intended `cat-MarginMSE-bm25.16-16`. It is a different system.

Affected: TREC DL 2019 and 2020, marginMSE, cross-encoder, BM25 -- four printed
values in Table 4.1. BEIR is unaffected (765 cells, none pooling two runs).

Outputs:
    recompute/results/contaminated_cells.csv

Usage:
    python recompute/contaminated_cells.py
"""

from __future__ import annotations

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOMPUTE = os.path.join(REPO, "recompute")

CONTAMINANT = "graph-16-INT"


def main() -> None:
    reported = pd.read_csv(os.path.join(RECOMPUTE, "reported", "table_4_1.csv"))
    rows = []

    for group in ["dl19", "dl20"]:
        d = pd.read_csv(
            os.path.join(REPO, "main_intermediate", f"perquery_{group}.tsv.gz"),
            sep="\t", low_memory=False,
        )
        cell = d[
            (d["loss"] == "marginMSE")
            & (d["arch"] == "CE")
            & (d["domain"] == "BM25")
        ]
        clean = cell[~cell["name"].str.contains(CONTAMINANT)]
        dirty = cell[cell["name"].str.contains(CONTAMINANT)]

        for metric in ["nDCG@10", "AP(rel=2)"]:
            pub = reported[
                (reported["collection"] == group)
                & (reported["loss"] == "marginMSE")
                & (reported["architecture"] == "CE")
                & (reported["miner"] == "BM25")
                & (reported["metric"] == metric)
            ]["reported_value"].iloc[0]
            rows.append({
                "collection": group,
                "metric": metric,
                "as_published": pub,
                "intended_run_only": clean[clean["measure"] == metric]["value"].mean(),
                "contaminating_run": dirty[dirty["measure"] == metric]["value"].mean(),
                "correction": clean[clean["measure"] == metric]["value"].mean() - pub,
                "intended_run": clean["name"].unique()[0],
                "contaminant": dirty["name"].unique()[0],
            })

    out = pd.DataFrame(rows)
    out.to_csv(
        os.path.join(RECOMPUTE, "results", "contaminated_cells.csv"), index=False
    )

    cols = ["collection", "metric", "as_published", "intended_run_only",
            "contaminating_run", "correction"]
    print("Table 4.1, marginMSE / BM25 / cross-encoder:\n")
    print(out[cols].to_string(index=False))
    print(f"\nintended run      : {out['intended_run'].iloc[0]}")
    print(f"contaminating run : {out['contaminant'].iloc[0]}")
    print(
        "\nThe published dip in this row (it is the only marginMSE cell below "
        "its neighbours)\nis the average of the two runs, not a property of "
        "BM25-mined negatives."
    )


if __name__ == "__main__":
    main()
