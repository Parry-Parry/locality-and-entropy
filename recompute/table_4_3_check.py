#!/usr/bin/env python3
"""Table 4.3 reproduction, and the BEIR aggregation inconsistency it exposes.

Table 4.3 reproduces exactly only if its three collection columns are read
with DIFFERENT conventions:

  * TREC DL 2019 / 2020  -- micro-average over queries;
  * BEIR, quartile rows  -- MACRO-average (mean of per-dataset means) over the
                            8 BEIR collections the quartile runs cover;
  * BEIR, "Original" row -- MICRO-average over all 24 BEIR collections, i.e.
                            the Table 4.1 convention.

The last two sit in the same column of the same table and are not comparable.
Section 4.3 reads down that column and concludes that "all cases degrade
out-of-domain"; restating the baseline on the same 8 collections with the same
aggregation removes most of that effect.

Outputs:
    recompute/results/table_4_3_reproduction.csv
    recompute/results/table_4_3_beir_like_for_like.csv

Usage:
    python recompute/table_4_3_check.py
"""

from __future__ import annotations

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOMPUTE = os.path.join(REPO, "recompute")

# Table 4.3 row label -> `subset` value in ent_intermediate
SUBSET = {
    "Lower": "lower_quartile",
    "Inner": "inner_quartiles",
    "Upper": "upper_quartile",
    "Outlier": "outlier_quartiles",
}


def load() -> tuple[dict, dict]:
    ent = {
        g: pd.read_csv(
            os.path.join(REPO, "ent_intermediate", f"perquery_cat_{g}.tsv.gz"),
            sep="\t", low_memory=False,
        )
        for g in ["dl19", "dl20", "beir"]
    }
    main = {
        g: pd.read_csv(
            os.path.join(REPO, "main_intermediate", f"perquery_{g}.tsv.gz"),
            sep="\t", low_memory=False,
        )
        for g in ["dl19", "dl20", "beir"]
    }
    return ent, main


def select(row, ent: dict, main: dict) -> pd.DataFrame:
    g = row["collection"]
    if row["miner"] == "Original":
        d = main[g]
        return d[
            (d["loss"] == row["loss"])
            & (d["arch"] == "CE")
            & (d["domain"] == "BM25")
            & (d["measure"] == row["metric"])
        ]
    d = ent[g]
    return d[
        (d["loss"] == row["loss"])
        & (d["subset"] == SUBSET[row["miner"]])
        & (d["measure"] == row["metric"])
    ]


def main_() -> None:
    ent, main = load()
    rep = pd.read_csv(os.path.join(RECOMPUTE, "reported", "table_4_3.csv"))

    rows = []
    for _, r in rep.iterrows():
        sel = select(r, ent, main)
        if not len(sel):
            continue
        micro = sel["value"].mean()
        macro = sel.groupby("dataset_id")["value"].mean().mean()
        # the convention under which this cell reproduces
        if r["collection"] != "beir":
            used, value = "micro", micro
        elif r["miner"] == "Original":
            used, value = "micro (24 collections)", micro
        else:
            used, value = "macro (8 collections)", macro
        rows.append({
            "loss": r["loss"],
            "row": r["miner"],
            "collection": r["collection"],
            "metric": r["metric"],
            "reported": r["reported_value"],
            "recomputed": value,
            "abs_diff": abs(value - r["reported_value"]),
            "aggregation_used": used,
            "n_collections": sel["dataset_id"].nunique(),
        })

    out = pd.DataFrame(rows)
    out.to_csv(
        os.path.join(RECOMPUTE, "results", "table_4_3_reproduction.csv"), index=False
    )
    bad = int((out["abs_diff"] > 0.001).sum())
    print(f"Table 4.3: {len(out)} cells, {bad} above the 0.001 threshold")
    print(out.groupby("aggregation_used").size().to_string())

    # ---- like-for-like BEIR restatement --------------------------------
    beir_ent = ent["beir"]
    covered = sorted(
        beir_ent[beir_ent["subset"].isin(SUBSET.values())]["dataset_id"].unique()
    )
    beir_main = main["beir"]

    lfl = []
    for loss in ["LCE", "marginMSE"]:
        base = beir_main[
            (beir_main["loss"] == loss)
            & (beir_main["arch"] == "CE")
            & (beir_main["domain"] == "BM25")
            & (beir_main["measure"] == "nDCG@10")
        ]
        base8 = (
            base[base["dataset_id"].isin(covered)]
            .groupby("dataset_id")["value"].mean().mean()
        )
        printed = rep[
            (rep["loss"] == loss)
            & (rep["miner"] == "Original")
            & (rep["collection"] == "beir")
        ]["reported_value"].iloc[0]
        lfl.append({
            "loss": loss, "row": "Original", "as_printed": printed,
            "like_for_like": base8, "delta_vs_baseline": 0.0,
        })
        for label, subset in SUBSET.items():
            sel = beir_ent[
                (beir_ent["loss"] == loss)
                & (beir_ent["subset"] == subset)
                & (beir_ent["measure"] == "nDCG@10")
            ]
            v = sel.groupby("dataset_id")["value"].mean().mean()
            pr = rep[
                (rep["loss"] == loss)
                & (rep["miner"] == label)
                & (rep["collection"] == "beir")
            ]["reported_value"].iloc[0]
            lfl.append({
                "loss": loss, "row": label, "as_printed": pr,
                "like_for_like": v, "delta_vs_baseline": v - base8,
            })

    df = pd.DataFrame(lfl)
    df.to_csv(
        os.path.join(RECOMPUTE, "results", "table_4_3_beir_like_for_like.csv"),
        index=False,
    )
    print(f"\nquartile runs cover {len(covered)} BEIR collections: "
          f"{', '.join(covered)}")
    print("\nlike-for-like BEIR (macro-average over those collections):")
    print(df.to_string(index=False))

    improved = int((df[df["row"] != "Original"]["delta_vs_baseline"] > 0).sum())
    total = int((df["row"] != "Original").sum())
    print(f"\n{improved} of {total} quartile rows IMPROVE on the like-for-like "
          f"baseline, against 0 of {total} as printed.")


if __name__ == "__main__":
    main_()
