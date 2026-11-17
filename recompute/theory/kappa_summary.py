#!/usr/bin/env python3
"""P2 item 4: recompute Table 4.2's kappa-hat with median and IQR.

Table 4.2 reports kappa-hat as "mean +/- X" with X in the hundreds while the
means are 8-14. This recomputes the per-query distribution from data/kappa.json
(produced by paper/get_kappa.py) and reports median and interquartile range
alongside mean and standard deviation, so the summary is not dominated by a
heavy right tail.

It also quantifies two defects in get_kappa.py visible by inspection:

  nu = scores / scores.sum()        # scores are raw teacher logits
  mu = np.full_like(nu, 1.0 / len(nu))
  kappa = np.max(mu / nu)

  1. `scores` are raw cross-encoder logits, which are routinely negative, so
     `nu` is not a probability distribution: entries can be negative and the
     sum can be near zero. mu/nu is then negative or explosive.
  2. missing documents default to 0.0 (`.get(str(doc_id), 0.0)`), making the
     corresponding mu/nu infinite.

Usage:
    python recompute/theory/kappa_summary.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA = os.path.join(HERE, "data", "kappa.json.gz")
# Table 4.2's bm25 row does NOT come from kappa.json (which gives 1.158 for
# bm25); it comes from the separate data/bm25_kappa.json. Verified by exact
# match of both mean and variance under the p95-trimming recipe below.
BM25_KAPPA = os.path.join(HERE, "data", "bm25_kappa.json")

# Table 4.2, as printed: "mean +/- X"
PUBLISHED = {
    "random": (14.202, 556.251),
    "bm25": (12.747, 461.588),
    "crossencoder": (11.116, 353.018),
    "ensemble": (8.276, 165.234),
}

ORDER = ["random", "bm25", "crossencoder", "ensemble"]


def main() -> None:
    with gzip.open(KAPPA, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    with open(BM25_KAPPA, encoding="utf-8") as fh:
        data["bm25"] = json.load(fh)["bm25"]

    rows = []
    for name in ORDER:
        vals = np.array(
            list(data[name]["kappa_per_query"].values()), dtype=float
        )
        n_total = vals.size
        finite = vals[np.isfinite(vals)]
        n_inf = n_total - finite.size
        n_neg = int((finite < 0).sum())

        # the published recipe: mean and variance of values at or below p95
        p95 = np.percentile(finite, 95)
        trimmed = finite[finite <= p95]

        pub_mean, pub_pm = PUBLISHED[name]
        rows.append({
            "source": name,
            "n_queries": n_total,
            "n_non_finite": n_inf,
            "n_negative": n_neg,
            "published_mean": pub_mean,
            "published_pm": pub_pm,
            "mean_le_p95": trimmed.mean(),
            "var_le_p95": trimmed.var(),
            "sd_le_p95": trimmed.std(ddof=1),
            "mean": finite.mean(),
            "sd": finite.std(ddof=1),
            "variance": finite.var(ddof=1),
            "median": np.median(finite),
            "q1": np.percentile(finite, 25),
            "q3": np.percentile(finite, 75),
            "iqr": np.percentile(finite, 75) - np.percentile(finite, 25),
            "min": finite.min(),
            "max": finite.max(),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "kappa_summary.csv")
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print("Table 4.2 kappa-hat, recomputed from data/kappa.json\n")
    print(df[[
        "source", "n_queries", "n_non_finite",
        "published_mean", "mean_le_p95", "published_pm", "var_le_p95",
        "sd_le_p95",
    ]].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\nrobust summary (what the thesis should report):\n")
    print(df[[
        "source", "median", "q1", "q3", "iqr", "min", "max",
    ]].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    # is the published "+/-" a variance rather than a standard deviation?
    dv = (df["var_le_p95"] - df["published_pm"]).abs() / df["published_pm"]
    ds = (df["sd_le_p95"] - df["published_pm"]).abs() / df["published_pm"]
    print(f"\nrelative error treating published +/- as VARIANCE : {dv.mean():.4f}")
    print(f"relative error treating published +/- as SD       : {ds.mean():.4f}")
    print(f"\nwrote {os.path.relpath(out, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()
