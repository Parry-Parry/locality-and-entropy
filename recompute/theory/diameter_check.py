#!/usr/bin/env python3
"""P2 item 3: which distance produced Table 4.2's diameter column.

Compares the two stored diameter files against the published values under the
aggregation used by notebooks/delta.ipynb (mean and variance of the per-query
values at or below their 95th percentile).

Usage:
    python recompute/theory/diameter_check.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# produced by paper/get_supremum.py  -> pdist(metric="euclidean"), alpha=0.95
EUCLIDEAN = os.path.join(HERE, "data", "supremum.json.gz")
# the file Table 4.2 actually comes from, despite its name
NAMED_COSINE = os.path.join(REPO, "supremum_cosine.json")

# Table 4.2, as printed: "mean +/- X"
PUBLISHED = {
    "random": (10.448, 0.044),
    "bm25": (9.862, 0.205),
    "crossencoder": (9.593, 0.251),
    "ensemble": (9.546, 0.233),
}
ORDER = ["random", "bm25", "crossencoder", "ensemble"]

# scipy's cosine distance is 1 - cos_sim, so it cannot exceed this
MAX_COSINE_DISTANCE = 2.0


def load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def summarise(path: str, label: str) -> pd.DataFrame:
    data = load(path)
    rows = []
    for name in ORDER:
        vals = np.array(list(data[name]["diameters"].values()), dtype=float)
        finite = vals[np.isfinite(vals)]
        p95 = np.percentile(finite, 95)
        trimmed = finite[finite <= p95]
        pub_mean, pub_pm = PUBLISHED[name]
        rows.append({
            "file": label,
            "source": name,
            "n": finite.size,
            "published_mean": pub_mean,
            "mean_le_p95": trimmed.mean(),
            "published_pm": pub_pm,
            "var_le_p95": trimmed.var(),
            "sd_le_p95": trimmed.std(ddof=1),
            "matches_published": abs(trimmed.mean() - pub_mean) < 0.001,
            "exceeds_cosine_bound": bool(finite.max() > MAX_COSINE_DISTANCE),
        })
    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for path, label in [
        (NAMED_COSINE, "supremum_cosine.json"),
        (EUCLIDEAN, "supremum.json (euclidean code path)"),
    ]:
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        frames.append(summarise(path, label))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(HERE, "diameter_check.csv"), index=False)

    pd.set_option("display.width", 200)
    for label, sub in df.groupby("file", sort=False):
        print(f"=== {label} ===")
        print(sub[[
            "source", "published_mean", "mean_le_p95",
            "published_pm", "var_le_p95", "sd_le_p95", "matches_published",
        ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()

    matched = df[df["matches_published"]]
    print(f"file reproducing Table 4.2: "
          f"{matched['file'].unique().tolist()}")

    # ordering across miners, by published value and by each file
    print("\nordering across miners (largest first):")
    pub_order = sorted(ORDER, key=lambda s: -PUBLISHED[s][0])
    print(f"  published            : {' > '.join(pub_order)}")
    for label, sub in df.groupby("file", sort=False):
        o = sub.sort_values("mean_le_p95", ascending=False)["source"].tolist()
        flag = "" if o == pub_order else "   <-- REVERSED"
        print(f"  {label[:20]:20s} : {' > '.join(o)}{flag}")

    # the bound argument
    print(f"\ncosine distance is bounded by {MAX_COSINE_DISTANCE}; "
          f"published values are {min(p[0] for p in PUBLISHED.values()):.3f} to "
          f"{max(p[0] for p in PUBLISHED.values()):.3f}")
    print("-> the published diameters are NOT cosine distances, whatever the "
          "filename and docstrings say")

    # is "+/-" a variance or an SD?
    m = df[df["matches_published"]]
    if len(m):
        dv = (m["var_le_p95"] - m["published_pm"]).abs() / m["published_pm"]
        ds = (m["sd_le_p95"] - m["published_pm"]).abs() / m["published_pm"]
        print(f"\nrelative error treating published +/- as VARIANCE : "
              f"{dv.mean():.4f}")
        print(f"relative error treating published +/- as SD       : "
              f"{ds.mean():.4f}")


if __name__ == "__main__":
    main()
