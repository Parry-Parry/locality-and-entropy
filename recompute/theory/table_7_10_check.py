#!/usr/bin/env python3
"""P2 item 6: Table 7.10 (Chapter 7) -- repeated row blocks and the 2dp deltas.

Parses Table 7.10 out of the thesis text and tests whether the Rel-P rows are
byte-identical to the corresponding Abs-P rows.

Usage:
    python recompute/theory/table_7_10_check.py
"""

from __future__ import annotations

import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(os.path.dirname(HERE), "thesis_extract.txt")

RETRIEVERS = [
    "BM25", "ColBERT", "monoT5", "Contriever", "TAS-B",
    "SPLADE", "ANCE", "E5",
]

# Abs has 3 positions x 3 generators, Rel has 2 x 3
EXPECTED_PER_RETRIEVER = 15

ROW_RE = re.compile(
    r"^\s*(Abs|Rel)\s+(Before|Middle|After)\s+"
    r"(GPT-3\.5|Llama-2|Static)\s+"
    r"([\d.]+)\s+"
    r"([\d.]+)\(([+-][\d.]+)\)(\S?)\s+"
    r"([\d.]+)\(([+-][\d.]+)\)(\S?)\s+"
    r"([\d.]+)\(([+-][\d.]+)\)(\S?)\s*$"
)


def parse() -> pd.DataFrame:
    with open(EXTRACT, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    start = next(i for i, l in enumerate(lines) if l.startswith("Table 7.10:"))
    # the table runs until the next numbered table caption
    end = next(
        (
            i
            for i, l in enumerate(lines[start + 1 :], start=start + 1)
            if re.match(r"^Table \d+\.\d+:", l)
        ),
        len(lines),
    )

    rows, retriever = [], None
    for line in lines[start:end]:
        stripped = line.strip()
        if stripped in RETRIEVERS:
            retriever = stripped
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        (scheme, position, generator, alpha,
         ndcg, ndcg_d, ndcg_s,
         rr, rr_d, rr_s,
         mrpr, mrpr_d, mrpr_s) = m.groups()
        rows.append({
            "retriever": retriever,
            "scheme": scheme,
            "position": position,
            "generator": generator,
            "alpha": alpha,
            "ndcg": ndcg, "ndcg_delta": ndcg_d, "ndcg_sig": ndcg_s,
            "rr": rr, "rr_delta": rr_d, "rr_sig": rr_s,
            "mrpr": mrpr, "mrpr_delta": mrpr_d, "mrpr_sig": mrpr_s,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = parse()

    # guard: a retriever block longer than expected means a header was missed
    # and rows were attributed to the previous retriever
    counts = df.groupby("retriever").size()
    bad = counts[counts > EXPECTED_PER_RETRIEVER]
    if len(bad):
        raise SystemExit(
            f"retriever block(s) longer than {EXPECTED_PER_RETRIEVER} rows: "
            f"{bad.to_dict()} -- a retriever header is missing from "
            f"RETRIEVERS, so rows are mislabelled"
        )

    df.to_csv(os.path.join(HERE, "table_7_10_parsed.csv"), index=False)
    print(f"parsed {len(df)} rows across "
          f"{df['retriever'].nunique()} retrievers: "
          f"{', '.join(df['retriever'].dropna().unique())}\n")

    value_cols = [c for c in df.columns
                  if c not in ("scheme", "retriever", "position", "generator")]

    # compare each Rel row against the Abs row with the same
    # (retriever, position, generator)
    comparisons = []
    for _, rel in df[df["scheme"] == "Rel"].iterrows():
        abs_match = df[
            (df["scheme"] == "Abs")
            & (df["retriever"] == rel["retriever"])
            & (df["position"] == rel["position"])
            & (df["generator"] == rel["generator"])
        ]
        if not len(abs_match):
            continue
        a = abs_match.iloc[0]
        identical = all(a[c] == rel[c] for c in value_cols)
        comparisons.append({
            "retriever": rel["retriever"],
            "position": rel["position"],
            "generator": rel["generator"],
            "identical_to_Abs": identical,
            "n_fields": len(value_cols),
        })

    cmp = pd.DataFrame(comparisons)
    cmp.to_csv(os.path.join(HERE, "table_7_10_duplication.csv"), index=False)

    n_ident = int(cmp["identical_to_Abs"].sum())
    print(f"Rel rows with a matching Abs row : {len(cmp)}")
    print(f"  byte-identical in all {len(value_cols)} fields : {n_ident}")
    print(f"  differing in any field                : {len(cmp) - n_ident}\n")
    print(cmp.to_string(index=False))

    # Rel-P has no Middle, per the Section 7.4 definition
    print("\npositions present per scheme:")
    print(df.groupby("scheme")["position"].unique().to_string())

    # the two-decimal deltas
    print("\ndeltas printed to 2dp rather than 3dp:")
    two_dp = []
    for _, r in df.iterrows():
        for metric in ["ndcg", "rr", "mrpr"]:
            d = r[f"{metric}_delta"]
            if len(d.split(".")[-1]) == 2:
                two_dp.append({
                    "retriever": r["retriever"], "scheme": r["scheme"],
                    "position": r["position"], "generator": r["generator"],
                    "metric": metric, "value": r[metric], "delta": d,
                    "implied_baseline": f"{float(r[metric]) - float(d):.3f}",
                })
    tdf = pd.DataFrame(two_dp)
    tdf.to_csv(os.path.join(HERE, "table_7_10_two_dp.csv"), index=False)
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
