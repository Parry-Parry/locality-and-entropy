#!/usr/bin/env python3
"""Step 3a: independently recompute per-query metrics from the run files.

The reproduction in recompute_tost.py compares the thesis against the per-query
scores the original pipeline committed. That checks the aggregation but trusts
the original's metric computation. This script goes back to the TREC run files
and re-evaluates them with ir_measures against ir_datasets qrels, then compares
per-query, so an error inside the original evaluation would surface.

Scope: TREC DL 2019 and 2020 (the collections whose runs feed Table 4.1's DL
columns). BEIR is not re-evaluated here -- 24 collections x 4 miners x 4 losses
x 2 architectures is a far longer job, and the DL check already exercises the
same code path.

trec_eval conventions via ir_measures: queries with no judged relevant document
are excluded, and a query missing from a run scores 0.

Outputs:
    recompute/results/verify_from_runs.csv

Usage:
    python recompute/verify_from_runs.py
"""

from __future__ import annotations

import os

import ir_datasets
import ir_measures
import pandas as pd
from ir_measures import AP, NDCG

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOMPUTE = os.path.join(REPO, "recompute")

DATASETS = {
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
}
# evaluate_main.py uses rel=2 for the TREC DL groups
MEASURES = [NDCG @ 10, AP(rel=2)]
NAME = {str(NDCG @ 10): "nDCG@10", str(AP(rel=2)): "AP(rel=2)"}


def main() -> None:
    man = pd.read_csv(os.path.join(RECOMPUTE, "manifest.csv"), dtype={"table": str})
    rows = []

    for group, dsid in DATASETS.items():
        qrels = list(ir_datasets.load(dsid).qrels_iter())
        committed = pd.read_csv(
            os.path.join(REPO, "main_intermediate", f"perquery_{group}.tsv.gz"),
            sep="\t", low_memory=False,
        )

        cells = man[
            (man["collection"] == group) & (man["status"] == "found")
        ][["loss", "architecture", "miner", "run_file"]].drop_duplicates()

        print(f"{group}: {len(cells)} runs")
        for _, c in cells.iterrows():
            path = os.path.join(REPO, c["run_file"])
            if not os.path.exists(path):
                print(f"  MISSING {c['run_file']}")
                continue
            run = list(ir_measures.read_trec_run(path))

            # committed scores for this cell, indexed by (metric, qid)
            cell = committed[
                (committed["loss"] == c["loss"])
                & (committed["arch"] == c["architecture"])
                & (committed["domain"] == c["miner"])
            ]
            lookup = {
                (r["measure"], str(r["qid"])): r["value"]
                for _, r in cell.iterrows()
            }

            for m in ir_measures.iter_calc(MEASURES, qrels, run):
                metric = NAME[str(m.measure)]
                key = (metric, str(m.query_id))
                if key not in lookup:
                    continue
                rows.append({
                    "collection": group,
                    "loss": c["loss"],
                    "arch": c["architecture"],
                    "miner": c["miner"],
                    "metric": metric,
                    "qid": m.query_id,
                    "committed": float(lookup[key]),
                    "from_run_file": float(m.value),
                })

    out = pd.DataFrame(rows)
    out["abs_diff"] = (out["committed"] - out["from_run_file"]).abs()
    dest = os.path.join(RECOMPUTE, "results", "verify_from_runs.csv")
    out.to_csv(dest, index=False)

    over = int((out["abs_diff"] > 1e-6).sum())
    print(f"\n{len(out)} per-query scores re-evaluated from run files")
    print(f"  differing by > 1e-6 : {over}")
    print(f"  max abs difference  : {out['abs_diff'].max():.3e}")
    if over:
        print("\nworst disagreements:")
        cols = ["collection", "loss", "arch", "miner", "metric", "qid",
                "committed", "from_run_file", "abs_diff"]
        print(out.nlargest(10, "abs_diff")[cols].to_string(index=False))
    print(f"\nwrote {os.path.relpath(dest, REPO)}")


if __name__ == "__main__":
    main()
