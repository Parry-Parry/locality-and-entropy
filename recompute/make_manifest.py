#!/usr/bin/env python3
"""Step 1: inventory every reported cell and locate its run and qrels files.

One row per (table, row, column) cell of Tables 4.1, 4.3 and 4.4-4.9.

The cell -> run-file mapping is taken from the `name` column of the committed
per-query files, which records the exact run each score came from; names are
then resolved against the run directories on disk.

Usage:
    python recompute/make_manifest.py
"""

from __future__ import annotations

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOMPUTE = os.path.join(REPO, "recompute")

RUN_DIRS = [
    "main_results",
    "runs",
    "beir_runs",
    "rerank_runs",
    "rerank_splade_runs",
    "entropy_beir_runs",
]

QRELS = {
    "dl19": "irds:msmarco-passage/trec-dl-2019/judged",
    "dl20": "irds:msmarco-passage/trec-dl-2020/judged",
}


def build_run_index() -> dict:
    """basename without .res.gz -> repo-relative path."""
    index = {}
    for d in RUN_DIRS:
        full = os.path.join(REPO, d)
        if not os.path.isdir(full):
            continue
        for fname in os.listdir(full):
            if fname.endswith(".res.gz"):
                index.setdefault(fname[: -len(".res.gz")], os.path.join(d, fname))
    return index


def build_name_map() -> dict:
    """(collection, loss, arch, miner) -> run name, from the per-query files."""
    out = {}
    for group in ["dl19", "dl20", "beir"]:
        path = os.path.join(REPO, "main_intermediate", f"perquery_{group}.tsv.gz")
        df = pd.read_csv(
            path, sep="\t", low_memory=False,
            usecols=["name", "domain", "loss", "arch", "dataset_id"],
        ).drop_duplicates()
        for _, r in df.iterrows():
            # per-dataset key (BEIR tables 4.4-4.9)
            out[(r["dataset_id"], r["loss"], r["arch"], r["domain"])] = r["name"]
            # group-level key (Table 4.1); BEIR is pooled, so keep a set
            key = (group, r["loss"], r["arch"], r["domain"])
            out.setdefault(key, set() if group == "beir" else r["name"])
            if group == "beir":
                out[key].add(r["name"])
    return out


ENT_SUBSET = {
    "Lower": "lower_quartile",
    "Inner": "inner_quartiles",
    "Upper": "upper_quartile",
    "Outlier": "outlier_quartiles",
}


def build_entropy_name_map() -> dict:
    """(collection, loss, 'CE', row_label) -> run name, for Table 4.3."""
    inverse = {v: k for k, v in ENT_SUBSET.items()}
    out = {}
    for group in ["dl19", "dl20", "beir"]:
        df = pd.read_csv(
            os.path.join(REPO, "ent_intermediate", f"perquery_cat_{group}.tsv.gz"),
            sep="\t", low_memory=False,
            usecols=["name", "loss", "arch", "subset", "dataset_id"],
        ).drop_duplicates()
        for _, r in df.iterrows():
            label = inverse.get(r["subset"])
            if label is None:          # above_median / below_median are unused
                continue
            key = (group, r["loss"], r["arch"], label)
            out.setdefault(key, set()).add(r["name"])
    return out


def qrels_for(collection: str) -> str:
    if collection in QRELS:
        return QRELS[collection]
    if collection == "beir":
        return "irds:beir/* (pooled over all BEIR collections)"
    return f"irds:beir/{collection}"


def main() -> None:
    runs = build_run_index()
    names = build_name_map()
    ent_names = build_entropy_name_map()

    reported = pd.concat(
        [
            pd.read_csv(os.path.join(RECOMPUTE, "reported", f))
            for f in ["table_4_1.csv", "table_4_3.csv", "table_beir.csv"]
        ],
        ignore_index=True,
    )
    # match the spelling used in the per-query data
    known = {k[0] for k in names}
    reported["collection"] = reported["collection"].map(
        lambda c: c if c in known else f"cqadupstack-{c}"
    )

    rows = []
    for _, r in reported.iterrows():
        key = (r["collection"], r["loss"], r["architecture"], r["miner"])
        # Table 4.3 "Original" is the unfiltered BM25 run from the main tables;
        # its quartile rows come from the entropy runs.
        if str(r["table"]) == "4.3":
            name = (
                names.get((r["collection"], r["loss"], "CE", "BM25"))
                if r["miner"] == "Original"
                else ent_names.get(key)
            )
        else:
            name = names.get(key)

        if pd.isna(r["reported_value"]):
            run_file, status = "", "missing (reported as n/a in the thesis)"
        elif name is None:
            run_file, status = "", "missing (no run name in per-query data)"
        elif isinstance(name, set):
            found = [n for n in sorted(name) if n in runs]
            if len(name) == 1:
                only = next(iter(name))
                run_file = runs.get(only, "")
                status = "found" if run_file else "missing"
            else:
                run_file = f"pooled: {len(found)}/{len(name)} run files"
                status = "found" if len(found) == len(name) else "missing"
        else:
            run_file = runs.get(name, "")
            status = "found" if run_file else "missing"

        rows.append(
            {
                "table": r["table"],
                "architecture": r["architecture"],
                "loss": r["loss"],
                "miner": r["miner"],
                "collection": r["collection"],
                "metric": r["metric"],
                "reported_value": r["reported_value"],
                "run_file": run_file,
                "qrels_file": qrels_for(r["collection"]),
                "status": status,
            }
        )

    man = pd.DataFrame(rows)
    dest = os.path.join(RECOMPUTE, "manifest.csv")
    man.to_csv(dest, index=False)

    print(f"{len(man)} cells")
    print(man["status"].value_counts().to_string())
    miss = man[man["status"].str.startswith("missing")]
    if len(miss):
        print("\nMISSING cells:")
        cols = ["table", "architecture", "loss", "miner", "collection", "metric"]
        print(miss[cols].drop_duplicates().to_string(index=False))
    print(f"\nwrote {os.path.relpath(dest, REPO)}")


if __name__ == "__main__":
    main()
