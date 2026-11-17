#!/usr/bin/env python3
"""Step 4: recompute every equivalence comparison under PROTOCOL.md.

Paired TOST on per-query differences, theta = eps * mean(baseline) with
eps = 0.01, alpha = 0.05. Comparison set is exactly the original's: all
unordered miner pairs within each (collection group, loss, architecture) cell.

Inputs (read-only, produced by the original analysis):
    main_intermediate/perquery_dl19.tsv.gz
    main_intermediate/perquery_dl20.tsv.gz
    main_intermediate/perquery_beir.tsv.gz
    recompute/reported/*.csv   (reported values + original equivalence claims)

Outputs:
    recompute/reproduction.csv
    recompute/results/tost_table_4_1.csv
    recompute/results/tost_beir_perdataset.csv
    recompute/results/summary.csv
    recompute/per_query/<collection>/<metric>.parquet

Usage:
    python recompute/recompute_tost.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tost import paired_tost  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOMPUTE = os.path.join(REPO, "recompute")

EPS = 0.01          # primary, pre-stated in PROTOCOL.md (thesis Section 4.2.5)
# Secondary sensitivity margins. These are exploratory and were added AFTER the
# primary result was seen; they never replace EPS as the headline (plan rule 2).
EPS_SECONDARY = [0.05, 0.10]
ALPHA = 0.05

# PROTOCOL.md: baseline is the member appearing EARLIER in this order.
DOMAIN_ORDER = ["Random", "BM25", "Cross-Encoder", "Ensemble"]

GROUPS = ["dl19", "dl20", "beir"]

# which BEIR dataset appears in which thesis table
BEIR_TABLE_BY_ARCH = {"BE": ["4.4", "4.5", "4.6"], "CE": ["4.7", "4.8", "4.9"]}


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)          # enforce monotonicity
        adjusted[idx] = min(1.0, running)
    return adjusted


# `cat-MarginMSE-bm25-graph-16-INT` is a different system that parse_run_meta
# mislabels as domain=BM25 because its name contains "bm25", pooling it into the
# marginMSE/BM25 cross-encoder cell. Excluded. See recompute/contaminated_cells.py.
CONTAMINANT = "graph-16-INT"


def load_perquery(group: str) -> pd.DataFrame:
    path = os.path.join(REPO, "main_intermediate", f"perquery_{group}.tsv.gz")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # CE-Teacher is not a miner comparison; the original short-circuits it
    df = df[df["arch"] != "CE-Teacher"]
    return df[~df["name"].str.contains(CONTAMINANT, na=False)].copy()


def save_per_query(group: str, df: pd.DataFrame) -> None:
    """One parquet per (collection, metric): rows = qid, columns = system."""
    for metric, sub in df.groupby("measure"):
        outdir = os.path.join(RECOMPUTE, "per_query", group)
        os.makedirs(outdir, exist_ok=True)
        wide = sub.pivot_table(
            index="qid", columns=["loss", "arch", "domain"], values="value"
        )
        wide.columns = ["|".join(map(str, c)) for c in wide.columns]
        safe = metric.replace("/", "_").replace("(", "").replace(")", "")
        wide.to_parquet(os.path.join(outdir, f"{safe}.parquet"))


SKIPPED: list[dict] = []


def compare_cell(sub: pd.DataFrame, meta: dict) -> list[dict]:
    """All unordered miner pairs within one (loss, arch, measure, collection)."""
    wide = sub.pivot_table(index="qid", columns="domain", values="value")
    present = [d for d in DOMAIN_ORDER if d in wide.columns]
    out = []
    for i, baseline in enumerate(present):
        for candidate in present[i + 1 :]:
            pair = wide[[baseline, candidate]].dropna()
            if len(pair) < 3:
                SKIPPED.append({**meta, "baseline": baseline,
                                "candidate": candidate,
                                "reason": f"only {len(pair)} aligned queries"})
                continue
            if pair[baseline].mean() == 0.0:
                # theta = eps * mean(baseline) = 0 and eps_star is undefined;
                # the baseline scored 0 on every query in this cell.
                SKIPPED.append({**meta, "baseline": baseline,
                                "candidate": candidate,
                                "reason": "baseline mean is 0"})
                continue
            r = paired_tost(
                pair[candidate].to_numpy(),
                pair[baseline].to_numpy(),
                eps=EPS,
                alpha=ALPHA,
            )
            # Secondary margins. Equivalence at margin e holds iff
            # eps_star < e; we recompute rather than rely on that identity and
            # assert the two agree.
            for e in EPS_SECONDARY:
                alt = paired_tost(
                    pair[candidate].to_numpy(),
                    pair[baseline].to_numpy(),
                    eps=e,
                    alpha=ALPHA,
                )
                label = f"equivalent_at_{int(round(e * 100))}pct"
                r[label] = alt["equivalent"]
                assert alt["equivalent"] == (r["eps_star"] < e), (
                    f"eps_star identity violated at eps={e}: "
                    f"eps_star={r['eps_star']!r} equivalent={alt['equivalent']!r}"
                )
            r["baseline"] = baseline
            r["candidate"] = candidate
            out.append(r)
    return out


def run_group(group: str, df: pd.DataFrame, per_dataset: bool) -> pd.DataFrame:
    keys = ["loss", "arch", "measure"]
    if per_dataset:
        keys = ["dataset_id"] + keys
    records = []
    for key, sub in df.groupby(keys):
        meta = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for r in compare_cell(sub, meta):
            r.update(meta)
            r["group"] = group
            r["collection"] = meta.get("dataset_id", group)
            records.append(r)
    return pd.DataFrame(records)


def attach_original_claim(res: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    """original_claim: did the thesis mark this pair equivalent?"""
    claims = {}
    for _, row in reported.iterrows():
        if not isinstance(row["reported_equivalent_to"], str):
            continue
        partners = [p for p in row["reported_equivalent_to"].split(";") if p]
        for partner in partners:
            key = (
                row["collection"],
                row["loss"],
                row["architecture"],
                row["metric"],
                *sorted([row["miner"], partner]),
            )
            claims[key] = True

    def lookup(r):
        key = (
            r["collection"],
            r["loss"],
            r["arch"],
            r["measure"],
            *sorted([r["baseline"], r["candidate"]]),
        )
        return bool(claims.get(key, False))

    res = res.copy()
    res["original_claim"] = res.apply(lookup, axis=1) if len(res) else []
    return res


def reproduction_check(reported: pd.DataFrame, perq: dict) -> pd.DataFrame:
    """Compare recomputed means against every reported cell."""
    rows = []
    for _, r in reported.iterrows():
        group = r["collection"]
        if group in perq:
            df = perq[group]
        elif group in set(perq["beir"]["dataset_id"].unique()):
            df = perq["beir"]
            df = df[df["dataset_id"] == group]
        else:
            continue
        sel = df[
            (df["loss"] == r["loss"])
            & (df["arch"] == r["architecture"])
            & (df["domain"] == r["miner"])
            & (df["measure"] == r["metric"])
        ]
        recomputed = sel["value"].mean() if len(sel) else np.nan
        reported_value = r["reported_value"]
        known = (
            "published cell pools the mislabelled cat-MarginMSE-bm25-graph-16-"
            "INT run; recomputed from the intended run only "
            "(see recompute/contaminated_cells.py)"
            if (
                r["loss"] == "marginMSE"
                and r["architecture"] == "CE"
                and r["miner"] == "BM25"
                and group in ("dl19", "dl20")
            )
            else ""
        )
        rows.append(
            {
                "cell": f"T{r['table']}|{group}|{r['loss']}|{r['architecture']}"
                f"|{r['miner']}|{r['metric']}",
                "reported": reported_value,
                "recomputed": recomputed,
                "abs_diff": abs(recomputed - reported_value)
                if pd.notna(recomputed) and pd.notna(reported_value)
                else np.nan,
                "known_reason": known,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(os.path.join(RECOMPUTE, "results"), exist_ok=True)

    perq = {g: load_perquery(g) for g in GROUPS}
    for g, df in perq.items():
        save_per_query(g, df)
        print(f"{g}: {df['qid'].nunique()} queries, {len(df)} per-query scores")

    # ---- main tables: dl19, dl20, beir(pooled) -------------------------
    main_parts = [run_group(g, perq[g], per_dataset=False) for g in GROUPS]
    main_res = pd.concat(main_parts, ignore_index=True)
    main_res["beir_pooled"] = main_res["group"] == "beir"

    t41 = pd.read_csv(os.path.join(RECOMPUTE, "reported", "table_4_1.csv"))
    main_res = attach_original_claim(main_res, t41)
    main_res["p_tost_holm"] = holm(main_res["p_tost"].to_numpy())
    main_res["table"] = "4.1"

    dest = os.path.join(RECOMPUTE, "results", "tost_table_4_1.csv")
    main_res.to_csv(dest, index=False)
    print(f"\ntable 4.1: {len(main_res)} comparisons -> {os.path.relpath(dest, REPO)}")

    # ---- BEIR per dataset (primary) ------------------------------------
    beir_res = run_group("beir", perq["beir"], per_dataset=True)
    tbeir = pd.read_csv(os.path.join(RECOMPUTE, "reported", "table_beir.csv"))
    # The thesis prints CQADupStack sub-forums bare ("android"); the data
    # spells them "cqadupstack-android". Prefix any reported collection that
    # is not already a dataset_id in the per-query data.
    known = set(perq["beir"]["dataset_id"].unique())
    tbeir["collection"] = tbeir["collection"].map(
        lambda c: c if c in known else f"cqadupstack-{c}"
    )
    unresolved = sorted(set(tbeir["collection"]) - known)
    if unresolved:
        raise ValueError(f"unmapped BEIR collections: {unresolved}")
    beir_res = attach_original_claim(beir_res, tbeir)
    beir_res["p_tost_holm"] = holm(beir_res["p_tost"].to_numpy())
    beir_res["table"] = "4.4-4.9"

    dest = os.path.join(RECOMPUTE, "results", "tost_beir_perdataset.csv")
    beir_res.to_csv(dest, index=False)
    print(f"BEIR per-dataset: {len(beir_res)} comparisons -> "
          f"{os.path.relpath(dest, REPO)}")

    # ---- summary -------------------------------------------------------
    allres = pd.concat([main_res, beir_res], ignore_index=True)
    summary = allres[
        [
            "table",
            "collection",
            "measure",
            "candidate",
            "baseline",
            "n",
            "mean_baseline",
            "mean_diff",
            "theta",
            "ci_low",
            "ci_high",
            "eps_star",
            "p_tost",
            "p_tost_holm",
            "p_diff_two_sided",
            "equivalent",
            "equivalent_at_5pct",
            "equivalent_at_10pct",
            "original_claim",
        ]
    ].rename(columns={"measure": "metric", "equivalent": "equivalent_at_1pct"})
    dest = os.path.join(RECOMPUTE, "results", "summary.csv")
    summary.to_csv(dest, index=False)

    if SKIPPED:
        skipped = pd.DataFrame(SKIPPED)
        skipped.to_csv(
            os.path.join(RECOMPUTE, "results", "skipped_comparisons.csv"),
            index=False,
        )
        print(f"\nskipped {len(skipped)} comparisons "
              f"(see results/skipped_comparisons.csv):")
        for reason, n in skipped["reason"].value_counts().items():
            print(f"  {n:4d}  {reason}")

    # ---- reproduction --------------------------------------------------
    reported_all = pd.concat([t41, tbeir], ignore_index=True)
    repro = reproduction_check(reported_all, perq)
    dest = os.path.join(RECOMPUTE, "reproduction.csv")
    repro.to_csv(dest, index=False)

    # ---- headline ------------------------------------------------------
    n_eq = int(summary["equivalent_at_1pct"].sum())
    n_claimed = int(summary["original_claim"].sum())
    print("\n" + "=" * 68)
    print(f"comparisons                       : {len(summary)}")
    print(f"claimed equivalent by thesis      : {n_claimed}")
    print(f"equivalent at 1% (recomputed)     : {n_eq}")
    print(f"eps_star  median                  : {summary['eps_star'].median():.4f}")
    print(f"eps_star  max                     : {summary['eps_star'].max():.4f}")
    print("-" * 68)
    print("margin sensitivity (secondary, exploratory):")
    for col, lbl in [("equivalent_at_1pct", "1% (primary, pre-stated)"),
                     ("equivalent_at_5pct", "5% (secondary)"),
                     ("equivalent_at_10pct", "10% (secondary)")]:
        n = int(summary[col].sum())
        print(f"  equivalent at {lbl:26s}: {n:5d} / {len(summary)}"
              f"  ({100.0 * n / len(summary):5.1f}%)")
    ok = repro["abs_diff"].dropna()
    print(f"reproduction cells compared       : {len(ok)}")
    unexplained = repro[(repro["abs_diff"] > 0.001) & (repro["known_reason"] == "")]
    print(f"  abs_diff > 0.001                : {int((ok > 0.001).sum())}"
          f"  (unexplained: {len(unexplained)})")
    print(f"  max abs_diff                    : {ok.max():.6f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
