#!/usr/bin/env python3
import os
import pandas as pd
import networkx as nx
import numpy as np
import string
import statsmodels.stats.weightstats as ws
from ir_measures import NDCG

# configuration
LOSS_ORDER    = ["LCE", "RankNet", "marginMSE", "KL"]
ARCH_ORDER    = ["BE", "CE"]
ALPHA         = 0.10
METRIC_NAME   = "nDCG@10"


def tost(x, y, low_eq=-0.01, high_eq=0.01):
    """
    Two‐one‐sided t-test for equivalence on nDCG, with fixed bounds.
    Returns (p_overall, p_low, p_high).
    """
    p, (_, pl, _), (_, ph, _) = ws.ttost_ind(x, y, low_eq, high_eq)
    return p, pl, ph


def annotate_equivalence(df_tost, alpha=ALPHA, measure=METRIC_NAME):
    """
    For each (loss, arch), build an equivalence graph over datasets,
    label connected components A, B, C… and return DataFrame
    with columns (loss, arch, dataset, comp).
    """
    records = []
    for (loss, arch), sub in df_tost.groupby(["loss", "arch"]):
        G = nx.Graph()
        # all datasets appearing in either column
        datasets = sorted(pd.concat([sub["dataset1"], sub["dataset2"]]).unique())
        G.add_nodes_from(datasets)
        for _, row in sub.iterrows():
            if row["measure"] == measure and row["p_lower"] > alpha and row["p_upper"] > alpha:
                G.add_edge(row["dataset1"], row["dataset2"])
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for ds in comp:
                records.append({
                    "loss":    loss,
                    "arch":    arch,
                    "dataset": ds,
                    "comp":    label
                })
    return pd.DataFrame.from_records(records)


def generate_summary(run_dir: str, out_dir: str):
    """
    1) Read per-query BEIR results,
    2) Aggregate to per-dataset mean nDCG,
    3) Perform TOST across datasets for each loss/arch,
    4) Annotate equivalences,
    5) Emit a LaTeX table with datasets as columns, losses & arch as rows,
       and superscripts denoting equivalent datasets.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) load per-query data
    perq = pd.read_csv(
        os.path.join(run_dir, "perquery_beir.tsv.gz"),
        sep="\t", compression="gzip", low_memory=False
    )

    # filter to our losses and architectures
    perq = perq[perq.loss.isin(LOSS_ORDER) & perq.arch.isin(ARCH_ORDER)]

    # 2) compute per-dataset mean nDCG
    means = (
        perq[perq.measure == METRIC_NAME]
        .groupby(["dataset_id", "loss", "arch"])["value"]
        .mean()
        .reset_index()
    )

    # 3) pairwise TOST across datasets
    tost_records = []
    for (loss, arch), sub in perq.groupby(["loss", "arch"]):
        ds = sorted(sub["dataset_id"].unique())
        for i in range(len(ds)):
            for j in range(i+1, len(ds)):
                d1, d2 = ds[i], ds[j]
                x = sub[(sub.dataset_id == d1) & (sub.measure == METRIC_NAME)]["value"].values
                y = sub[(sub.dataset_id == d2) & (sub.measure == METRIC_NAME)]["value"].values
                if len(x) < 2 or len(y) < 2:
                    continue
                p, pl, ph = tost(x, y)
                tost_records.append({
                    "loss":     loss,
                    "arch":     arch,
                    "dataset1": d1,
                    "dataset2": d2,
                    "measure":  METRIC_NAME,
                    "p_value":  p,
                    "p_lower":  pl,
                    "p_upper":  ph
                })
    df_tost = pd.DataFrame.from_records(tost_records)

    # 4) annotate equivalence classes
    df_eq = annotate_equivalence(df_tost)

    # 5) merge comp labels into means
    merged = means.merge(
        df_eq,
        left_on=["loss", "arch", "dataset_id"],
        right_on=["loss", "arch", "dataset"],
        how="left"
    )

    # prepare pivot: index = loss, arch; columns = datasets
    datasets = sorted(merged["dataset_id"].unique())
    pivot = merged.pivot_table(
        index=["loss", "arch"],
        columns="dataset_id",
        values="value",
        aggfunc="first"
    ).reindex(
        index=pd.MultiIndex.from_product([LOSS_ORDER, ARCH_ORDER], names=["loss", "arch"]),
        columns=datasets
    )

    # prepare superscript mapping
    eq_map = df_eq.set_index(["loss", "arch", "dataset"])["comp"].to_dict()
    comp_groups = {
        (loss, arch, comp): set(df_eq.query("loss==@loss and arch==@arch and comp==@comp")["dataset"])
        for loss, arch, comp in df_eq[["loss", "arch", "comp"]].itertuples(False, None)
    }
    # assign letter codes to datasets
    dataset_codes = {ds: string.ascii_uppercase[i] for i, ds in enumerate(datasets)}

    # 6) assemble LaTeX
    latex = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{ll" + "c"*len(datasets) + "}",
        r"  \toprule",
        "    Loss & Arch & " + " & ".join(datasets) + r" \\",
        r"  \midrule"
    ]

    for loss in LOSS_ORDER:
        for arch in ARCH_ORDER:
            cells = []
            for ds in datasets:
                v = pivot.loc[(loss, arch), ds]
                comp = eq_map.get((loss, arch, ds), "")
                members = comp_groups.get((loss, arch, comp), set())
                # superscript = codes of other datasets in same comp
                codes = sorted(dataset_codes[d] for d in members if d != ds)
                sup = "".join(codes)
                if pd.isna(v):
                    cells.append("–")
                else:
                    cells.append(f"{v:.2f}\\textsuperscript{{{sup}}}")
            latex.append(f"    {loss} & {arch} & " + " & ".join(cells) + r" \\")
    latex += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]

    # write out
    out_path = os.path.join(out_dir, "beir_summary.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Wrote LaTeX summary to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_summary(args.run_dir, args.out_dir)