#!/usr/bin/env python3
import os
import pandas as pd
import networkx as nx
import numpy as np
import string
import statsmodels.stats.weightstats as ws
from ir_measures import NDCG

# fixed orders
LOSS_ORDER  = ["LCE", "marginMSE"]
ARCH_ORDER  = ["BE", "CE"]
METRIC      = "nDCG"
GROUP_NAME  = "beir"
ALPHA       = 0.10

def tost(x, y, low_eq=-0.01, high_eq=0.01):
    """
    Two‐one‐sided t-test for equivalence on nDCG, with fixed bounds.
    Returns (p_overall, p_low, p_high).
    """
    p, (_, pl, _), (_, ph, _) = ws.ttost_ind(x, y, low_eq, high_eq)
    return p, pl, ph

def annotate_equivalence(df_tost, alpha=ALPHA, measure="nDCG@10"):
    """
    Build equivalence graph over datasets for each (loss, arch),
    label connected components A, B, C… and return DataFrame
    with (loss, arch, dataset, comp).
    """
    records = []
    for (loss, arch), sub in df_tost.groupby(["loss","arch"]):
        G = nx.Graph()
        # combine dataset1 and dataset2 into one list
        ds_list = sorted(pd.concat([sub["dataset1"], sub["dataset2"]]).unique())
        G.add_nodes_from(ds_list)
        for _, row in sub.iterrows():
            if (row["measure"] == measure
                and row["p_lower"] > alpha
                and row["p_upper"] > alpha):
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
    1) Read aggregated BEIR per-query results,
    2) Compute per-dataset mean nDCG,
    3) Perform TOST across datasets for each loss/arch,
    4) Annotate equivalences,
    5) Emit LaTeX table with only nDCG and superscripts.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) load per-query BEIR data
    perq = pd.read_csv(
        os.path.join(run_dir, "perquery_beir.tsv.gz"),
        sep="\t", compression="gzip", low_memory=False
    )

    # keep only our two losses and architectures BE, CE
    perq = perq[
        perq.loss.isin(LOSS_ORDER) &
        perq.arch.isin(ARCH_ORDER)
    ]

    # 2) compute per-dataset mean nDCG
    means = (
        perq
        .groupby(["dataset_id","loss","arch"])["value"]
        .mean()
        .reset_index()
    )

    # 3) pairwise TOST across datasets
    records = []
    for (loss, arch), sub in perq.groupby(["loss","arch"]):
        ds = sorted(sub["dataset_id"].unique())
        for i in range(len(ds)):
            for j in range(i+1, len(ds)):
                d1, d2 = ds[i], ds[j]
                x = sub[(sub.dataset_id==d1)&(sub.measure=="nDCG@10")]["value"].values
                y = sub[(sub.dataset_id==d2)&(sub.measure=="nDCG@10")]["value"].values
                if len(x) == 0 or len(y) == 0:
                    continue
                p, pl, ph = tost(x, y)
                records.append({
                    "loss":      loss,
                    "arch":      arch,
                    "dataset1":  d1,
                    "dataset2":  d2,
                    "measure":   "nDCG@10",
                    "p_value":   p,
                    "p_lower":   pl,
                    "p_upper":   ph
                })
    tost_df = pd.DataFrame.from_records(records)

    # 4) build equivalence labels
    eq_df = annotate_equivalence(tost_df)

    # 5) merge comp-labels into means
    merged = means.merge(
        eq_df,
        left_on=["loss","arch","dataset_id"],
        right_on=["loss","arch","dataset"],
        how="left"
    )

    # pivot into table
    pivot = merged.pivot_table(
        index=["loss","dataset_id"],
        columns="arch",
        values="value",
        aggfunc="first"
    )

    # reindex in fixed order
    idx = pd.MultiIndex.from_product(
        [LOSS_ORDER, sorted(merged.dataset_id.unique())],
        names=["loss","dataset"]
    )
    pivot = pivot.reindex(index=idx, columns=ARCH_ORDER)

    # prepare comp-lookup
    comp_map = eq_df.set_index(["loss","arch","dataset"])["comp"].to_dict()
    comp_groups = {
        (row.loss,row.arch,row.comp):
            set(eq_df.query("loss==@row.loss and arch==@row.arch and comp==@row.comp")["dataset"])
        for row in eq_df.itertuples()
    }

    # 6) assemble LaTeX
    latex = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{lllcc}",
        r"  \toprule",
        r"    Loss & Dataset & BE nDCG & CE nDCG \\",
        r"  \midrule"
    ]

    for loss in LOSS_ORDER:
        for ds in sorted(merged.dataset_id.unique()):
            cells = []
            for arch in ARCH_ORDER:
                v = pivot[arch].loc[(loss, ds)]
                comp = comp_map.get((loss, arch, ds), "")
                members = comp_groups.get((loss, arch, comp), set())
                # superscript codes = other datasets in same comp
                codes = [
                    string.ascii_uppercase[i]
                    for i, d in enumerate(sorted(merged.dataset_id.unique()))
                    if d in members and d != ds
                ]
                sup = "".join(codes)
                if pd.isna(v):
                    cells.append("–")
                else:
                    cells.append(f"{v:.2f}\\textsuperscript{{{sup}}}")
            latex.append(f"  {loss} & {ds} & " + " & ".join(cells) + r" \\")

    latex += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]

    # write file
    path = os.path.join(out_dir, "beir_summary.tex")
    with open(path, "w") as f:
        f.write("\n".join(latex))

    print(f"Wrote {path}")


if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_summary(args.run_dir, args.out_dir)