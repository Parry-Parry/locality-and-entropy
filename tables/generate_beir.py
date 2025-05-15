#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import networkx as nx
import statsmodels.stats.weightstats as ws
import string
from ir_measures import NDCG

# Configuration
LOSS_ORDER   = ["LCE", "RankNet", "marginMSE", "KL"]
ARCHS        = ["BE", "CE"]
ALPHA        = 0.10
METRIC_NAME  = "nDCG@10"

def tost(x, y, low_eq=-0.01, high_eq=0.01):
    """
    Two‐one‐sided t-test for equivalence on METRIC_NAME, fixed bounds.
    Returns (p_overall, p_lower, p_upper).
    """
    p, (_, pl, _), (_, ph, _) = ws.ttost_ind(x, y, low_eq, high_eq)
    return p, pl, ph

def annotate_equivalence(df_tost, alpha=ALPHA, measure=METRIC_NAME):
    """
    For each (loss, arch), build graph over datasets,
    connect pairs with p_lower>alpha and p_upper>alpha,
    label connected components A, B, C…,
    return DataFrame(loss, arch, dataset, comp).
    """
    recs = []
    for (loss, arch), sub in df_tost.groupby(["loss","arch"]):
        G = nx.Graph()
        ds_all = sorted(pd.concat([sub.dataset1, sub.dataset2]).unique())
        G.add_nodes_from(ds_all)
        for _, row in sub.iterrows():
            if (row.measure == measure
                and row.p_lower > alpha
                and row.p_upper > alpha):
                G.add_edge(row.dataset1, row.dataset2)
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for ds in comp:
                recs.append({"loss":loss, "arch":arch, "dataset":ds, "comp":label})
    return pd.DataFrame.from_records(recs)

def main(run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # load per-query BEIR results
    perq = pd.read_csv(
        os.path.join(run_dir, "perquery_beir.tsv.gz"),
        sep="\t", compression="gzip", low_memory=False
    )

    # filter losses and architectures and metric
    perq = perq[
        perq.loss.isin(LOSS_ORDER) &
        perq.arch.isin(ARCHS) &
        (perq.measure == METRIC_NAME)
    ]

    # compute per-dataset mean nDCG
    means = (
        perq
        .groupby(["dataset_id","loss","arch"])["value"]
        .mean()
        .reset_index()
    )

    # pairwise TOST across datasets for each loss,arch
    tost_recs = []
    for (loss, arch), sub in perq.groupby(["loss","arch"]):
        ds_list = sorted(sub.dataset_id.unique())
        for i in range(len(ds_list)):
            for j in range(i+1, len(ds_list)):
                d1, d2 = ds_list[i], ds_list[j]
                x = sub[(sub.dataset_id==d1)]["value"].values
                y = sub[(sub.dataset_id==d2)]["value"].values
                if len(x)<2 or len(y)<2:
                    continue
                p, pl, ph = tost(x, y)
                tost_recs.append({
                    "loss":     loss,
                    "arch":     arch,
                    "dataset1": d1,
                    "dataset2": d2,
                    "measure":  METRIC_NAME,
                    "p_value":  p,
                    "p_lower":  pl,
                    "p_upper":  ph
                })
    df_tost = pd.DataFrame.from_records(tost_recs)

    # annotate equivalence classes
    df_eq = annotate_equivalence(df_tost)

    # build lookup maps
    comp_map = df_eq.set_index(["loss","arch","dataset"])["comp"].to_dict()
    comp_groups = {
        (row.loss,row.arch,row.comp):
            set(df_eq.query("loss==@row.loss and arch==@row.arch and comp==@row.comp")["dataset"])
        for row in df_eq.itertuples()
    }

    # sorted dataset list
    datasets = sorted(means.dataset_id.unique())
    # assign letter codes to datasets
    ds_codes = {ds: string.ascii_uppercase[i] for i, ds in enumerate(datasets)}

    # for each architecture, produce a table
    for arch in ARCHS:
        df_m = means[means.arch==arch]
        pivot = (
            df_m.pivot(index="loss", columns="dataset_id", values="value")
                .reindex(index=LOSS_ORDER, columns=datasets)
        )

        # begin LaTeX
        latex = [
            r"\begin{table}[t]",
            rf"  \caption{{Mean {METRIC_NAME} for architecture {arch} across BEIR datasets}}",
            r"  \centering",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{3pt}",
            r"  \begin{tabular}{l" + "c"*len(datasets) + "}",
            r"  \toprule",
            "Loss & " + " & ".join(datasets) + r" \\",
            r"  \midrule"
        ]

        for loss in LOSS_ORDER:
            cells = []
            for ds in datasets:
                v = pivot.loc[loss, ds]
                sup = ""
                comp = comp_map.get((loss, arch, ds), "")
                members = comp_groups.get((loss, arch, comp), set())
                # superscript codes for other equivalent datasets
                codes = sorted(ds_codes[d] for d in members if d!=ds)
                if codes:
                    sup = "".join(codes)
                if pd.isna(v):
                    cells.append("–")
                else:
                    cells.append(f"{v:.3f}\\textsuperscript{{{sup}}}")
            latex.append(f"{loss} & " + " & ".join(cells) + r" \\")

        latex += [
            r"  \bottomrule",
            r"  \end{tabular}",
            r"\end{table}"
        ]

        # write file
        out_path = os.path.join(out_dir, f"beir_summary_{arch}.tex")
        with open(out_path, "w") as f:
            f.write("\n".join(latex))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    main(args.run_dir, args.out_dir)