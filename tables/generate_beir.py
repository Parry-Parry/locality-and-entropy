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
DOMAINS      = ["Random", "BM25", "Cross-Encoder", "Ensemble"]
ARCHS        = ["BE", "CE"]
ALPHA        = 0.10
MEASURE_NAME = "nDCG@10"

# map each domain to a single‐letter code for superscripts
DOMAIN_CODES = {d: string.ascii_uppercase[i] for i, d in enumerate(DOMAINS)}

def tost(x, y, low_eq=-0.01, high_eq=0.01):
    """
    Two‐one‐sided t-test for equivalence on MEASURE_NAME, fixed bounds.
    Returns (p_overall, p_lower, p_upper).
    """
    p, (_, pl, _), (_, ph, _) = ws.ttost_ind(x, y, low_eq, high_eq)
    return p, pl, ph

def annotate_equivalence(df_tost):
    """
    For each (dataset, loss, arch), build a graph over DOMAINS, connect
    domains whose TOST p_lower>ALPHA and p_upper>ALPHA, label connected
    components A–D, return DataFrame(dataset, loss, arch, domain, comp).
    """
    recs = []
    for (ds, loss, arch), sub in df_tost.groupby(["dataset", "loss", "arch"]):
        G = nx.Graph()
        G.add_nodes_from(DOMAINS)
        for _, row in sub.iterrows():
            if (row.measure == MEASURE_NAME
                and row.p_lower > ALPHA
                and row.p_upper > ALPHA):
                G.add_edge(row.domain1, row.domain2)
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for d in comp:
                recs.append({
                    "dataset": ds,
                    "loss":    loss,
                    "arch":    arch,
                    "domain":  d,
                    "comp":    label
                })
    return pd.DataFrame.from_records(recs)

def main(run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1) load per‐query BEIR results
    perq = pd.read_csv(
        os.path.join(run_dir, "perquery_beir.tsv.gz"),
        sep="\t", compression="gzip", low_memory=False
    )

    # 2) filter to our losses, archs, and measure
    perq = perq[
        perq.loss.isin(LOSS_ORDER) &
        perq.arch.isin(ARCHS) &
        (perq.measure == MEASURE_NAME)
    ]

    # 3) compute per‐dataset, per‐domain means
    means = (
        perq
        .groupby(["dataset_id", "loss", "arch", "domain"])["value"]
        .mean()
        .reset_index()
    )

    # 4) perform pairwise TOST across domains within each (dataset, loss, arch)
    tost_recs = []
    for (ds, loss, arch), sub in perq.groupby(["dataset_id","loss","arch"]):
        for i in range(len(DOMAINS)):
            for j in range(i+1, len(DOMAINS)):
                d1, d2 = DOMAINS[i], DOMAINS[j]
                x = sub[sub.domain == d1]["value"].values
                y = sub[sub.domain == d2]["value"].values
                if len(x) < 2 or len(y) < 2:
                    continue
                p, pl, ph = tost(x, y)
                tost_recs.append({
                    "dataset": ds,
                    "loss":    loss,
                    "arch":    arch,
                    "domain1": d1,
                    "domain2": d2,
                    "measure": MEASURE_NAME,
                    "p_value": p,
                    "p_lower": pl,
                    "p_upper": ph
                })
    df_tost = pd.DataFrame.from_records(tost_recs)

    # 5) annotate equivalence classes
    df_eq = annotate_equivalence(df_tost)

    # 6) build lookup maps
    comp_map = df_eq.set_index(["dataset","loss","arch","domain"])["comp"].to_dict()
    comp_groups = {
        (row.dataset, row.loss, row.arch, row.comp):
            set(df_eq.query(
                "dataset==@row.dataset and loss==@row.loss and arch==@row.arch and comp==@row.comp"
            )["domain"])
        for row in df_eq.itertuples()
    }

    # 7) list of datasets (columns)
    datasets = sorted(means.dataset_id.unique())

    # 8) for each architecture, build & write table
    for arch in ARCHS:
        m = means[means.arch == arch]
        # pivot: rows=(loss,domain), columns=datasets
        pivot = (
            m
            .pivot(index=["loss","domain"], columns="dataset_id", values="value")
            .reindex(
                index=pd.MultiIndex.from_product([LOSS_ORDER, DOMAINS], names=["loss","domain"]),
                columns=datasets
            )
        )

        # begin LaTeX
        header = " & ".join(datasets)
        latex = [
            r"\begin{table}[t]",
            rf"  \caption{{Mean {MEASURE_NAME} for architecture {arch} across BEIR datasets}}",
            r"  \centering",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{3pt}",
            r"  \begin{tabular}{ll" + "c"*len(datasets) + "}",
            r"  \toprule",
            rf"    Loss & Domain & {header} \\",
            r"  \midrule"
        ]

        for loss in LOSS_ORDER:
            for domain in DOMAINS:
                cells = []
                for ds in datasets:
                    v = pivot.loc[(loss, domain), ds]
                    comp = comp_map.get((ds, loss, arch, domain), "")
                    members = comp_groups.get((ds, loss, arch, comp), set())
                    # superscript = codes of other domains in same comp
                    codes = sorted(DOMAIN_CODES[d] for d in members if d != domain)
                    sup = "".join(codes)
                    if pd.isna(v):
                        cells.append("–")
                    else:
                        cells.append(f"{v:.3f}\\textsuperscript{{{sup}}}")
                latex.append(f"{loss} & {domain} & " + " & ".join(cells) + r" \\")

        latex += [
            r"  \bottomrule",
            r"  \end{tabular}",
            r"\end{table}"
        ]

        out_path = os.path.join(out_dir, f"beir_summary_{arch}.tex")
        with open(out_path, "w") as f:
            f.write("\n".join(latex))
        print(f"Wrote {out_path}")

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    main(args.run_dir, args.out_dir)