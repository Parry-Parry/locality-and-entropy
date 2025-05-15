#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import networkx as nx
import statsmodels.stats.weightstats as ws
import string

# Configuration
LOSS_ORDER   = ["LCE", "RankNet", "marginMSE", "KL"]
DOMAIN_ORDER = ["Random", "BM25", "Cross-Encoder", "Ensemble"]
ALPHA        = 0.10
MEASURE_NAME = "nDCG@10"

def tost(x, y, low_eq=-0.01, high_eq=0.01):
    """
    Two‐one‐sided t‐test for equivalence on MEASURE_NAME, fixed bounds.
    Returns (p_overall, p_low, p_high).
    """
    p, (_, pl, _), (_, ph, _) = ws.ttost_ind(x, y, low_eq, high_eq)
    return p, pl, ph

def annotate_equivalence(df_tost, alpha=ALPHA, measure=MEASURE_NAME):
    """
    For each loss, build a graph over DOMAIN_ORDER; connect domains if
    p_lower>alpha and p_upper>alpha, then label connected components A, B, C, D.
    """
    records = []
    for loss, sub in df_tost.groupby("loss"):
        G = nx.Graph()
        G.add_nodes_from(DOMAIN_ORDER)
        for _, row in sub.iterrows():
            if (row["measure"] == measure
                and row["p_lower"] > alpha
                and row["p_upper"] > alpha):
                G.add_edge(row["domain1"], row["domain2"])
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for d in comp:
                records.append({"loss": loss, "domain": d, "comp": label})
    return pd.DataFrame.from_records(records)

def main(run_dir: str, out_dir: str):
    """
    Reads perquery_beir.tsv.gz, aggregates over domains for BE and CE,
    computes equivalence classes via TOST, and writes two LaTeX tables
    (one per architecture) with Loss × Domains.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load per‐query BEIR results
    df = pd.read_csv(
        os.path.join(run_dir, "perquery_beir.tsv.gz"),
        sep="\t", compression="gzip", low_memory=False
    )
    # Filter to our losses, architectures, and measure
    df = df[
        df.loss.isin(LOSS_ORDER) &
        df.arch.isin(["BE", "CE"]) &
        (df.measure == MEASURE_NAME)
    ]

    for arch in ["BE", "CE"]:
        df_arch = df[df.arch == arch]

        # 2) Compute per‐domain means
        means = (
            df_arch
            .groupby(["loss", "domain"])["value"]
            .mean()
            .reset_index()
        )
        means_pivot = (
            means
            .pivot(index="loss", columns="domain", values="value")
            .reindex(index=LOSS_ORDER, columns=DOMAIN_ORDER)
        )

        # 3) Pairwise TOST across domains
        recs = []
        for loss, sub in df_arch.groupby("loss"):
            for i in range(len(DOMAIN_ORDER)):
                for j in range(i+1, len(DOMAIN_ORDER)):
                    d1, d2 = DOMAIN_ORDER[i], DOMAIN_ORDER[j]
                    x = sub[sub.domain == d1]["value"].values
                    y = sub[sub.domain == d2]["value"].values
                    # require at least two samples per domain
                    if len(x) < 2 or len(y) < 2:
                        continue
                    p, pl, ph = tost(x, y)
                    recs.append({
                        "loss":     loss,
                        "domain1":  d1,
                        "domain2":  d2,
                        "measure":  MEASURE_NAME,
                        "p_value":  p,
                        "p_lower":  pl,
                        "p_upper":  ph
                    })
        tost_df = pd.DataFrame.from_records(recs)

        # 4) Annotate equivalence classes
        eq_df = annotate_equivalence(tost_df)

        # 5) Merge comp labels into means
        merged = means.merge(eq_df, on=["loss", "domain"], how="left")
        comp_map = dict(zip(zip(merged.loss, merged.domain), merged.comp))

        # 6) Build LaTeX table
        latex = [
            r"\begin{table}[t]",
            rf"  \caption{{Mean {MEASURE_NAME} for {arch} across domains}}",
            r"  \centering",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{3pt}",
            r"  \begin{tabular}{l" + "c"*len(DOMAIN_ORDER) + "}",
            r"  \toprule",
            "Loss & " + " & ".join(DOMAIN_ORDER) + r" \\",
            r"  \midrule"
        ]

        for loss in LOSS_ORDER:
            row_cells = []
            for d in DOMAIN_ORDER:
                v = means_pivot.loc[loss, d]
                sup = comp_map.get((loss, d), "")
                if pd.isna(v):
                    cell = "–"
                else:
                    cell = f"{v:.3f}\\textsuperscript{{{sup}}}"
                row_cells.append(cell)
            latex.append(f"{loss} & " + " & ".join(row_cells) + r" \\")
        latex += [
            r"  \bottomrule",
            r"  \end{tabular}",
            r"\end{table}"
        ]

        # 7) Write out
        out_path = os.path.join(out_dir, f"summary_{arch}.tex")
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