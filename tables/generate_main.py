#!/usr/bin/env python3
import os
import pandas as pd
import networkx as nx
import numpy as np
import string

def annotate_equivalence(df_tost, alpha=0.05, metric="nDCG@10"):
    """
    Assigns equivalence‐class letters to each domain within each (group, loss)
    where TOST p_lower>alpha and p_upper>alpha.
    """
    records = []
    for (group, loss), sub in df_tost.groupby(["group", "loss"]):
        G = nx.Graph()
        doms = pd.unique(sub[["domain1", "domain2"]].values.ravel())
        G.add_nodes_from(doms)
        for _, row in sub.iterrows():
            if (row["measure"] == metric
                and row["p_lower"] > alpha
                and row["p_upper"] > alpha):
                G.add_edge(row["domain1"], row["domain2"])
        # find connected components and label them A, B, C…
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for d in comp:
                records.append({
                    "group":    group,
                    "loss":     loss,
                    "domain":   d,
                    "eq_class": label
                })
    return pd.DataFrame.from_records(records)


def generate_table(out_dir, alpha=0.05):
    groups = ["dl19", "dl20", "beir"]

    # 1) load the long‐form "means" and "tost" files
    means = {
        g: pd.read_csv(os.path.join(out_dir, f"means_{g}.tsv"), sep="\t")
        for g in groups
    }
    tosts = {
        g: pd.read_csv(os.path.join(out_dir, f"tost_{g}.tsv"), sep="\t")
        for g in groups
    }

    # 2) build the equivalence‐class table
    eq_frames = []
    for g in groups:
        df_eq = annotate_equivalence(tosts[g], alpha=alpha)
        df_eq["group"] = g
        eq_frames.append(df_eq)
    df_eq_all = pd.concat(eq_frames, ignore_index=True)

    # 3) merge eq_class into each means DataFrame, then concat
    df_merged = []
    for g in groups:
        df = means[g].copy()
        df["group"] = g
        df = df.merge(
            df_eq_all[df_eq_all.group == g],
            on=["group", "loss", "domain"],
            how="left"
        )
        df_merged.append(df)
    df_all = pd.concat(df_merged, ignore_index=True)

    # 4) drop the CE-Teacher baseline entirely
    df_all = df_all[df_all.arch != "CE-Teacher"]

    # 5) normalize metric labels
    df_all["metric"] = df_all["measure"].map({
        "AP(rel=2)": "MAP",
        "nDCG@10":   "nDCG"
    })

    # 6) pivot only the 'value' into a MultiIndex (group, arch, metric)
    table = df_all.pivot_table(
        index=["loss", "domain"],
        columns=["group", "arch", "metric"],
        values="value",
        aggfunc="mean",
        dropna=False
    )

    # 7) ensure every (loss,domain) × (group,arch,metric) appears
    full_idx = pd.MultiIndex.from_product(
        [df_all.loss.unique(), df_all.domain.unique()],
        names=["loss", "domain"]
    )
    full_cols = pd.MultiIndex.from_product(
        [groups, ["BE", "CE"], ["nDCG", "MAP"]],
        names=["group", "arch", "metric"]
    )
    table = table.reindex(index=full_idx, columns=full_cols)

    # 8) build a lookup for eq_class superscripts
    eq_map = df_eq_all.set_index(["group", "loss", "domain"])["eq_class"].to_dict()

    # 9) assemble the LaTeX table
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"  \centering")
    latex.append(r"  \footnotesize")
    latex.append(r"  \setlength{\tabcolsep}{3pt}")
    latex.append(r"  \begin{tabular}{ll" + "cccc" * len(groups) + "}")
    latex.append(r"  \toprule")

    # header row with superscripts
    sp = []
    for g in groups:
        subs = df_eq_all[df_eq_all.group == g]
        be = subs[subs.domain == "BM25"].eq_class
        ce = subs[subs.domain == "Cross-Encoder"].eq_class
        be_cl = be.iloc[0] if not be.empty else ""
        ce_cl = ce.iloc[0] if not ce.empty else ""
        sp.append(f"{be_cl},{ce_cl}" if ce_cl else be_cl)

    latex.append(
        rf"    &  & "
        rf"\multicolumn{{4}}{{c}}{{TREC DL'19\textsuperscript{{{sp[0]}}}}} "
        rf"& \multicolumn{{4}}{{c}}{{TREC DL'20\textsuperscript{{{sp[1]}}}}} "
        rf"& \multicolumn{{4}}{{c}}{{BEIR mean\textsuperscript{{{sp[2]}}}}} \\"
    )
    latex.append(r"  \cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}")
    latex.append(
        r"    &  & "
        r"\multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} "
        r"& \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} "
        r"& \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} \\"
    )
    latex.append(
        r"  \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
        r"\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}"
    )
    latex.append(
        r"    Loss & Domain & nDCG & MAP & nDCG & MAP & nDCG & MAP & "
        r"nDCG & MAP & nDCG & MAP & nDCG & MAP \\"
    )
    latex.append(r"  \midrule")

    # body rows: append superscript per cell
    for loss in table.index.levels[0]:
        for dom in table.loc[loss].index:
            row_vals = []
            for g in groups:
                for arch in ["BE", "CE"]:
                    for m in ["nDCG", "MAP"]:
                        v = table[g, arch, m].loc[(loss, dom)]
                        sup = eq_map.get((g, loss, dom), "")
                        if pd.isna(v):
                            cell = "–"
                        else:
                            cell = f"{v:.2f}\\textsuperscript{{{sup}}}"
                        row_vals.append(cell)
            latex.append(f"  {loss} & {dom} & " + " & ".join(row_vals) + r" \\")
    latex.append(r"  \bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    print("\n".join(latex))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_table(args.out_dir)