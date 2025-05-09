#!/usr/bin/env python3
import os
import pandas as pd
import networkx as nx
import numpy as np
import string

def annotate_equivalence(df_tost, alpha=0.05, metric="nDCG@10"):
    """
    Assigns equivalence‐class letters to each domain within each (group, loss, arch)
    based on TOST: domains whose p_lower>alpha and p_upper>alpha get connected,
    components labeled A, B, C…
    """
    records = []
    # now group by arch as well as group & loss
    for (group, loss, arch), sub in df_tost.groupby(["group", "loss", "arch"]):
        G = nx.Graph()
        doms = pd.unique(sub[["domain1", "domain2"]].values.ravel())
        G.add_nodes_from(doms)
        # connect any two domains that are equivalent under this arch/loss
        for _, row in sub.iterrows():
            if (row["measure"] == metric
                and row["p_lower"] > alpha
                and row["p_upper"] > alpha):
                G.add_edge(row["domain1"], row["domain2"])
        # label components A, B, C…
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for d in comp:
                records.append({
                    "group":    group,
                    "loss":     loss,
                    "arch":     arch,
                    "domain":   d,
                    "eq_class": label
                })
    return pd.DataFrame.from_records(records)


def generate_table(out_dir, alpha=0.05):
    groups = ["dl19", "dl20", "beir"]

    # 1) load the long‐form means and tost files
    means = {
        g: pd.read_csv(os.path.join(out_dir, f"means_{g}.tsv"), sep="\t")
        for g in groups
    }
    tosts = {
        g: pd.read_csv(os.path.join(out_dir, f"tost_{g}.tsv"), sep="\t")
        for g in groups
    }

    # 2) annotate equivalence classes per (group,loss,arch)
    eq_frames = []
    for g in groups:
        df_eq = annotate_equivalence(tosts[g], alpha=alpha)
        df_eq["group"] = g
        eq_frames.append(df_eq)
    df_eq_all = pd.concat(eq_frames, ignore_index=True)

    # 3) merge eq_class into the means and concat
    df_merged = []
    for g in groups:
        df = means[g].copy()
        df["group"] = g
        df = df.merge(
            df_eq_all[df_eq_all.group == g],
            on=["group", "loss", "arch", "domain"],
            how="left"
        )
        df_merged.append(df)
    df_all = pd.concat(df_merged, ignore_index=True)

    # 4) drop CE-Teacher baseline entirely
    df_all = df_all[df_all.arch != "CE-Teacher"]

    # 5) normalize metric labels
    df_all["metric"] = df_all["measure"].map({
        "AP(rel=2)": "MAP",
        "nDCG@10":   "nDCG"
    })

    # 6) pivot only the 'value' into (group,arch,metric)
    table = df_all.pivot_table(
        index=["loss", "domain"],
        columns=["group", "arch", "metric"],
        values="value",
        aggfunc="mean",
        dropna=False
    )

    # 7) reindex to full grid of combinations
    full_idx = pd.MultiIndex.from_product(
        [df_all.loss.unique(), df_all.domain.unique()],
        names=["loss", "domain"]
    )
    full_cols = pd.MultiIndex.from_product(
        [groups, ["BE", "CE"], ["nDCG", "MAP"]],
        names=["group", "arch", "metric"]
    )
    table = table.reindex(index=full_idx, columns=full_cols)

    # 8) build eq_class lookup per (group,loss,arch,domain)
    eq_map = df_eq_all.set_index(
        ["group", "loss", "arch", "domain"]
    )["eq_class"].to_dict()

    # 9) assemble LaTeX
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"  \centering")
    latex.append(r"  \footnotesize")
    latex.append(r"  \setlength{\tabcolsep}{3pt}")
    latex.append(r"  \begin{tabular}{ll" + "cccc" * len(groups) + "}")
    latex.append(r"  \toprule")

    # header superscripts: BM25 for BE, Cross-Encoder for CE
    sp = []
    for g in groups:
        subs = df_eq_all[df_eq_all.group == g]
        be_cl = ""
        ce_cl = ""
        filt = subs[(subs.arch=="BE") & (subs.domain=="BM25")]["eq_class"]
        if not filt.empty: be_cl = filt.iloc[0]
        filt = subs[(subs.arch=="CE") & (subs.domain=="Cross-Encoder")]["eq_class"]
        if not filt.empty: ce_cl = filt.iloc[0]
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

    # body rows: append superscript per (group,arch,domain)
    for loss in table.index.levels[0]:
        for dom in table.loc[loss].index:
            row_vals = []
            for g in groups:
                for arch in ["BE", "CE"]:
                    for m in ["nDCG", "MAP"]:
                        v = table[g, arch, m].loc[(loss, dom)]
                        sup = eq_map.get((g, loss, arch, dom), "")
                        if pd.isna(v):
                            cell = "–"
                        else:
                            cell = f"{v:.2f}\\textsuperscript{{{sup}}}"
                        row_vals.append(cell)
            latex.append(f"  {loss} & {dom} & " + " & ".join(row_vals) + r" \\")
    latex.append(r"  \bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    # write to file
    output = "\n".join(latex)
    out_path = os.path.join(out_dir, "table.tex")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"Wrote LaTeX table with per-arch equivalence superscripts to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_table(args.out_dir)