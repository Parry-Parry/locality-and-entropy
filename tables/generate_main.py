#!/usr/bin/env python3
import os
import pandas as pd
import networkx as nx
import numpy as np
import string

# fixed orders
LOSS_ORDER   = ["LCE", "RankNet", "marginMSE", "KL"]
DOMAIN_ORDER = ["Random", "BM25", "Cross-Encoder", "Ensemble"]
ARCH_ORDER   = ["BE", "CE"]
METRICS      = ["nDCG", "MAP"]
GROUPS       = ["dl19", "dl20", "beir"]

def annotate_equivalence(df_tost, alpha=0.025, metric="nDCG@10"):
    """
    Assign component labels per (group, loss, arch, domain) based on TOST equivalence.
    """
    records = []
    for (group, loss, arch), sub in df_tost.groupby(["group", "loss", "arch"]):
        G = nx.Graph()
        doms = DOMAIN_ORDER
        G.add_nodes_from(doms)
        for _, row in sub.iterrows():
            if (row["measure"] == metric
                and row["p_lower"] > alpha
                and row["p_upper"] > alpha):
                G.add_edge(row["domain1"], row["domain2"])
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for d in comp:
                records.append({
                    "group":    group,
                    "loss":     loss,
                    "arch":     arch,
                    "domain":   d,
                    "comp":     label
                })
    return pd.DataFrame.from_records(records)


def generate_table(out_dir, alpha=0.05):
    # 1) load means and tost tables
    means = {
        g: pd.read_csv(os.path.join(out_dir, f"means_{g}.tsv"), sep="\t")
        for g in GROUPS
    }
    tosts = {
        g: pd.read_csv(os.path.join(out_dir, f"tost_{g}.tsv"), sep="\t")
        for g in GROUPS
    }

    # 2) annotate equivalences
    eq_frames = []
    for g in GROUPS:
        df_eq = annotate_equivalence(tosts[g], alpha=alpha)
        df_eq["group"] = g
        eq_frames.append(df_eq)
    df_eq_all = pd.concat(eq_frames, ignore_index=True)

    # 3) merge eq into means
    merged = []
    for g in GROUPS:
        df = means[g].copy()
        df["group"] = g
        df = df.merge(
            df_eq_all[df_eq_all.group == g],
            on=["group", "loss", "arch", "domain"],
            how="left"
        )
        merged.append(df)
    df_all = pd.concat(merged, ignore_index=True)

    # 4) drop CE-Teacher baseline
    df_all = df_all[df_all.arch != "CE-Teacher"]

    # 5) normalize metric labels
    df_all["metric"] = df_all["measure"].map({
        "AP(rel=2)": "MAP",
        "nDCG@10":   "nDCG"
    })

    # 6) pivot only values into (group,arch,metric)
    table = df_all.pivot_table(
        index=["loss", "domain"],
        columns=["group", "arch", "metric"],
        values="value",
        aggfunc="mean",
        dropna=False
    )

    # 7) reindex to full grid with fixed order
    full_idx  = pd.MultiIndex.from_product([LOSS_ORDER, DOMAIN_ORDER],
                                           names=["loss", "domain"])
    full_cols = pd.MultiIndex.from_product([GROUPS, ARCH_ORDER, METRICS],
                                           names=["group", "arch", "metric"])
    table = table.reindex(index=full_idx, columns=full_cols)

    # 8) build comp membership maps
    eq_map = df_eq_all.set_index(["group", "loss", "arch", "domain"])["comp"].to_dict()
    comp_members = {}
    for (g, loss, arch), sub in df_eq_all.groupby(["group","loss","arch"]):
        for comp_label, grp in sub.groupby("comp"):
            comp_members[(g, loss, arch, comp_label)] = set(grp.domain)

    # 9) assemble LaTeX
    latex = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{ll" + "cccc"*len(GROUPS) + "}",
        r"  \toprule"
    ]

    # header superscripts for BM25 (BE) & Cross-Encoder (CE)
    sp = []
    for g in GROUPS:
        sub = df_eq_all[df_eq_all.group == g]
        be_cl = sub[(sub.arch=="BE")    & (sub.domain=="BM25")]["comp"]
        ce_cl = sub[(sub.arch=="CE")    & (sub.domain=="Cross-Encoder")]["comp"]
        be_sup = be_cl.iloc[0] if not be_cl.empty else ""
        ce_sup = ce_cl.iloc[0] if not ce_cl.empty else ""
        sp.append(f"{be_sup},{ce_sup}" if ce_sup else be_sup)

    latex.append(
        rf"    &  & "
        rf"\multicolumn{{4}}{{c}}{{TREC DL'19\textsuperscript{{{sp[0]}}}}} "
        rf"& \multicolumn{{4}}{{c}}{{TREC DL'20\textsuperscript{{{sp[1]}}}}} "
        rf"& \multicolumn{{4}}{{c}}{{BEIR mean\textsuperscript{{{sp[2]}}}}} \\"
    )
    latex += [
        r"  \cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
        r"    &  & "
        r"\multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} "
        r"& \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} "
        r"& \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} \\",
        r"  \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
        r"\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
        r"    Loss & Domain & nDCG & MAP & nDCG & MAP & nDCG & MAP & "
        r"nDCG & MAP & nDCG & MAP & nDCG & MAP \\",
        r"  \midrule"
    ]

    # body rows in fixed order
    for loss in LOSS_ORDER:
        for dom in DOMAIN_ORDER:
            cells = []
            for g in GROUPS:
                for arch in ARCH_ORDER:
                    for m in METRICS:
                        v = table[g, arch, m].loc[(loss, dom)]
                        comp_label = eq_map.get((g, loss, arch, dom), "")
                        # list other domains in same comp
                        members = comp_members.get((g, loss, arch, comp_label), set())
                        other_codes = sorted(
                            d for d in members if d != dom
                        )
                        # map domain names to single‐letter codes in fixed domain order
                        code_map = {d: string.ascii_uppercase[i] for i, d in enumerate(DOMAIN_ORDER)}
                        sup = "".join(code_map[d] for d in other_codes)
                        if pd.isna(v):
                            cell = "–"
                        else:
                            cell = f"{v:.2f}\\textsuperscript{{{sup}}}"
                        cells.append(cell)
            latex.append(f"  {loss} & {dom} & " + " & ".join(cells) + r" \\")

    latex += [r"  \bottomrule", r"\end{tabular}", r"\end{table}"]

    # write to file
    output = "\n".join(latex)
    with open(os.path.join(out_dir, "table.tex"), "w") as f:
        f.write(output)
    print(f"Wrote table.tex to {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_table(args.out_dir)