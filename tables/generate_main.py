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
GROUPS       = ["dl19", "dl20", "beir"]
# define which metrics to output per group
GROUP_METRICS = {
    "dl19": ["nDCG", "MAP"],
    "dl20": ["nDCG", "MAP"],
    "beir": ["nDCG"]            # drop MAP for BEIR
}

def annotate_equivalence(df_tost, alpha=0.10, metric="nDCG@10"):
    """
    Assign component labels per (group, loss, arch, domain) based on TOST equivalence.
    """
    records = []
    for (group, loss, arch), sub in df_tost.groupby(["group", "loss", "arch"]):
        G = nx.Graph()
        G.add_nodes_from(DOMAIN_ORDER)
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
                    "group":  group,
                    "loss":   loss,
                    "arch":   arch,
                    "domain": d,
                    "comp":   label
                })
    return pd.DataFrame.from_records(records)


def generate_table(out_dir, alpha=0.1):
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

    # 6) pivot values into MultiIndex (group,arch,metric)
    table = df_all.pivot_table(
        index=["loss", "domain"],
        columns=["group", "arch", "metric"],
        values="value",
        aggfunc="mean",
        dropna=False
    )

    # 7) reindex to full grid, dropping BEIR-MAP
    full_idx = pd.MultiIndex.from_product([LOSS_ORDER, DOMAIN_ORDER],
                                          names=["loss", "domain"])
    # build full_cols dynamically from GROUP_METRICS and ARCH_ORDER
    col_tuples = []
    for g in GROUPS:
        for arch in ARCH_ORDER:
            for m in GROUP_METRICS[g]:
                col_tuples.append((g, arch, m))
    full_cols = pd.MultiIndex.from_tuples(col_tuples, names=["group", "arch", "metric"])

    table = table.reindex(index=full_idx, columns=full_cols)

    # 8) build comp membership maps
    eq_map = df_eq_all.set_index(["group", "loss", "arch", "domain"])["comp"].to_dict()
    comp_members = {}
    for (g, loss, arch), sub in df_eq_all.groupby(["group","loss","arch"]):
        for comp_label, grp in sub.groupby("comp"):
            comp_members[(g, loss, arch, comp_label)] = set(grp.domain)

    # 9) assemble LaTeX
    # header group spans
    header = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{ll" + "c" * len(col_tuples) + "}",
        r"  \toprule"
    ]

    # header rows
    # 9a) top-level group labels with their superscripts from BM25 (BE) & Cross-Encoder (CE)
    sp = []
    for g in GROUPS:
        sub = df_eq_all[df_eq_all.group == g]
        be_sup = sub[(sub.arch=="BE") & (sub.domain=="BM25")]["comp"].iloc[0:1].reindex().tolist()[0] if not sub[(sub.arch=="BE") & (sub.domain=="BM25")].empty else ""
        ce_sup = sub[(sub.arch=="CE") & (sub.domain=="Cross-Encoder")]["comp"].iloc[0:1].reindex().tolist()[0] if not sub[(sub.arch=="CE") & (sub.domain=="Cross-Encoder")].empty else ""
        # combine and drop empty
        parts = [x for x in (be_sup, ce_sup) if x]
        sp.append(",".join(parts))

    header.append(
        rf"    &  & "
        rf"\multicolumn{{{len(col_tuples[:len(col_tuples)//3])}}}{{c}}{{TREC DL'19\textsuperscript{{{sp[0]}}}}} "
        rf"& \multicolumn{{{len(col_tuples[len(col_tuples)//3:2*len(col_tuples)//3])}}}{{c}}{{TREC DL'20\textsuperscript{{{sp[1]}}}}} "
        rf"& \multicolumn{{{len(col_tuples[2*len(col_tuples)//3:])}}}{{c}}{{BEIR mean\textsuperscript{{{sp[2]}}}}} \\"
    )

    header += [
        r"  \cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
        r"    &  & "
        r"\multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} "
        r"& \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} "
        r"& \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} \\",
        r"  \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
        r"\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
        r"    Loss & Domain & " 
        + " & ".join(f"{arch} {m}" for _, arch, m in full_cols) 
        + r" \\",
        r"  \midrule"
    ]

    # 9b) body rows with sorted superscripts
    for loss, dom in full_idx:
        cells = []
        for g, arch, m in full_cols:
            v = table[g, arch, m].loc[(loss, dom)]
            comp_label = eq_map.get((g, loss, arch, dom), "")
            members = comp_members.get((g, loss, arch, comp_label), set())
            # build sorted letter codes
            codes = sorted(
                string.ascii_uppercase[DOMAIN_ORDER.index(d)]
                for d in members
                if d != dom
            )
            sup = "".join(codes)
            if pd.isna(v):
                cells.append("–")
            else:
                cells.append(f"{v:.2f}\\textsuperscript{{{sup}}}")
        header.append(f"  {loss} & {dom} & " + " & ".join(cells) + r" \\")

    # footer
    header += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]

    # write out
    out_path = os.path.join(out_dir, "table.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(header))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_table(args.out_dir)
