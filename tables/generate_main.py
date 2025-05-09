#!/usr/bin/env python3
import os
import pandas as pd
import networkx as nx
import numpy as np
import string

def annotate_equivalence(df_tost, alpha=0.05, metric="nDCG@10"):
    """
    Returns a DataFrame with columns (group, loss, arch, domain, comp)
    where 'comp' is the connected‐component label (A, B, C…) under TOST.
    """
    records = []
    for (group, loss, arch), sub in df_tost.groupby(["group", "loss", "arch"]):
        # build equivalence graph over domains
        G = nx.Graph()
        doms = sorted(pd.unique(sub[["domain1","domain2"]].values.ravel()))
        G.add_nodes_from(doms)
        for _, row in sub.iterrows():
            if (row["measure"] == metric
                and row["p_lower"] > alpha
                and row["p_upper"] > alpha):
                G.add_edge(row["domain1"], row["domain2"])
        # label components A, B, C…
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            comp_label = string.ascii_uppercase[idx]
            for d in comp:
                records.append({
                    "group": group,
                    "loss":  loss,
                    "arch":  arch,
                    "domain":d,
                    "comp":  comp_label
                })
    return pd.DataFrame.from_records(records)


def generate_table(out_dir, alpha=0.05):
    groups = ["dl19", "dl20", "beir"]

    # load mean‐values and tost outputs
    means = {g: pd.read_csv(os.path.join(out_dir, f"means_{g}.tsv"), sep="\t")
             for g in groups}
    tosts = {g: pd.read_csv(os.path.join(out_dir, f"tost_{g}.tsv"), sep="\t")
             for g in groups}

    # build component labels per (group,loss,arch)
    eq_frames = []
    for g in groups:
        df_eq = annotate_equivalence(tosts[g], alpha=alpha)
        df_eq["group"] = g
        eq_frames.append(df_eq)
    df_eq_all = pd.concat(eq_frames, ignore_index=True)

    # merge eq info into means and concat
    merged = []
    for g in groups:
        df = means[g].copy()
        df["group"] = g
        df = df.merge(
            df_eq_all[df_eq_all.group == g],
            on=["group","loss","arch","domain"],
            how="left"
        )
        merged.append(df)
    df_all = pd.concat(merged, ignore_index=True)

    # drop the baseline CE-Teacher
    df_all = df_all[df_all.arch != "CE-Teacher"]

    # simplify metric names
    df_all["metric"] = df_all["measure"].map({
        "AP(rel=2)": "MAP",
        "nDCG@10":   "nDCG"
    })

    # pivot values only
    table = df_all.pivot_table(
        index=["loss","domain"],
        columns=["group","arch","metric"],
        values="value",
        aggfunc="mean",
        dropna=False
    )

    # reindex to full grid
    full_idx = pd.MultiIndex.from_product(
        [df_all.loss.unique(), df_all.domain.unique()],
        names=["loss","domain"]
    )
    full_cols = pd.MultiIndex.from_product(
        [groups, ["BE","CE"], ["nDCG","MAP"]],
        names=["group","arch","metric"]
    )
    table = table.reindex(index=full_idx, columns=full_cols)

    # prepare domain‐code mapping and comp membership
    #  a) domain_code[(g,loss,arch,domain)] = 'A','B',...
    #  b) comp_members[(g,loss,arch,comp_label)] = set(domains)
    domain_code = {}
    comp_members = {}
    for (group, loss, arch), sub in df_eq_all.groupby(["group","loss","arch"]):
        doms = sorted(sub.domain.unique())
        for i, d in enumerate(doms):
            domain_code[(group,loss,arch,d)] = string.ascii_uppercase[i]
        for comp_label, rows in sub.groupby("comp"):
            comp_members[(group,loss,arch,comp_label)] = set(rows.domain)

    # build latex
    latex = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{ll" + "cccc"*len(groups) + "}",
        r"  \toprule"
    ]

    # header superscripts: BM25→BE, Cross-Encoder→CE
    sp = []
    for g in groups:
        sub = df_eq_all[df_eq_all.group==g]
        be_sup = ""
        ce_sup = ""
        # find comp for BM25 under BE
        mask = (sub.arch=="BE") & (sub.domain=="BM25")
        if mask.any():
            be_sup = sub[mask].comp.iloc[0]
        mask = (sub.arch=="CE") & (sub.domain=="Cross-Encoder")
        if mask.any():
            ce_sup = sub[mask].comp.iloc[0]
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
        r"    Loss & Domain & nDCG & MAP & nDCG & MAP & nDCG & MAP & nDCG & MAP & nDCG & MAP & nDCG & MAP \\",
        r"  \midrule"
    ]

    # body
    for loss in table.index.levels[0]:
        for dom in table.loc[loss].index:
            cells = []
            for g in groups:
                for arch in ["BE","CE"]:
                    for m in ["nDCG","MAP"]:
                        v = table[g,arch,m].loc[(loss,dom)]
                        # find this domain's comp label
                        comp_lbl = df_eq_all[
                            (df_eq_all.group==g)&
                            (df_eq_all.loss==loss)&
                            (df_eq_all.arch==arch)&
                            (df_eq_all.domain==dom)
                        ]["comp"]
                        if comp_lbl.empty:
                            sup = ""
                        else:
                            comp_label = comp_lbl.iloc[0]
                            members = comp_members[(g,loss,arch,comp_label)]
                            # map other domains to their codes
                            other_codes = sorted(
                                domain_code[(g,loss,arch,d2)]
                                for d2 in members if d2!=dom
                            )
                            sup = "".join(other_codes)
                        cell = "–" if pd.isna(v) else f"{v:.2f}\\textsuperscript{{{sup}}}"
                        cells.append(cell)
            latex.append(f"  {loss} & {dom} & " + " & ".join(cells) + r" \\")
    latex += [r"  \bottomrule", r"\end{tabular}", r"\end{table}"]

    # write out
    output = "\n".join(latex)
    with open(os.path.join(out_dir,"table.tex"),"w") as f:
        f.write(output)
    print(f"Wrote table.tex to {out_dir}")


if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_table(args.out_dir)