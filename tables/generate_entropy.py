#!/usr/bin/env python3
import os
import pandas as pd
import networkx as nx
import numpy as np
import string

# fixed orders
LOSS_ORDER    = ["LCE", "marginMSE"]
SUBSET_ORDER  = ["lower_quartile", "below_median", "inner_quartiles", "above_median", "upper_quartile"]
METRICS       = ["nDCG", "MAP"]
GROUPS        = ["dl19", "dl20", "beir"]
ALLOWED_SPLITS = set(SUBSET_ORDER)

def annotate_equivalence(df_tost, alpha=0.10, metric="nDCG@10"):
    """
    For each (group, loss), builds an equivalence graph over SUBSET_ORDER
    where edges connect splits whose TOST p_lower>alpha & p_upper>alpha.
    Labels each connected component A, B, C… and returns a DataFrame of
    (group, loss, subset, comp_label).
    """
    records = []
    for (group, loss), sub in df_tost.groupby(["group", "loss"]):
        G = nx.Graph()
        G.add_nodes_from(SUBSET_ORDER)
        for _, row in sub.iterrows():
            if (row["measure"] == metric
                and row["p_lower"] > alpha
                and row["p_upper"] > alpha):
                G.add_edge(row["subset1"], row["subset2"])
        comps = sorted(nx.connected_components(G), key=lambda c: sorted(c)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for s in comp:
                records.append({
                    "group":   group,
                    "loss":    loss,
                    "subset":  s,
                    "comp":    label
                })
    return pd.DataFrame.from_records(records)


def generate_table(out_dir, alpha=0.10):
    # 1) load long‐form means and tost tables for CAT
    means = {
        g: pd.read_csv(os.path.join(out_dir, f"means_cat_{g}.tsv"), sep="\t")
        for g in GROUPS
    }
    tosts = {
        g: pd.read_csv(os.path.join(out_dir, f"tost_cat_{g}.tsv"), sep="\t")
        for g in GROUPS
    }

    # 2) annotate equivalences across splits
    eq_frames = []
    for g in GROUPS:
        df_eq = annotate_equivalence(tosts[g], alpha=alpha)
        df_eq["group"] = g
        eq_frames.append(df_eq)
    df_eq_all = pd.concat(eq_frames, ignore_index=True)

    # 3) merge comp labels into means
    merged = []
    for g in GROUPS:
        df = means[g].copy()
        df["group"] = g
        df = df[df["subset"].isin(ALLOWED_SPLITS)]
        df = df.merge(
            df_eq_all[df_eq_all.group == g],
            on=["group", "loss", "subset"],
            how="left"
        )
        merged.append(df)
    df_all = pd.concat(merged, ignore_index=True)

    # 4) keep only the two losses we care about
    df_all = df_all[df_all.loss.isin(LOSS_ORDER)]

    # 5) normalize metric labels
    df_all["metric"] = df_all["measure"].map({
        "AP(rel=2)": "MAP",
        "nDCG@10":   "nDCG"
    })

    # 6) pivot values into MultiIndex (group, metric)
    table = df_all.pivot_table(
        index=["loss", "subset"],
        columns=["group", "metric"],
        values="value",
        aggfunc="mean",
        dropna=False
    )

    # 7) reindex to full grid
    full_idx  = pd.MultiIndex.from_product(
        [LOSS_ORDER, SUBSET_ORDER],
        names=["loss", "subset"]
    )
    full_cols = pd.MultiIndex.from_product(
        [GROUPS, METRICS],
        names=["group", "metric"]
    )
    table = table.reindex(index=full_idx, columns=full_cols)

    # 8) build comp‐lookup for superscripts
    eq_map = df_eq_all.set_index(["group", "loss", "subset"])["comp"].to_dict()
    comp_members = {
        (row.group, row.loss, row.comp): 
          set(df_eq_all.query("group==@row.group and loss==@row.loss and comp==@row.comp")["subset"])
        for row in df_eq_all.itertuples()
    }

    # 9) assemble LaTeX
    latex = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{ll" + "cc"*len(GROUPS) + "}",
        r"  \toprule",
        r"    Loss & Split  " + " & ".join(f"& {g} nDCG & {g} MAP" for g in GROUPS) + r" \\",
        r"  \midrule"
    ]

    # body rows
    for loss, subset in full_idx:
        cells = []
        for g, m in full_cols:
            v = table[g, m].loc[(loss, subset)]
            comp = eq_map.get((g, loss, subset), "")
            members = comp_members.get((g, loss, comp), set())
            # build and sort letter‐codes alphabetically
            codes = sorted(
                string.ascii_uppercase[SUBSET_ORDER.index(s)]
                for s in members
                if s != subset
            )
            sup = "".join(codes)
            if pd.isna(v):
                cell = "-"
            else:
                cell = f"{v:.2f}\\textsuperscript{{{sup}}}"
            cells.append(cell)
        latex.append(
            f"  {loss} & {subset.replace('_', ' ')} "
            + " & ".join(cells)
            + r" \\"
        )
    latex += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]

    # write out
    output = "\n".join(latex)
    path = os.path.join(out_dir, "table_cat.tex")
    with open(path, "w") as f:
        f.write(output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_table(args.out_dir)
