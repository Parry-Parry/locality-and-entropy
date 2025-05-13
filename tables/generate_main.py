#!/usr/bin/env python3
import os
import pandas as pd
import networkx as nx
import string

# fixed orders
LOSS_ORDER   = ["LCE", "RankNet", "marginMSE", "KL"]
DOMAIN_ORDER = ["Random", "BM25", "Cross-Encoder", "Ensemble"]
ARCH_ORDER   = ["BE", "CE"]
GROUPS       = ["dl19", "dl20", "beir"]
# only nDCG+MAP for dl19/20, but nDCG only for BEIR
GROUP_METRICS = {
    "dl19": ["nDCG", "MAP"],
    "dl20": ["nDCG", "MAP"],
    "beir": ["nDCG"]
}

def annotate_equivalence(df_tost, alpha=0.10, metric="nDCG@10"):
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


def generate_table(out_dir, alpha=0.10):
    # 1) load data
    means = {g: pd.read_csv(os.path.join(out_dir, f"means_{g}.tsv"), sep="\t")
             for g in GROUPS}
    tosts = {g: pd.read_csv(os.path.join(out_dir, f"tost_{g}.tsv"), sep="\t")
             for g in GROUPS}

    # 2) annotate equivalences
    eq_frames = []
    for g in GROUPS:
        df_eq = annotate_equivalence(tosts[g], alpha=alpha)
        df_eq["group"] = g
        eq_frames.append(df_eq)
    df_eq_all = pd.concat(eq_frames, ignore_index=True)

    # 3) merge back into means
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

    # 4) drop any CE-Teacher baseline
    df_all = df_all[df_all.arch != "CE-Teacher"]

    # 5) normalize metric names
    df_all["metric"] = df_all["measure"].map({
        "nDCG@10":   "nDCG",
        "AP(rel=2)": "MAP"
    })

    # 6) pivot
    table = df_all.pivot_table(
        index=["loss", "domain"],
        columns=["group", "arch", "metric"],
        values="value",
        aggfunc="mean",
        dropna=False
    )

    # 7) reindex (drops BEIR‐MAP)
    full_idx   = pd.MultiIndex.from_product([LOSS_ORDER, DOMAIN_ORDER],
                                            names=["loss", "domain"])
    col_tuples = [(g, arch, m)
                  for g in GROUPS
                  for arch in ARCH_ORDER
                  for m in GROUP_METRICS[g]]
    full_cols  = pd.MultiIndex.from_tuples(col_tuples,
                                           names=["group", "arch", "metric"])
    table = table.reindex(index=full_idx, columns=full_cols)

    # 8) build lookup for superscript letters
    eq_map = df_eq_all.set_index(
        ["group", "loss", "arch", "domain"]
    )["comp"].to_dict()
    comp_members = {}
    for (g, loss, arch), sub in df_eq_all.groupby(["group", "loss", "arch"]):
        for comp_label, grp in sub.groupby("comp"):
            comp_members[(g, loss, arch, comp_label)] = set(grp.domain)

    # maximum number of other‐domain letters = len(DOMAIN_ORDER)-1
    max_sup_len = len(DOMAIN_ORDER) - 1

    # 9) assemble LaTeX
    spans = [len(ARCH_ORDER) * len(GROUP_METRICS[g]) for g in GROUPS]
    latex = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{ll" + "c" * sum(spans) + "}",
        r"  \toprule",
        # top headers
        "    Loss & Domain "
        + " ".join(
            f"& \\multicolumn{{{sp}}}{{c}}{{{hdr}}}"
            for sp, hdr in zip(spans, ["TREC DL'19", "TREC DL'20", "BEIR"])
        )
        + r" \\",
        "  \\cmidrule(lr){3-" + str(2 + spans[0]) + "}"
        + "\\cmidrule(lr){" + str(3 + spans[0]) + "-" + str(2 + spans[0] + spans[1]) + "}"
        + "\\cmidrule(lr){" + str(3 + spans[0] + spans[1]) + "-" + str(2 + sum(spans)) + "}",
        # sub‐headers
        "    &  & "  # <-- ensures blank under Loss & Domain
        + " & ".join(f"{arch} {m}" for _, arch, m in full_cols)
        + r" \\",
        r"  \midrule"
    ]

    for loss, dom in full_idx:
        cells = []
        for g, arch, m in full_cols:
            val = table.loc[(loss, dom), (g, arch, m)]
            comp_label = eq_map.get((g, loss, arch, dom), "")
            members = comp_members.get((g, loss, arch, comp_label), set())
            # build sorted codes
            codes = sorted(
                string.ascii_lowercase[DOMAIN_ORDER.index(d)]
                for d in members
                if d != dom
            )
            sup_codes = "".join(codes)
            # pad to length max_sup_len with phantom 'A's
            missing = max_sup_len - len(sup_codes)
            phantom_arg = "a" * missing
            sup_text = f"{sup_codes}\\phantom{{{phantom_arg}}}"

            if pd.isna(val):
                # hyphen plus phantom superscript
                cells.append(f"–\\textsuperscript{{\\phantom{{{'a'*max_sup_len}}}}}")
            else:
                cells.append(f"{val:.2f}\\textsuperscript{{{sup_text}}}")

        latex.append(f"  {loss} & {dom} & " + " & ".join(cells) + r" \\")

    latex += [
        r"  \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ]

    with open(os.path.join(out_dir, "table.tex"), "w") as f:
        f.write("\n".join(latex))
    print(f"Wrote table.tex to {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    generate_table(args.out_dir)
