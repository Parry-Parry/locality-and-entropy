#!/usr/bin/env python3
"""Generate LaTeX table of effectiveness for CAT experiments.

The table reports nDCG and MAP for the two TREC DL test collections
(DL'19, DL'20) and nDCG only for the BEIR benchmark.

Statistical equivalence across sampling subsets is annotated using the
TOST procedure (α = 0.10). Subsets that are not statistically different
share the same superscript letter.

Input files
-----------
For each evaluation group *g* ∈ {dl19, dl20, beir} located in *out_dir*:

* means_cat_<g>.tsv – Mean effectiveness per (loss, subset, measure) triple.
* tost_cat_<g>.tsv  – TOST p‑values per (loss, subset1, subset2, measure) triple.

Both files are assumed to be produced by `cat_eval.py` in the same project.

Output
------
`table_cat.tex` – Formatted LaTeX table stored in *out_dir*.

Example
-------
$ ./generate_table.py --out_dir runs/2025‑05‑14
"""

from __future__ import annotations

import argparse
import os
import string
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd

# ----------------------------------------------------------------------
# Immutable configuration
# ----------------------------------------------------------------------

LOSS_ORDER: List[str] = ["LCE", "marginMSE"]

SUBSET_ORDER: List[str] = [
    "lower_quartile",
    "inner_quartiles",
    "upper_quartile",
    "outlier_quartiles",
]

GROUPS: List[str] = ["dl19", "dl20", "beir"]

#: Metrics to *display* per evaluation group
GROUP_METRICS: Dict[str, Tuple[str, ...]] = {
    "dl19": ("nDCG", "MAP"),
    "dl20": ("nDCG", "MAP"),
    "beir": ("nDCG",),
}

ALLOWED_SPLITS: Set[str] = set(SUBSET_ORDER)

# ----------------------------------------------------------------------

def annotate_equivalence(
    df_tost: pd.DataFrame, *, alpha: float = 0.10, metric: str = "nDCG@10"
) -> pd.DataFrame:
    """Label statistically equivalent sampling subsets using TOST.

    Parameters
    ----------
    df_tost : pd.DataFrame
        Long‑form TOST results containing the columns
        [`group`, `loss`, `subset1`, `subset2`, `measure`, `p_lower`, `p_upper`].
    alpha : float, optional
        Significance level for the equivalence test, by default ``0.10``.
    metric : str, optional
        Measure to consider when building the equivalence graph,
        by default ``"nDCG@10"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [`group`, `loss`, `subset`, `comp`].
    """
    records: List[dict] = []

    for (group, loss), sub in df_tost.groupby(["group", "loss"]):
        graph = nx.Graph()
        graph.add_nodes_from(SUBSET_ORDER)

        mask = (
            (sub["measure"] == metric)
            & (sub["p_lower"] > alpha)
            & (sub["p_upper"] > alpha)
        )
        for _, row in sub[mask].iterrows():
            graph.add_edge(row["subset1"], row["subset2"])

        components = sorted(nx.connected_components(graph), key=lambda c: sorted(c)[0])
        for idx, component in enumerate(components):
            label = string.ascii_uppercase[idx]
            for s in component:
                records.append({
                    "group": group,
                    "loss": loss,
                    "subset": s,
                    "comp": label,
                })

    return pd.DataFrame.from_records(records)


# ----------------------------------------------------------------------

def generate_table(out_dir: str, *, alpha: float = 0.10) -> None:
    """Generate LaTeX table and save it to *out_dir*."""

    # ------------------------------------------------------------------
    # 1) Load means and TOST results
    # ------------------------------------------------------------------
    means: Dict[str, pd.DataFrame] = {
        g: pd.read_csv(os.path.join(out_dir, f"means_cat_{g}.tsv"), sep="\t")
        for g in GROUPS
    }
    tosts: Dict[str, pd.DataFrame] = {
        g: pd.read_csv(os.path.join(out_dir, f"tost_cat_{g}.tsv"), sep="\t")
        for g in GROUPS
    }

    # ------------------------------------------------------------------
    # 2) Annotate equivalence classes
    # ------------------------------------------------------------------
    eq_frames = [annotate_equivalence(tosts[g], alpha=alpha) for g in GROUPS]
    df_eq_all = pd.concat(eq_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 3) Merge equivalence labels into the means
    # ------------------------------------------------------------------
    merged = []
    for g in GROUPS:
        df = means[g].copy()
        df["group"] = g
        df = df[df["subset"].isin(ALLOWED_SPLITS)]
        df = df.merge(
            df_eq_all[df_eq_all.group == g],
            on=["group", "loss", "subset"],
            how="left",
        )
        merged.append(df)

    df_all = pd.concat(merged, ignore_index=True)

    # Keep only losses of interest
    df_all = df_all[df_all.loss.isin(LOSS_ORDER)]

    # Normalise metric labels
    df_all["metric"] = df_all["measure"].map({
        "AP(rel=2)": "MAP",
        "nDCG@10": "nDCG",
    })

    # ------------------------------------------------------------------
    # 4) Pivot: index = (loss, subset) ; columns = (group, metric)
    # ------------------------------------------------------------------
    table = df_all.pivot_table(
        index=["loss", "subset"],
        columns=["group", "metric"],
        values="value",
        aggfunc="mean",
        dropna=False,
    )

    # ------------------------------------------------------------------
    # 5) Re‑index to dense grid and drop unwanted metric columns
    # ------------------------------------------------------------------
    full_idx = pd.MultiIndex.from_product(
        [LOSS_ORDER, SUBSET_ORDER], names=["loss", "subset"]
    )

    full_cols = pd.MultiIndex.from_tuples(
        [(g, m) for g in GROUPS for m in GROUP_METRICS[g]],
        names=["group", "metric"],
    )

    table = table.reindex(index=full_idx, columns=full_cols)

    # ------------------------------------------------------------------
    # 6) Maps for superscript annotation
    # ------------------------------------------------------------------
    eq_map = df_eq_all.set_index(["group", "loss", "subset"])["comp"].to_dict()

    comp_members: Dict[Tuple[str, str, str], Set[str]] = {
        (row.group, row.loss, row.comp): set(
            df_eq_all.query(
                "group == @row.group and loss == @row.loss and comp == @row.comp"
            )["subset"]
        )
        for row in df_eq_all.itertuples()
    }

    # ------------------------------------------------------------------
    # 7) Assemble LaTeX lines
    # ------------------------------------------------------------------
    n_value_cols = len(full_cols)
    col_spec = "ll" + "c" * n_value_cols

    latex: List[str] = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"  \toprule",
    ]

    # Header row
    header_cells: List[str] = []
    for g in GROUPS:
        header_cells.append("& nDCG")
        if "MAP" in GROUP_METRICS[g]:
            header_cells.append("& MAP")
    latex.append("    Loss & Split  " + " ".join(header_cells) + r" \\")
    latex.append(r"  \midrule")

    # Body rows
    for loss, subset in full_idx:
        cells: List[str] = []
        for g, m in full_cols:
            value = table.at[(loss, subset), (g, m)]
            comp = eq_map.get((g, loss, subset), "")
            members = comp_members.get((g, loss, comp), set())

            superscript = "".join(
                sorted(
                    string.ascii_uppercase[SUBSET_ORDER.index(s)]
                    for s in members
                    if s != subset
                )
            )

            if pd.isna(value):
                cell = "-"
            else:
                cell = f"{value:.2f}"
            cells.append(cell)

        latex.append(
            f"  {loss} & {subset.replace('_', ' ')} " + " & ".join(cells) + r" \\")

    latex += [
        r"  \bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    output_path = os.path.join(out_dir, "table_cat.tex")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(latex))

    print(f"Wrote {output_path}")


# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="Directory containing CAT result TSV files.")
    parser.add_argument("--alpha", type=float, default=0.10, help="Equivalence test significance level.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_table(args.out_dir, alpha=args.alpha)
