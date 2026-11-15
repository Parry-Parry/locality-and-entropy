#!/usr/bin/env python3
"""Step 4.3: eps_star figure per table.

One dot per comparison with its 90% CI, expressed as a percentage of the
baseline mean, with vertical lines at +/-1% (the pre-stated margin).

Usage:
    python recompute/make_figures.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOMPUTE = os.path.join(REPO, "recompute")
FIGDIR = os.path.join(RECOMPUTE, "figures")

EPS_PCT = 1.0  # the pre-stated 1% margin


def plot_table(df: pd.DataFrame, table: str) -> None:
    """Dot-and-CI plot of mean(d) as a percentage of the baseline mean."""
    df = df.dropna(subset=["mean_baseline", "ci_low", "ci_high"])
    df = df[df["mean_baseline"] > 0].copy()
    if df.empty:
        print(f"table {table}: nothing to plot")
        return

    # express everything as a percentage of the baseline mean
    df["centre_pct"] = 100.0 * df["mean_diff"] / df["mean_baseline"]
    df["lo_pct"] = 100.0 * df["ci_low"] / df["mean_baseline"]
    df["hi_pct"] = 100.0 * df["ci_high"] / df["mean_baseline"]
    df = df.sort_values("centre_pct").reset_index(drop=True)

    height = max(3.0, min(0.12 * len(df), 22.0))
    fig, ax = plt.subplots(figsize=(8.5, height))

    y = range(len(df))
    colours = ["#1b7837" if e else "#999999" for e in df["equivalent_at_1pct"]]

    ax.hlines(y, df["lo_pct"], df["hi_pct"], color=colours, linewidth=0.9, alpha=0.8)
    ax.scatter(df["centre_pct"], y, s=7, c=colours, zorder=3)

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    for x in (-EPS_PCT, EPS_PCT):
        ax.axvline(x, color="#c51b7d", linewidth=1.2, linestyle="--")

    n_eq = int(df["equivalent_at_1pct"].sum())
    ax.set_xlabel("mean(candidate - baseline), % of baseline mean  (90% CI)")
    ax.set_ylabel(f"comparison (n = {len(df)}, sorted)")
    ax.set_title(
        f"Table {table}: equivalence at the pre-stated 1% margin\n"
        f"{n_eq} of {len(df)} comparisons equivalent "
        f"(green); dashed lines at +/-1%"
    )
    ax.set_yticks([])

    # clip the x range so a few huge intervals do not flatten the rest
    lo_q = df["lo_pct"].quantile(0.02)
    hi_q = df["hi_pct"].quantile(0.98)
    span = max(abs(lo_q), abs(hi_q), 2 * EPS_PCT) * 1.1
    ax.set_xlim(-span, span)

    fig.tight_layout()
    safe = table.replace(".", "_").replace("-", "_")
    for ext in ("pdf", "png"):
        dest = os.path.join(FIGDIR, f"eps_star_table_{safe}.{ext}")
        fig.savefig(dest, dpi=200)
    plt.close(fig)
    print(f"table {table}: {len(df)} comparisons plotted "
          f"({n_eq} equivalent) -> figures/eps_star_table_{safe}.(pdf|png)")


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(RECOMPUTE, "results", "summary.csv"))
    for table, sub in summary.groupby("table"):
        plot_table(sub, str(table))


if __name__ == "__main__":
    main()
