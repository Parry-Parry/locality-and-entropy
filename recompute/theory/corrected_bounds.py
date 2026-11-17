#!/usr/bin/env python3
"""Numerical estimates of the CORRECTED Chapter 4 bounds.

Source: "Chapter 4, corrected theory: full proofs, one operation per step"
(17 November 2026). Everything here is an estimate of a quantity that appears
in that document; nothing from the submitted theory is reused.

What the corrected bound contains (Theorem "Excess risk, unweighted training",
which is the one matching the experiments -- training here is unweighted):

    R(f_hat)  <=  H(g)/(2 log 2)                        teacher floor
                + zeta * sqrt( eps_F                    approximation error
                             + 4 Rad_n(L)               capacity
                             + 2B sqrt(log(2/delta)/2n) estimation
                             + 2B (1 - 1/kappa^2) )     distribution-shift bias

Note what is NOT in it: the empirical diameter Delta-hat of Table 4.2 plays no
role in the corrected statement. It was an artefact of the submitted proof.
The miner-dependent terms are H(g) and kappa; eps_F, Rad_n and B are
properties of the student class and the sample size, shared across miners.

Also reported is the two-sided sandwich on the floor itself,

    eta(H) <= R* <= H/(2 log 2),     eta(H) = max(0, 1/2 - sqrt((log2 - H)/2))

from Corollary "A valid lower bound on the floor" and Theorem "Teacher floor".
The submitted thesis used eta as an UPPER bound; it is a lower bound.

ON THE EVALUATION MEASURE mu. The corrected theory is per query, with mu the
measure over candidates seen at evaluation and pi the mining measure. The
experiments rerank a BM25 candidate list to depth 100, so the operative mu is
uniform over BM25's top 100. In rank-bin terms that is

    mu_eval = (10/100, 90/100, 0, 0)   over bins 1-10, 11-100, 101-1000, >1000

which is reported alongside the two hypothetical mu of item 2. Using the real
evaluation depth is the main way these estimates improve on the submitted ones.

Usage:
    python recompute/theory/corrected_bounds.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG2 = math.log(2.0)

BINS = ["1-10", "11-100", "101-1000", ">1000"]

# candidate evaluation measures over the four rank bins
MU = {
    # what the experiments actually do: rerank BM25 top-100
    "eval_top100": np.array([10 / 100, 90 / 100, 0.0, 0.0]),
    # the two hypotheticals from the plan, kept for comparison
    "uniform": np.array([0.25, 0.25, 0.25, 0.25]),
    "top_weighted": np.array([0.4, 0.3, 0.2, 0.1]),
}

ZETA = {"RankNet": math.sqrt(2.0), "MarginMSE": 0.5}


def eta(h: float) -> float:
    """Lower bound on the Bayes floor from teacher entropy (Pinsker)."""
    inner = (LOG2 - h) / 2.0
    return max(0.0, 0.5 - math.sqrt(max(inner, 0.0)))


def floor_upper(h: float) -> float:
    """Hellman-Raviv upper bound on the Bayes floor."""
    return h / (2.0 * LOG2)


def kappa_tv(pi: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """kappa = sup mu/pi over bins, and TV(mu, pi).

    Bins where mu > 0 but pi == 0 give kappa = inf: the miner never samples a
    region the evaluation measure puts mass on, so no finite density ratio
    exists and the coverage assumption fails outright.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pi > 0, mu / np.where(pi > 0, pi, 1.0), np.inf)
    ratio = np.where((mu == 0) & (pi == 0), 0.0, ratio)
    return float(np.max(ratio)), float(0.5 * np.abs(mu - pi).sum())


def main() -> None:
    ent = pd.read_csv(os.path.join(HERE, "pairwise_entropy.csv")).set_index("miner")
    rb = pd.read_csv(os.path.join(HERE, "rank_bins_bm25.csv")).set_index("miner")
    miners = ["random", "bm25", "crossencoder", "ensemble"]

    # ---------- the floor sandwich, per miner ----------------------------
    rows = []
    for m in miners:
        h = float(ent.loc[m, "mean_entropy_nats"])
        rows.append({
            "miner": m,
            "H_nats": h,
            "H_pct_of_log2": 100 * h / LOG2,
            "floor_lower_eta": eta(h),
            "floor_upper_HellmanRaviv": floor_upper(h),
            "sandwich_width": floor_upper(h) - eta(h),
        })
    floor = pd.DataFrame(rows)
    floor.to_csv(os.path.join(HERE, "bounds_floor.csv"), index=False)

    print("=" * 74)
    print("TEACHER FLOOR, two-sided (corrected theory)")
    print("=" * 74)
    print("  eta(H) <= R* <= H/(2 log 2).  The submitted thesis used eta as an")
    print("  UPPER bound; it is a LOWER bound (Lemma 'What Pinsker actually")
    print("  gives'). H is measured on each miner's own training negatives.\n")
    print(floor.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------- kappa and the shift bias, per miner and per mu -----------
    rows = []
    for m in miners:
        pi = np.array([float(rb.loc[m, f"frac_{b}"]) for b in BINS])
        for mu_name, mu in MU.items():
            k, tv = kappa_tv(pi, mu)
            bias_coeff = (
                float("inf") if not np.isfinite(k) else 2.0 * (1.0 - 1.0 / k**2)
            )
            rows.append({
                "miner": m,
                "mu": mu_name,
                "kappa": k,
                "TV_mu_pi": tv,
                "TV_pairs_bound": (
                    float("inf") if not np.isfinite(k) else 1.0 - 1.0 / k**2
                ),
                "bias_term_over_B": bias_coeff,
            })
    shift = pd.DataFrame(rows)
    shift.to_csv(os.path.join(HERE, "bounds_shift.csv"), index=False)

    print("\n" + "=" * 74)
    print("DISTRIBUTION SHIFT (unweighted training)")
    print("=" * 74)
    print("  kappa = sup_bins mu/pi;  bias term = 2B(1 - 1/kappa^2), in units")
    print("  of B. 'eval_top100' is the measure the experiments actually use:")
    print("  uniform over BM25's top 100, the rerank depth.\n")
    for mu_name in MU:
        sub = shift[shift["mu"] == mu_name]
        print(f"-- mu = {mu_name} "
              f"({', '.join(f'{v:g}' for v in MU[mu_name])}) --")
        print(sub[["miner", "kappa", "TV_mu_pi", "bias_term_over_B"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()

    # ---------- the miner-dependent part of the bound --------------------
    print("=" * 74)
    print("MINER-DEPENDENT PART OF THE BOUND")
    print("=" * 74)
    print("  Holding the student class, n and B fixed across miners, the only")
    print("  terms that differ are the floor H/(2log2) and the shift bias")
    print("  inside the square root. With B = 1 and the other terms written")
    print("  as a common constant c = eps_F + 4 Rad_n + 2B sqrt(log(2/d)/2n):\n")
    print("     bound(miner) = H/(2log2) + zeta * sqrt( c + 2(1 - 1/kappa^2) )\n")

    comp = []
    for m in miners:
        h = float(ent.loc[m, "mean_entropy_nats"])
        pi = np.array([float(rb.loc[m, f"frac_{b}"]) for b in BINS])
        k, _ = kappa_tv(pi, MU["eval_top100"])
        bias = float("inf") if not np.isfinite(k) else 2.0 * (1.0 - 1.0 / k**2)
        row = {"miner": m, "floor": floor_upper(h), "kappa_eval": k,
               "bias_over_B": bias}
        for loss, z in ZETA.items():
            for c in (0.0, 0.1):
                key = f"{loss}_c{c:g}"
                row[key] = (
                    float("inf") if not np.isfinite(bias)
                    else floor_upper(h) + z * math.sqrt(c + bias)
                )
        comp.append(row)
    cdf = pd.DataFrame(comp)
    cdf.to_csv(os.path.join(HERE, "bounds_summary.csv"), index=False)
    print(cdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nNotes on what these numbers do and do not say:")
    print("  * kappa from four rank bins is a LOWER bound on the true")
    print("    sup dmu/dpi: binning averages the ratio within each bin.")
    print("  * random has kappa = inf against any mu with mass on BM25's top")
    print("    100, because it places essentially no mass there (0.0000 and")
    print("    0.0000 in bins 1-10 and 11-100). The coverage assumption fails,")
    print("    so the unweighted bound is vacuous for random -- which is the")
    print("    correct conclusion, not a defect of the estimate.")
    print("  * the bound is an upper bound; a larger value is a weaker")
    print("    guarantee, not a prediction of worse effectiveness.")
    print(f"\nwrote bounds_floor.csv, bounds_shift.csv, bounds_summary.csv")


if __name__ == "__main__":
    main()
