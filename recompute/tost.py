"""Paired TOST reference implementation for the Chapter 4 recompute.

This is the ONLY equivalence test used anywhere in recompute/. It is the
implementation specified in the recompute plan, reproduced verbatim.
"""

import numpy as np
from scipy import stats


def paired_tost(candidate, baseline, eps=0.01, alpha=0.05):
    """Paired TOST of mean(candidate - baseline) against margin eps * mean(baseline).

    candidate and baseline are per-query scores aligned by query ID.
    """
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(baseline, dtype=float)
    assert a.shape == b.shape and a.ndim == 1
    d = a - b
    n = d.size
    theta = eps * b.mean()
    mean_d = d.mean()
    sd = d.std(ddof=1)

    if sd == 0.0:  # identical systems: no variance, equivalence is exact
        p_lower = p_upper = 0.0 if abs(mean_d) < theta else 1.0
        lo = hi = mean_d
    else:
        p_lower = stats.ttest_1samp(d, -theta, alternative="greater").pvalue
        p_upper = stats.ttest_1samp(d, theta, alternative="less").pvalue
        half = stats.t.ppf(1 - alpha, n - 1) * sd / np.sqrt(n)
        lo, hi = mean_d - half, mean_d + half          # 90% CI when alpha = 0.05

    p_tost = max(p_lower, p_upper)
    p_diff = stats.ttest_rel(a, b).pvalue if sd > 0 else 1.0
    return {
        "n": n,
        "mean_baseline": b.mean(),
        "mean_diff": mean_d,
        "theta": theta,
        "ci_low": lo,
        "ci_high": hi,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < alpha),
        "eps_star": max(abs(lo), abs(hi)) / b.mean(),
        "p_diff_two_sided": p_diff,
    }
