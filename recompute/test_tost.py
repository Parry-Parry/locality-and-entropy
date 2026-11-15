"""Validation of recompute/tost.py (plan Step 3d).

Must pass before any real comparison is run.

    uv run pytest recompute/test_tost.py -v
"""

import numpy as np
import pytest
from statsmodels.stats.weightstats import ttost_paired

from tost import paired_tost

ALPHA = 0.05
EPS = 0.01


def _rng(seed: int) -> np.random.Generator:
    """Fixed-seed generator (plan rule 6)."""
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# 1. Agreement with an independent implementation
# --------------------------------------------------------------------------


def test_agrees_with_statsmodels_on_200_random_samples():
    """p_tost must match statsmodels' ttost_paired to within 1e-10."""
    rng = _rng(20260101)
    for i in range(200):
        n = int(rng.integers(10, 200))
        baseline = rng.uniform(0.05, 0.95, size=n)
        candidate = baseline + rng.normal(0.0, 0.05, size=n)

        got = paired_tost(candidate, baseline, eps=EPS, alpha=ALPHA)
        theta = got["theta"]

        expected = ttost_paired(
            np.asarray(candidate, dtype=float),
            np.asarray(baseline, dtype=float),
            low=-theta,
            upp=theta,
        )[0]

        assert got["p_tost"] == pytest.approx(expected, abs=1e-10), (
            f"sample {i}: n={n} ours={got['p_tost']!r} statsmodels={expected!r}"
        )


# --------------------------------------------------------------------------
# 2. CI equivalence
# --------------------------------------------------------------------------


def test_equivalent_iff_ci_inside_margin():
    """`equivalent` is True exactly when -theta < ci_low and ci_high < theta."""
    rng = _rng(20260102)
    seen_true = seen_false = False

    for _ in range(300):
        n = int(rng.integers(10, 150))
        baseline = rng.uniform(0.05, 0.95, size=n)
        # vary the shift so both branches are exercised
        shift = rng.uniform(-0.05, 0.05)
        candidate = baseline + shift + rng.normal(0.0, 0.03, size=n)

        r = paired_tost(candidate, baseline, eps=EPS, alpha=ALPHA)
        ci_inside = (-r["theta"] < r["ci_low"]) and (r["ci_high"] < r["theta"])

        assert r["equivalent"] == ci_inside, r

        seen_true |= r["equivalent"]
        seen_false |= not r["equivalent"]

    assert seen_true and seen_false, "test did not exercise both outcomes"


# --------------------------------------------------------------------------
# 3. Sanity cases
# --------------------------------------------------------------------------


def test_identical_arrays_are_equivalent():
    x = _rng(20260103).uniform(0.05, 0.95, size=50)
    r = paired_tost(x, x, eps=EPS, alpha=ALPHA)

    assert r["mean_diff"] == 0.0
    assert r["equivalent"] is True
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0


def test_five_percent_shift_not_equivalent_at_1pct_but_is_at_10pct():
    """A +5% shift with n=50 and small noise: fails eps=0.01, passes eps=0.10."""
    rng = _rng(20260104)
    n = 50
    baseline = rng.uniform(0.2, 0.8, size=n)
    candidate = baseline + 0.05 * baseline.mean() + rng.normal(0.0, 0.002, size=n)

    tight = paired_tost(candidate, baseline, eps=0.01, alpha=ALPHA)
    loose = paired_tost(candidate, baseline, eps=0.10, alpha=ALPHA)

    assert tight["equivalent"] is False, tight
    assert loose["equivalent"] is True, loose


def test_swapping_candidate_and_baseline_negates_mean_diff():
    rng = _rng(20260105)
    n = 80
    baseline = rng.uniform(0.1, 0.9, size=n)
    candidate = baseline + rng.normal(0.01, 0.05, size=n)

    fwd = paired_tost(candidate, baseline, eps=EPS, alpha=ALPHA)
    rev = paired_tost(baseline, candidate, eps=EPS, alpha=ALPHA)

    assert rev["mean_diff"] == pytest.approx(-fwd["mean_diff"], abs=1e-12)


# --------------------------------------------------------------------------
# 4. Size check by simulation
# --------------------------------------------------------------------------


def test_rejection_rate_at_boundary_is_at_most_alpha_plus_001():
    """True difference exactly theta: rejection rate <= alpha + 0.01 over 5000 sims.

    At the boundary the TOST null is true, so concluding equivalence is a type-I
    error. The nominal rate is alpha; we allow alpha + 0.01 for Monte-Carlo noise.
    """
    rng = _rng(20260106)
    n_sims = 5000
    n = 60
    mu_baseline = 0.5
    theta = EPS * mu_baseline  # true difference sits exactly on the margin

    rejections = 0
    for _ in range(n_sims):
        baseline = np.full(n, mu_baseline) + rng.normal(0.0, 0.10, size=n)
        # shift the difference to exactly theta relative to this sample's mean
        candidate = baseline + theta + rng.normal(0.0, 0.05, size=n)

        r = paired_tost(candidate, baseline, eps=EPS, alpha=ALPHA)
        # theta is recomputed from the sample baseline mean; compare like for like
        if r["equivalent"]:
            rejections += 1

    rate = rejections / n_sims
    assert rate <= ALPHA + 0.01, f"rejection rate {rate:.4f} exceeds {ALPHA + 0.01}"
