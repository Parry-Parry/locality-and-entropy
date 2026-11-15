# PROTOCOL — Chapter 4 equivalence recompute

**Frozen before any test was run.** Committed prior to Step 4 (plan rule 2). The
margin, test type, unit of analysis and comparison set below are fixed. If they
appear wrong after seeing results, the result is to stop and report — not to
amend this file.

Authority for the statistical choices: the recompute plan. Authority for the
comparison set and table structure: the original analysis, as recorded in
`AUDIT.md`, per Andrew's instruction that **table shape must be identical**.

---

## Test

Paired TOST on per-query differences

> d_q = candidate_q − baseline_q

aligned by query ID. Implemented once in `recompute/tost.py::paired_tost`,
validated by `recompute/test_tost.py` (6 tests, including agreement with
`statsmodels.stats.weightstats.ttost_paired` to 1e-10 and a 5,000-simulation
size check at the margin boundary). No other equivalence test is used.

## Margin

> θ = ε · mean(baseline),  with **ε = 0.01**

as stated in Section 4.2.5 of the thesis. This is the primary, pre-stated
margin. It is computed from the baseline mean of the queries entering that
comparison, never from the observed difference.

## Level

α = 0.05. Equivalence is concluded iff **both** one-sided tests reject at α,
i.e. `max(p_lower, p_upper) < α`, which is identical to the 90 % confidence
interval of mean(d) lying inside (−θ, θ).

Note this differs from the original code in two ways: the original used α = 0.10,
and it applied the **inverted** rule `p_lower > α and p_upper > α` (AUDIT.md §8).

## Also reported, for every comparison

- the 90 % confidence interval of mean(d);
- the smallest supported margin **ε\* = max(|CI low|, |CI high|) / mean(baseline)**,
  so a reader may apply any margin they prefer;
- a standard paired two-sided t-test of mean(d) = 0, separating "shown
  different" from "inconclusive";
- Holm-adjusted TOST p-values across all comparisons **within each table**,
  alongside the unadjusted values.

## Comparisons

Exactly those the original analysis made (AUDIT.md §6): **all unordered pairs**
of miners within each (collection group, loss, architecture) cell. No comparison
is added or dropped.

- Main tables: pairs drawn from `DOMAIN_ORDER = ["Random", "BM25",
  "Cross-Encoder", "Ensemble"]` → 6 pairs per cell.
- Entropy tables: pairs drawn from `SUBSET_ORDER` as defined in
  `tables/generate_entropy.py`.

The original's baseline choice is unambiguous — it made all-pairs comparisons,
not candidate-vs-reference — so the plan's BM25-reference fallback does not
apply.

**Orientation convention.** TOST is symmetric in the pair, but θ and ε\* are
defined relative to a baseline mean, so the assignment must be deterministic.
For each unordered pair, the **baseline** is whichever member appears **earlier**
in `DOMAIN_ORDER` (resp. `SUBSET_ORDER`), and the candidate is the later member.
This is fixed by the declared order and does not depend on row order in the data
— unlike the original, which used `sub_la['domain'].unique()`, i.e. data order
(`evaluate_main.py:199`).

## BEIR

- **Primary: per dataset.** Each BEIR collection is tested separately, matching
  `generate_beir.py`'s grouping and avoiding the mixing of score scales.
- **Secondary: pooled** over all BEIR queries, clearly labelled as such, because
  pooling combines datasets with very different score scales.

The main tables (`table.tex`) carry a single BEIR column that the original
produced by pooling (`evaluate_main.py:79-83, 168, 186`). To keep the table shape
identical, that column is populated from the **pooled** analysis and is labelled
as secondary wherever it is reported. The per-dataset primary results populate
the BEIR summary tables, which are already per-dataset.

## Unit of analysis

The query.

## Per-query scoring conventions

trec_eval conventions, via `ir_measures`:

- evaluate only queries with at least one judged relevant document;
- a query present in the qrels but missing from a run scores 0;
- metrics: nDCG@10 and MAP. Following `evaluate_main.py:103, 111`, MAP uses
  `rel=2` for TREC DL 2019/2020 and `rel=1` for BEIR.

## Retained from the original, under protest

**Connected-component labelling.** After thresholding, equivalence classes are
formed as connected components of the pairwise-equivalence graph and labelled
A–D (`generate_main.py:29`, `generate_beir.py:43`, `generate_entropy.py:102`).
Pairwise equivalence is **not transitive**, so a component label can assert
equivalence between two systems whose own pairwise test failed or was never run.

This is retained because Andrew requires the table shape to be identical. It is
recorded here and in `REPORT.md` as a known limitation of the presentation, and
the underlying pairwise results — which carry no such assumption — are published
in full in `recompute/results/summary.csv` so any reader can bypass the labels.

## Determinism

Seeds are fixed in all scripts. Package versions are recorded in
`recompute/ENVIRONMENT.txt`. No GPU is used for Steps 1–5.
