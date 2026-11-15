# AUDIT: the original equivalence analysis

Audited commit: `e61012d` ("make it"), branch `main`.
All files cited below are tracked and **byte-identical to HEAD** (`git diff HEAD -- tables/`
is empty), so this audit describes exactly the code that produced the committed
`tost_*.tsv` outputs.

Evidence numbers in this document are produced by `recompute/audit_evidence.py`
(output: `recompute/results/audit_evidence.csv`). No number here was typed by hand.

---

## 1. Where the equivalence tests live

There are **two independent TOST implementations** and **three** consumers that
turn p-values into the table superscripts.

| Role | File | Function | Lines |
|---|---|---|---|
| TOST producer (main tables) | `tables/evaluate_main.py` | `tost` | 15–48 |
| TOST caller (main tables) | `tables/evaluate_main.py` | `main` | 184–241 |
| TOST producer (entropy tables) | `tables/evaluate_entropy.py` | `tost` | 16–25 |
| TOST caller (entropy tables) | `tables/evaluate_entropy.py` | `main` | 178–205 |
| TOST producer + caller (BEIR) | `tables/generate_beir.py` | `tost` / `generate_table` | 20–26 / 80–102 |
| Equivalence labelling | `tables/generate_main.py` | `annotate_equivalence` | 19–40 |
| Equivalence labelling | `tables/generate_beir.py` | `annotate_equivalence` | 28–55 |
| Equivalence labelling | `tables/generate_entropy.py` | `annotate_equivalence` | 67–110 |

## 2. Statistical calls and their arguments

All three producers call **statsmodels**, not SciPy:

```python
import statsmodels.stats.weightstats as ws
...
pval, (t_low, p_low, _), (t_high, p_high, _) = ws.ttost_ind(x, y, low_eq, high_eq)
```

`ws.ttost_ind` is the **independent two-sample** TOST. The paired routine
(`ws.ttost_paired`) is never called anywhere in the repository.

## 3. The margin θ

`tables/evaluate_main.py:39-43` (identical logic at `evaluate_entropy.py:21-23`):

```python
    # observed difference in means
    diff_mean = np.mean(x) - np.mean(y)
    # equivalence margin = bound_pct% of |diff_mean|
    bound = abs(diff_mean) * (bound_pct / 100.0)
    low_eq, high_eq = -bound, bound
```

Three separate defects in four lines:

1. **The margin is derived from the data.** θ is a function of the observed
   difference in the very samples being tested. An equivalence margin must be
   fixed before seeing results; here it shrinks automatically whenever two
   systems happen to score similarly.
2. **It is a fraction of the difference, not of the baseline.** Section 4.2.5 of
   the thesis defines θ = ε · µ_baseline. The code uses a fraction of
   |µ_x − µ_y|, a quantity typically two orders of magnitude smaller than
   µ_baseline.
3. **The percentage is 10× smaller than documented.** The signature is
   `bound_pct=0.1` while the docstring (`evaluate_main.py:26-28`) states
   *"default is 1.0, i.e. ±1 %"*. With `bound_pct=0.1`, `bound_pct / 100.0`
   = 0.001, so θ = 0.1 % of an already-tiny difference — effectively zero.

`tables/generate_beir.py:20` uses a **different** margin again — fixed absolute
bounds, unrelated to ε or to any mean:

```python
def tost(x, y, low_eq=-0.01, high_eq=0.01):
```

So the margin definition is not even consistent between the BEIR tables and the
rest of the thesis.

## 4. Paired or unpaired

**Unpaired**, in all cases — and in the main tables this is a deliberate discard
of pairing that the code has already computed. `tables/evaluate_main.py:206-227`
aligns the two systems by query id and builds matched arrays:

```python
                        ids1 = set(sub.loc[sub['domain'] == d1, 'qid'])
                        ids2 = set(sub.loc[sub['domain'] == d2, 'qid'])
                        common_qids = sorted(ids1 & ids2)
...
                        x = np.array([val_map1[q] for q in common_qids])
                        y = np.array([val_map2[q] for q in common_qids])

                        p, p_lo, p_hi = tost(x, y)
```

`x[i]` and `y[i]` are the same query. The arrays are correctly paired and are
then passed to the *independent*-samples test, which throws the pairing away and
estimates the variance of the difference as if the two samples were unrelated.
Because per-query retrieval scores are strongly correlated across systems, this
inflates the standard error substantially and makes every interval wider than it
should be.

`tables/generate_beir.py:86-87` is worse — it does not align at all:

```python
                x = sub[sub.domain == d1]["value"].values
                y = sub[sub.domain == d2]["value"].values
```

Values are taken in whatever order the frame happens to hold them, with no qid
correspondence and no check that the two domains cover the same queries.

## 5. Per-query scores or aggregates

**Per-query.** All producers operate on per-query values (`value` column of
`perquery_*.tsv.gz`), so the input granularity was right. The defect is in how
those per-query values are tested, not in their provenance.

## 6. Which comparisons were made

**All unordered pairs of miners, within each (group, loss, arch) cell** — not
candidate-vs-baseline. `tables/evaluate_main.py:204-205`:

```python
                for i, d1 in enumerate(doms):
                    for d2 in doms[i+1:]:
```

with `DOMAIN_ORDER = ["Random", "BM25", "Cross-Encoder", "Ensemble"]`
(`generate_main.py:9`), i.e. 6 pairs per cell. The entropy tables do the same
over `SUBSET_ORDER` (`evaluate_entropy.py:180-184`).

One special case: `evaluate_main.py:187-198` hardcodes `p_lower = 1.0` and
`p_upper = 1.0` for `arch == "CE-Teacher"` with `domain1 == domain2 == "BM25"`.
Under the (inverted) labelling rule below, a hardcoded p of 1.0 is read as
"equivalent". It is a self-pair, so it only ever forms a self-loop in the graph
and does not merge distinct nodes.

## 7. How BEIR was handled

**Both ways, inconsistently.**

- `tables/evaluate_main.py:79-83` puts every non-DL dataset into a single
  `"beir"` group, concatenates all of them into one frame (line 168), and then
  groups only by `(loss, arch)` at line 186. BEIR is therefore **pooled across
  all datasets** — queries from datasets with very different score scales
  (e.g. ArguAna vs TREC-COVID) enter one unpaired t-test together. This is the
  file that produced the committed `main_intermediate/tost_beir.tsv`.
- `tables/generate_beir.py:82` groups by `["dataset_id", "loss", "arch"]`, i.e.
  **per dataset**, and recomputes its own TOST in memory with the fixed ±0.01
  bounds — never reading `tost_beir.tsv` at all.

## 8. The decisive defect: the equivalence criterion is inverted

All three labelling functions add a graph edge when **both one-sided p-values are
large**. `tables/generate_main.py:24-28`:

```python
        for _, row in sub.iterrows():
            if (row["measure"] == metric
                    and row["p_lower"] > alpha
                    and row["p_upper"] > alpha):
                G.add_edge(row["domain1"], row["domain2"])
```

Identically at `generate_beir.py:39-42` and `generate_entropy.py:94-98`.

TOST concludes equivalence when **both one-sided tests reject**, i.e.
`max(p_lower, p_upper) < α`. The condition above is the logical opposite: it
fires when the data provide *least* evidence of equivalence. It is the classic
error of reading a non-significant result as proof of no difference — and here it
is applied to a test specifically designed to avoid that error.

This interacts catastrophically with the near-zero margin of §3. A margin of
0.1 % of an observed difference drives both one-sided p-values toward 1
(observed medians: p_lower ≈ 0.41–0.51, p_upper ≈ 0.49–0.70 across the six
output files). Under the correct rule those p-values mean "nothing is
established"; under the coded rule they mean "everything is equivalent". The two
bugs reinforce one another rather than cancelling.

### Measured effect

Recounting the committed `tost_*.tsv` files at the code's own α = 0.10
(`recompute/audit_evidence.py`):

| file | comparisons | "equivalent" as coded | equivalent under correct criterion |
|---|---|---|---|
| `main_intermediate/tost_dl19.tsv` | 97 | 75 | 0 |
| `main_intermediate/tost_dl20.tsv` | 97 | 61 | 0 |
| `main_intermediate/tost_beir.tsv` | 97 | 50 | 0 |
| `ent_intermediate/tost_cat_dl19.tsv` | 60 | 57 | 0 |
| `ent_intermediate/tost_cat_dl20.tsv` | 60 | 50 | 0 |
| `ent_intermediate/tost_cat_beir.tsv` | 84 | 31 | 0 |
| **total** | **495** | **324** | **0** |

Of 495 comparisons, the original code labelled 324 equivalent. Using the *same*
stored p-values with the correct inequality, **none** is equivalent.

## 9. A further defect in labelling: transitivity

After thresholding, all three consumers take connected components of the
equivalence graph (`generate_main.py:29`, `generate_beir.py:43`,
`generate_entropy.py:102`) and assign a shared letter to each component. Pairwise
equivalence is not transitive: if A~B and B~C both pass, A and C receive the same
label even when A vs C was tested and failed, or was never tested. The published
superscripts therefore assert more than any pairwise test supports.

Per Andrew's instruction the table *shape* is preserved in the recompute, so
connected-component labelling is retained; this is recorded as a known
limitation and is restated in `PROTOCOL.md` and `REPORT.md` rather than silently
fixed.

## 10. α

`α = 0.10` throughout (`generate_main.py:19`, `generate_beir.py:14`,
`generate_entropy.py:68`, `evaluate_entropy.py:272`). Section 4.2.5 of the thesis
and the recompute protocol both specify α = 0.05.

---

## Summary

The original analysis computed, for every unordered pair of miners within each
(collection group, loss, architecture) cell, an **independent**-samples TOST on
**correctly paired but deliberately unpaired-ised** per-query scores, against an
equivalence margin defined as **0.1 % of the observed difference in means**
rather than the pre-stated θ = ε · µ_baseline with ε = 0.01 — a margin that is
both data-dependent and effectively zero — at **α = 0.10** rather than 0.05, with
BEIR **pooled across datasets** in one code path and **per-dataset with fixed
±0.01 bounds** in another. It then labelled a pair "equivalent" when **both
one-sided p-values exceeded α**, which is the exact negation of the TOST decision
rule. Because the degenerate margin forces both p-values close to 1, the inverted
rule marked 324 of 495 comparisons as equivalent; under the correct rule, applied
to the very same p-values, the count is 0. Every equivalence superscript in
Tables 4.1–4.9 is therefore an artefact of a sign error amplified by a broken
margin, and none of them is evidence for H1.2 as the thesis states it.
