# P2: extra quantities for the corrected theory

Each item below says how it was computed and what it found. Scripts are
rerunnable from the repository root.

Status: items 3, 4, 5 and 6 are complete and need no GPU. Items 1 and 2
(pairwise teacher entropy, rank-bin mismatch) require the teacher model and the
training negatives and are in progress.

| item | what | status |
|---|---|---|
| 1 | pairwise teacher entropy (Bradley-Terry) | arm A: 2 of 4 miners done; arm B blocked |
| 2 | rank-bin mismatch | blocked -- needs a linux box with a v1 PISA index |
| 3 | diameter metric | **done** |
| 4 | kappa-hat summary | **done** |
| 5 | loss definitions | **done** |
| 6 | Table 7.10 | **done** |

---

## 1. Pairwise teacher entropy (Bradley-Terry)

`python recompute/theory/pairwise_entropy.py --all`

h(sigma(g(x) - g(x'))) in nats, averaged over pairs within a query and then
over queries. h -> 0 means the teacher is certain which document is better;
h -> log 2 = 0.6931 means it cannot tell them apart.

**Arm A, pairs from each miner's own training negatives.** No GPU or teacher
inference needed: the teacher scores are already stored in
`data/<miner>.scores.json.gz`.

| miner | mean h (nats) | se | sd | median | % of log 2 | queries |
|---|---|---|---|---|---|---|
| random | 0.6596 | 0.00011 | 0.0703 | 0.6887 | 95.2% | 398,792 |
| BM25 | 0.4292 | 0.00019 | 0.1182 | 0.4318 | 61.9% | 397,734 |
| cross-encoder | 0.2167 | 0.00020 | 0.1269 | 0.1745 | 31.3% | 398,792 |
| ensemble | 0.1794 | 0.00016 | 0.1000 | 0.1493 | 25.9% | 396,664 |

Strictly monotone, and the spread is large: the teacher is almost blind between
random negatives (95% of maximum entropy) and nearly decisive between ensemble
negatives (26%).

**A tension this creates with the chapter's stated mechanism.** Teacher entropy
keeps falling well past the point where effectiveness stops improving. From
BM25 to ensemble, h drops by 58% (0.4292 -> 0.1794), yet Table 4.1 shows no
corresponding effectiveness gain -- BM25 and ensemble are within a point or two
of each other on TREC DL, and the recomputed equivalence tests cannot separate
them at any margin below about 9%. The same is true of the other two quantities
in Table 4.2: kappa-hat falls 12.7 -> 8.3 and Delta-hat 9.86 -> 9.55 over the
same step.

So all three terms of the bound improve monotonically across the mining
cascade while effectiveness plateaus after BM25. The Verdict on H1.2 says the
operative factors are "teacher pairwise entropy and the essential diameter of
the query manifold, neither of which the mining cascade tightens enough to
register". For the diameter that holds -- it moves 3%. For teacher entropy it
does not: the cascade tightens it by more than half and nothing happens to
effectiveness. **The H1.2 conclusion survives (expensive mining does not pay),
but this particular explanation of why does not.** Andrew should decide how to
reword; the measurement is in `pairwise_entropy.csv`.

**A defect found in the training files.** Every query's group is repeated
roughly 99.75 times, with an identical positive and an identical negative set
each time (verified for qid 400296: 90 instances whose negative sets union to
exactly 15 documents). `random.16.jsonl.gz` holds 39,780,811 lines for 398,792
queries. Averaging over lines rather than queries weights each query by its
arbitrary repeat count and understates the standard error by about sqrt(100).
The figures above are per query.

**The ensemble miner covers a different query set.** Its score file holds
500,295 queries against ~398,790 for the other three. If training used the
full set, the ensemble arm of Table 4.1 saw ~25% more training queries than
its comparators, which would confound exactly the comparison H1.2 rests on.
Needs checking against the training invocation. The figures above use each
miner's own queries; item 2 intersects to the 396,621 common to all four.

**Arm B, uniform pairs from BM25's top 1,000: still outstanding.** The v1 PISA
index needed for it exists on the GPU node
(`/mnt/indices/msmarco-passage.pisa`, with the `k1-1.2.b-0.75` variant mining
used), so this is now a matter of running it rather than a blocker.

## 2. Rank-bin mismatch

`python recompute/theory/rank_bins.py --n-queries 2000`

Each miner's training negatives, binned by their rank in BM25's top 1,000 for
the same query (">1000" if absent). BM25 is the index and parameters the mining
itself used (`PisaIndex(...).bm25(k1=1.2, b=0.75)` on
`/mnt/indices/msmarco-passage.pisa`). 2,000 queries sampled with seed 20260101
from the 396,621 common to all four miners; 30,000 negatives per miner.

| miner | 1-10 | 11-100 | 101-1000 | >1000 |
|---|---|---|---|---|
| random | 0.0000 | 0.0000 | 0.0001 | 0.9999 |
| BM25 | 0.0503 | 0.4489 | 0.4940 | 0.0068 |
| cross-encoder | 0.0842 | 0.5329 | 0.3755 | 0.0074 |
| ensemble | 0.0626 | 0.2389 | 0.3143 | 0.3842 |

kappa = max over bins of mu/pi, TV = half the L1 distance, against two target
distributions mu: uniform over the four bins, and top-weighted
(0.4, 0.3, 0.2, 0.1):

| miner | kappa (uniform) | TV (uniform) | kappa (top-weighted) | TV (top-weighted) |
|---|---|---|---|---|
| random | inf | 0.7499 | inf | 0.8999 |
| BM25 | 36.59 | 0.4429 | 14.63 | 0.4429 |
| cross-encoder | 33.78 | 0.4084 | 13.51 | 0.4084 |
| ensemble | 3.99 | 0.1985 | 6.39 | 0.3985 |

Three things follow.

1. **Cross-encoder mining barely moves the sampling region away from BM25's.**
   Their bin profiles are close (TV against uniform 0.4084 vs 0.4429, a
   difference of 0.035) and both place about 99.3% of negatives inside BM25's
   top 1,000. The expensive miner is drawing from nearly the same place the
   cheap one does -- a direct, distributional mechanism for H1.2 that does not
   depend on the entropy argument above. What the cross-encoder changes is
   *which* documents it picks within a similar rank band, shifting mass from
   101-1000 into 11-100.
2. **Random negatives are essentially disjoint from BM25's candidates**
   (99.99% beyond rank 1,000), and kappa is infinite because two bins are
   empty. Reporting kappa for random is not meaningful; TV (0.75 uniform,
   0.90 top-weighted) is.
3. **Ensemble is the genuine outlier**, with 38.4% of its negatives outside
   BM25's top 1,000, and it is much the closest to uniform (TV 0.1985). It is
   the one miner that samples a materially different region -- and it still
   does not deliver a matching effectiveness gain, which is the same plateau
   the entropy result shows.

**Outstanding:** the plan also asks for this analysis with the deployed
bi-encoder's ranking in place of BM25. Not yet run.

## 3. Diameter metric

`python recompute/theory/diameter_check.py`

**Which distance does the code actually compute?** There are three different
implementations in the repository, and they disagree:

| file | function | metric | quantile |
|---|---|---|---|
| `paper/get_supremum.py` | `robust_diameter` | `pdist(..., metric="euclidean")` | 0.95 |
| `paper/get_supremum_adhoc.py` | `robust_diameter` | `pdist(..., metric="cosine")` | 0.99 |
| `paper/analysis.py` | `compute_mean_supremum` | none -- reads a precomputed distance matrix, takes max over negatives then mean | n/a |

`get_supremum.py` is **documented as cosine and computes euclidean**. Its
docstring says "the alpha-quantile of all pairwise cosine distances" and the
inline comment says "pairwise cosine distances", directly above:

```python
    dists = pdist(embeddings, metric="euclidean")
```

**Which reproduces Table 4.2 (9.55 to 10.45)?** Neither `.py` file. The table
comes from `supremum_cosine.json` at the repository root, via
`notebooks/delta.ipynb` cell 3, which computes the mean and variance of the
per-query values at or below their 95th percentile. That reproduces all four
published values exactly:

| source | published | mean of values <= p95 | published +/- | variance of values <= p95 |
|---|---|---|---|---|
| Random | 10.448 | 10.448 | 0.044 | 0.0445 |
| BM25 | 9.862 | 9.862 | 0.205 | 0.2053 |
| Cross-Encoder | 9.593 | 9.593 | 0.251 | 0.2510 |
| Ensemble | 9.546 | 9.546 | 0.233 | 0.2329 |

Two consequences:

1. **The published values cannot be cosine distances.** Cosine distance is
   bounded in `[0, 2]`; 9.55 to 10.45 is outside that range by a factor of
   five. Despite the filename `supremum_cosine.json` and the docstrings, these
   are not cosine distances.
2. **The "+/-" column is a variance, not a standard deviation.** The notebook
   column is literally `var_<=95th`. The standard deviations are 0.211, 0.453,
   0.501 and 0.483 respectively.

`supremum.json` (produced by the euclidean code path) gives means of 17.95 to
18.71 -- and **reverses the ordering across miners**: it ranks
ensemble > crossencoder > bm25 > random, whereas Table 4.2 ranks
random > bm25 > crossencoder > ensemble. So the two files are not two views of
one quantity; at most one of them supports the claim the table makes.

**Not yet settled.** Confirming what `supremum_cosine.json` *is* (L2 on
unnormalised embeddings is the obvious candidate, since L2 on unit vectors is
also bounded by 2) requires regenerating the embeddings. That is the one part
of item 3 still outstanding.

## 4. kappa-hat summary

`python recompute/theory/kappa_summary.py`

The published values use the same recipe as the diameter column -- mean and
variance of per-query values at or below the 95th percentile -- and reproduce
exactly, again confirming that **"+/-" is a variance**:

| source | published | mean <= p95 | published +/- | variance <= p95 | true SD |
|---|---|---|---|---|---|
| Random | 14.202 | 14.202 | 556.251 | 556.251 | 23.585 |
| BM25 | 12.747 | 12.747 | 461.588 | 461.588 | 21.485 |
| Cross-Encoder | 11.116 | 11.116 | 353.018 | 353.018 | 18.789 |
| Ensemble | 8.276 | 8.276 | 165.234 | 165.234 | 12.854 |

Relative error treating the published "+/-" as a variance: **0.0000**. As a
standard deviation: 0.945.

**Robust summary (requested):**

| source | median | Q1 | Q3 | IQR | max |
|---|---|---|---|---|---|
| Random | 6.605 | 3.689 | 15.319 | 11.629 | 294,197,103 |
| BM25 | 5.796 | 3.343 | 13.452 | 10.109 | 170,642,895 |
| Cross-Encoder | 5.014 | 2.929 | 11.646 | 8.717 | 92,850,937 |
| Ensemble | 4.087 | 2.505 | 8.998 | 6.493 | 73,142,346 |

The ordering the thesis relies on (Random > BM25 > Cross-Encoder > Ensemble)
**survives** under the median, so the qualitative claim holds. But the
distribution is extremely heavy-tailed -- maxima of 10^8 against medians under
7 -- so the mean is not a usable summary and the p95 trim is doing almost all
the work. The median and IQR above are the honest statistic.

Two defects in `paper/get_kappa.py` that explain the tail:

```python
            nu = scores / scores.sum()
            mu = np.full_like(nu, 1.0 / len(nu))
            kappa = np.max(mu / nu)
```

1. `scores` are raw cross-encoder logits, which are routinely **negative**, so
   `nu` is not a probability distribution: entries can be negative and the
   denominator can be near zero. `mu / nu` is then negative or explosive.
2. Documents missing from the score file default to `0.0`
   (`.get(str(doc_id), 0.0)`), making the corresponding `mu / nu` infinite.

**Provenance note.** Table 4.2's BM25 row does *not* come from
`data/kappa.json`, whose BM25 entry gives 1.158 and drops 67,688 of 398,749
queries as non-finite. It comes from the separate `data/bm25_kappa.json`
(398,792 queries, none dropped), which matches the published mean and variance
exactly. The two files disagree by an order of magnitude for the same
quantity; which is correct needs Andrew's call.

## 5. Loss definitions

The losses are not defined in this repository. `src/implicit/train_cat.py`
imports from the external `rankers` package; only
`src/implicit/modelling/loss.py` is local and it defines
`onesided_margin_mse`, which the run names show was **not** used (they say
`MarginMSE`, `RankNet`, `KL_Divergence`, `LCE`).

Read from `rankers` **0.1.0** as installed at
`/opt/anaconda3/lib/python3.12/site-packages/rankers`. Caveat: nothing pins the
version used for training, so this is the installed version, not a verified
match.

**RankNet: hard labels, and binary cross-entropy, not KL.**
`rankers/train/loss/torch/listwise.py:29-53`:

```python
    def __init__(self, reduction="mean", temperature=1.0):
        super().__init__(reduction)
        self.temperature = temperature
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred: Tensor, labels: Tensor = None, **kwargs) -> Tensor:
        _, g = pred.shape
        i1, i2 = torch.triu_indices(g, g, offset=1)
        pred_diff = pred[:, i1] - pred[:, i2]
        if labels is None:
            labels = torch.zeros_like(pred_diff)
            labels[:, 0] = 1.0
        else:
            label_diff = labels[:, i1] - labels[:, i2]
            labels = (label_diff > 0).float()

        return self.bce(pred_diff, labels)
```

`labels = (label_diff > 0).float()` **binarises the teacher scores**: it uses
hard labels `1{g(x) > g(x')}`, discarding the magnitude of the teacher's
preference. The objective is `BCEWithLogitsLoss`, i.e. binary cross-entropy on
the score differences, not KL. All pairs within the group are used
(`triu_indices`, offset 1), not just positive-against-negative.

Note also that `temperature` is stored but **never used** in `forward` -- it is
a dead parameter, so any temperature passed to RankNet silently did nothing.

**KL: genuine KL on soft teacher targets.**
`listwise.py:11-26`:

```python
    def __init__(self, reduction="batchmean", temperature=1.0):
        ...
        self.kl_div = torch.nn.KLDivLoss(reduction=self.reduction)

    def forward(self, pred: Tensor, labels: Tensor, **kwargs) -> Tensor:
        return self.kl_div(
            F.log_softmax(pred / self.temperature, dim=1),
            F.softmax(labels / self.temperature, dim=1),
        )
```

Soft targets via `softmax(labels / T)`, `reduction="batchmean"` (the correct
normalisation for KL), temperature actually applied, default 1.0.

So RankNet and KL differ in **two** ways, not one: hard versus soft targets,
and cross-entropy versus KL.

**MarginMSE: no scaling factor, mean reduction.**
`rankers/train/loss/torch/pairwise.py:9-23`:

```python
def residual(x):
    return x[:, 0].unsqueeze(1) - x[:, 1:]


@register_loss("margin_mse")
class MarginMSELoss(BaseLoss):
    """Margin MSE loss with residual calculation."""

    name = "MarginMSE"

    def forward(self, pred: Tensor, labels: Tensor, **kwargs) -> Tensor:
        residual_pred = residual(pred)
        residual_label = residual(labels)
        return F.mse_loss(residual_pred, residual_label, reduction=self.reduction)
```

No multiplicative factor. `self.reduction` is `BaseLoss`'s default `"mean"`
(`rankers/train/loss/torch/__init__.py:48`), so it is a **mean over all
batch x (group-1) residual elements**, not a sum. The residual is
positive-minus-each-negative, so the loss sees `group_size - 1` terms per
instance.

## 6. Table 7.10 (Chapter 7)

`python recompute/theory/table_7_10_check.py`

**The repeated blocks are a copy error.** Every `Rel` row in Table 7.10 is
byte-identical to the `Abs` row with the same (retriever, position, generator):
**30 of 30**, across 5 retrievers (BM25, ColBERT, monoT5, Contriever, TAS-B),
2 positions and 3 generators, matching in all 10 printed fields -- nDCG@10, RR
and MRPR together with their deltas and significance markers, **and the
optimised hyperparameter alpha\***.

The evidence that this is an error rather than a property of the design:

1. **The two schemes are defined differently.** Section 7.4:
   > Absolute Position (Abs-P): places a span of text before, in the middle or
   > after **a document**
   >
   > Relative Position (Rel-P): places a span before and after **the most
   > salient sentence** in a document

   "Before the document" and "before the most salient sentence" coincide only
   when the salient sentence happens to be the first one. Over a whole
   collection they cannot agree on every query, let alone to three decimals.
2. **alpha\* is identical too.** It is an optimised per-condition
   hyperparameter; two genuinely different interventions arriving at the same
   optimum in all 30 cells is not credible.
3. **The chapter's own narrative asserts a difference.** Section 7.4 argues
   that "by injecting a span after the most salient text ('after', determined
   by relative position), we reduce the effect of a sequence on the overall
   relevance" -- a claim about relative positioning specifically, which
   identical numbers cannot support.
4. `Rel` has only Before and After, consistent with the definition; the
   duplication looks like the Abs Before/After blocks were pasted in to fill
   the Rel rows.

**The two-decimal deltas.** Fifteen deltas print to 2dp where the rest of the
column uses 3dp. Of these, the ones the plan asks about:

| retriever | scheme | position | generator | metric | value | delta | implied baseline |
|---|---|---|---|---|---|---|---|
| monoT5 | Abs | Before | GPT-3.5 | nDCG@10 | 0.634 | +0.19 | 0.444 |
| TAS-B | Abs | Before | Llama-2 | nDCG@10 | 0.662 | +0.19 | 0.472 |
| TAS-B | Abs | Before | Static | nDCG@10 | 0.662 | +0.18 | 0.482 |
| TAS-B | Abs | Before | Llama-2 | RR | 0.879 | +0.17 | 0.709 |

(each duplicated in the corresponding `Rel` row). Full list in
`table_7_10_two_dp.csv`.

The baselines above are **implied by subtraction**, not read from source data.
Note the formatting is not a simple trailing-zero rule: `+0.050` and `+0.000`
appear at 3dp in the same column as `+0.14`, so the 2dp values did not come
from the same formatter as the rest. That is consistent with the deltas being
assembled from more than one source.

**Outstanding.** The Chapter 7 source data is not in this repository, so the
raw values behind those deltas could not be read directly. Confirming them, and
confirming the duplication against the underlying runs, needs that project's
data -- see "Needs Andrew" in `../REPORT.md`.
