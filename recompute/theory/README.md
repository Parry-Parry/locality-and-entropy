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

`python recompute/theory/pairwise_entropy.py --miner <name>`

h(sigma(g(x) - g(x'))) in nats, averaged over pairs within a query and then
over queries. h -> 0 means the teacher is certain which document is better;
h -> log 2 = 0.6931 means it cannot tell them apart.

**Arm A, pairs from each miner's own training negatives** (no GPU needed: the
teacher scores are already stored in `data/<miner>.scores.json.gz`):

| miner | mean h (nats) | se | sd | median | % of log 2 | queries |
|---|---|---|---|---|---|---|
| random | 0.6596 | 0.00011 | 0.0703 | 0.6887 | 95.2% | 398,792 |
| BM25 | 0.4292 | 0.00019 | 0.1182 | 0.4318 | 61.9% | 397,734 |
| cross-encoder | pending | | | | | |
| ensemble | pending | | | | | |

The gap is large and in the direction the theory needs: against random
negatives the teacher is almost blind (95% of maximum entropy), while against
BM25-mined negatives it is far more decisive (62%). A teacher that cannot
separate a pair transmits almost no gradient signal, so this is the mechanism
by which BM25 mining buys most of what expensive mining buys.

cross-encoder and ensemble are not yet computed: the cluster host became
unreachable mid-transfer, so `crossencoder.16.jsonl.gz` and
`ensemble.16.jsonl.gz` are not local.

**A defect found in the training files.** Every query's group is repeated
roughly 99.75 times, with an identical positive and an identical negative set
each time (verified for qid 400296: 90 instances whose negative sets union to
exactly 15 documents). `random.16.jsonl.gz` holds 39,780,811 lines for 398,792
queries. Averaging over lines rather than queries weights each query by its
arbitrary repeat count and understates the standard error by about sqrt(100).
The figures above are per query.

**Arm B, uniform pairs from BM25's top 1,000: not computed.** It needs a BM25
top-1000 ranking over the 398k training queries. None is stored, and mining
retrieved on the fly from a PISA index
(`src/implicit/data_processing/mine_negatives.py`, `PisaIndex.from_dataset`).
`pyterrier-pisa` publishes linux x86_64 wheels only -- no macOS, no arm64 --
and the only PISA index on the cluster is `msmarco-passage-v2-dedup`, a
different corpus from the v1 passage collection used for training. A
substitute BM25 would produce different rankings, which defeats the purpose.

## 2. Rank-bin mismatch

**Blocked, same cause as arm B above.** The analysis is defined in terms of
each negative's rank within BM25's top 1,000, so it needs the same BM25
ranking. Deferred to a linux host with `pyterrier_pisa` and a v1
`msmarco-passage` index.

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
