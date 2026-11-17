#!/usr/bin/env python3
"""Step 5b: locate every equivalence claim in the thesis and judge its status.

Quotes are pulled verbatim from recompute/thesis_extract.txt by line number, so
no claim text is retyped. Status is assigned from recompute/results/summary.csv.

NOTE ON `file`: only the PDF was available, so locations are line numbers in the
pdftotext extract, not the LaTeX source. Andrew must supply the .tex tree for
true file:line references before these edits are applied.

Usage:
    python recompute/make_claims.py
"""

from __future__ import annotations

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOMPUTE = os.path.join(REPO, "recompute")
EXTRACT = os.path.join(RECOMPUTE, "thesis_extract.txt")

HOLDS_WIDER = "holds at a wider margin"
FAILS = "does not hold"
DESCRIPTIVE = "does not hold (describes the defective method)"

# (first_line, last_line, status, note) -- 1-indexed, inclusive
CLAIMS = [
    (49, 50, FAILS,
     "Abstract. 'statistically equivalent' at the stated 1% margin is not "
     "supported: 25/1623 comparisons are equivalent at 1%."),
    (647, 648, HOLDS_WIDER,
     "H1.2 statement. Supported only at a margin of roughly 10%, and not at "
     "all on TREC DL at any defensible margin given 43/54 queries."),
    (2991, 2991, FAILS,
     "Chapter 3 forward-reference to 'the equivalence of BM25-sampled and "
     "expensively mined negatives'."),
    (3062, 3062, FAILS,
     "Chapter 4 introduction restating in/out-of-domain statistical "
     "equivalence."),
    (3558, 3560, FAILS,
     "Methods. States alpha = 0.05 and 'means bounded at 1%'. The code used "
     "alpha = 0.10 and a margin of 0.1% of the observed difference "
     "(AUDIT.md sections 3 and 10). Neither parameter matches."),
    (3630, 3631, FAILS,
     "Table 4.1 caption. '(1% bound, alpha = 0.05)' describes neither the "
     "implemented test nor the appendix definition."),
    (3744, 3746, FAILS,
     "Section 4.3 entropy discussion. 'all cases degrade out-of-domain ... "
     "induces overfitting' compares a 24-collection micro-averaged baseline "
     "against 8-collection macro-averaged quartile rows. Like-for-like, 5 of "
     "8 quartile rows improve on the baseline; only Outlier degrades. See "
     "REPORT.md section 6. Independent of the TOST defect."),
    (3755, 3757, HOLDS_WIDER,
     "Discussion. 'statistically equivalent ... under all semi-supervised "
     "criteria tested' holds for no margin uniformly; at 10% agreement with "
     "the published labels is still only 67.2%."),
    (3761, 3763, FAILS,
     "Discussion. 'does not yield statistically distinguishable improvements' "
     "conflates non-significance with equivalence."),
    (3764, 3769, HOLDS_WIDER,
     "Verdict on H1.2. 'statistically equivalent ... under every "
     "semi-supervised criterion tested' fails at 1%; a ~10% margin is the "
     "most that the per-query data support, and TREC DL supports none."),
    (3786, 3788, HOLDS_WIDER,
     "Scoping paragraph. The scope limits are fine; the word 'equivalence' "
     "still needs its margin restated."),
    (4294, 4300, DESCRIPTIVE,
     "Appendix TOST definition. Defines theta = |mu2 - mu1| * eps (margin as "
     "a fraction of the observed difference) and 'independent samples'. This "
     "documents the defect rather than the intended method, and its "
     "'confidence 1 - 2alpha compensating for multiple hypothesis testing' is "
     "also incorrect: two one-sided tests at alpha give overall level alpha. "
     "Replaced by dropin/appendix_tost.tex."),
    (7650, 7652, HOLDS_WIDER,
     "Conclusions chapter restatement of the H1.2 verdict."),
]


def main() -> None:
    with open(EXTRACT, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    rows = []
    for first, last, status, note in CLAIMS:
        quote = " ".join(l.strip() for l in lines[first - 1 : last])
        rows.append(
            {
                "file": "thesis_aparry-2.pdf (via recompute/thesis_extract.txt)",
                "line": first if first == last else f"{first}-{last}",
                "quoted_sentence": quote,
                "status": status,
                "note": note,
            }
        )

    df = pd.DataFrame(rows)
    dest = os.path.join(RECOMPUTE, "claims.csv")
    df.to_csv(dest, index=False)

    print(f"{len(df)} claim sites")
    print(df["status"].value_counts().to_string())
    print(f"\nwrote {os.path.relpath(dest, REPO)}")


if __name__ == "__main__":
    main()
