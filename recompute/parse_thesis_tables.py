#!/usr/bin/env python3
"""Parse reported values and equivalence superscripts out of the thesis text.

Source of truth: thesis_aparry-2.pdf, md5 38879e94ec603194b8564d7ae583077b
(identical to andrew-parry-thesis.pdf), extracted with:

    pdftotext -layout thesis_aparry-2.pdf recompute/thesis_extract.txt

No reported value is typed by hand anywhere in recompute/ (plan rule 3); every
`reported_value` and `original_claim` originates here.

Outputs:
    recompute/reported/table_4_1.csv   main effectiveness + equivalence letters
    recompute/reported/table_4_3.csv   entropy-quartile effectiveness (no letters)
    recompute/reported/table_beir.csv  Tables 4.4-4.9, per-dataset BEIR

Usage:
    python recompute/parse_thesis_tables.py
"""

from __future__ import annotations

import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT = os.path.join(REPO, "recompute", "thesis_extract.txt")
OUTDIR = os.path.join(REPO, "recompute", "reported")

# Table 4.1 column order, read off the three-row header:
#   "TREC DL'19 | TREC DL'20 | BEIR" x "BE | CE" x "nDCG | MAP"
TABLE_4_1_COLUMNS = [
    ("dl19", "BE", "nDCG@10"),
    ("dl19", "BE", "AP(rel=2)"),
    ("dl19", "CE", "nDCG@10"),
    ("dl19", "CE", "AP(rel=2)"),
    ("dl20", "BE", "nDCG@10"),
    ("dl20", "BE", "AP(rel=2)"),
    ("dl20", "CE", "nDCG@10"),
    ("dl20", "CE", "AP(rel=2)"),
    ("beir", "BE", "nDCG@10"),
    ("beir", "CE", "nDCG@10"),
]

# Table 4.3 column order
TABLE_4_3_COLUMNS = [
    ("dl19", "nDCG@10"),
    ("dl19", "AP(rel=2)"),
    ("dl20", "nDCG@10"),
    ("dl20", "AP(rel=2)"),
    ("beir", "nDCG@10"),
]

LOSSES = ["LCE", "RankNet", "marginMSE", "mMSE", "KL"]
LOSS_CANON = {"mMSE": "marginMSE"}

# Table prints the teacher cross-encoder miner as "CE"
DOMAIN_CANON = {
    "Random": "Random",
    "BM25": "BM25",
    "CE": "Cross-Encoder",
    "Ensemble": "Ensemble",
}
# superscript letter -> miner, by position in DOMAIN_ORDER
LETTER_TO_DOMAIN = {
    "a": "Random",
    "b": "BM25",
    "c": "Cross-Encoder",
    "d": "Ensemble",
}

# A cell is either a number optionally followed by equivalence letters
# (lowercase in 4.1, uppercase in 4.4-4.9; pdftotext sometimes inserts spaces
# before the letters), or the literal "n/a" where the thesis has no run.
CELL_RE = re.compile(r"(?:(\d\.\d{3})\s*([a-dA-D]*))|(n/a)")


def _cells(rest: str) -> list[tuple[float | None, str]]:
    """Tokenise a table row body into (value, letters) cells.

    A value of None marks a cell the thesis prints as "n/a".
    """
    out: list[tuple[float | None, str]] = []
    for m in CELL_RE.finditer(rest):
        number, letters, na = m.groups()
        if na:
            out.append((None, ""))
        else:
            out.append((float(number), letters))
    return out

LOSS_DOMAIN_RE = re.compile(
    r"^\s*(" + "|".join(LOSSES) + r")\s+(Random|BM25|CE|Ensemble)\s+(.*)$"
)
LOSS_QUARTILE_RE = re.compile(
    r"^\s*(" + "|".join(LOSSES) + r")\s+(Original|Lower|Inner|Upper|Outlier)\s+(.*)$"
)


def _lines() -> list[str]:
    with open(EXTRACT, encoding="utf-8") as fh:
        return fh.read().splitlines()


def _letters_to_domains(letters: str) -> str:
    """'bcd' -> 'BM25;Cross-Encoder;Ensemble'. Case-insensitive."""
    if not letters:
        return ""
    return ";".join(LETTER_TO_DOMAIN[c] for c in sorted(set(letters.lower())))


def parse_table_4_1(lines: list[str]) -> pd.DataFrame:
    """Rows of Table 4.1: 16 (loss, domain) rows x 10 value columns."""
    start = next(i for i, l in enumerate(lines) if l.startswith("Table 4.1:"))
    rows = []
    for line in lines[start : start + 40]:
        m = LOSS_DOMAIN_RE.match(line)
        if not m:
            continue
        loss, domain, rest = m.groups()
        loss = LOSS_CANON.get(loss, loss)
        domain = DOMAIN_CANON[domain]
        vals = _cells(rest)
        if len(vals) != len(TABLE_4_1_COLUMNS):
            raise ValueError(
                f"Table 4.1: {loss}/{domain}: got {len(vals)} values, "
                f"expected {len(TABLE_4_1_COLUMNS)}\n  line: {line!r}"
            )
        for (collection, arch, metric), (value, letters) in zip(
            TABLE_4_1_COLUMNS, vals
        ):
            rows.append(
                {
                    "table": "4.1",
                    "architecture": arch,
                    "loss": loss,
                    "miner": domain,
                    "collection": collection,
                    "metric": metric,
                    "reported_value": value,
                    "reported_equivalent_to": _letters_to_domains(letters),
                }
            )
    if not rows:
        raise ValueError("Table 4.1: no rows parsed")
    return pd.DataFrame(rows)


def parse_table_4_3(lines: list[str]) -> pd.DataFrame:
    """Table 4.3: entropy-quartile subsets. Carries no equivalence letters."""
    start = next(i for i, l in enumerate(lines) if l.startswith("Table 4.3:"))
    rows = []
    for line in lines[start : start + 40]:
        m = LOSS_QUARTILE_RE.match(line)
        if not m:
            continue
        loss, quartile, rest = m.groups()
        loss = LOSS_CANON.get(loss, loss)
        # strip the leading "H(g) = mean±sd" column before reading values
        rest = re.sub(r"^\s*\d\.\d+\s*±\s*\d\.\d+", "", rest)
        vals = _cells(rest)
        if len(vals) != len(TABLE_4_3_COLUMNS):
            raise ValueError(
                f"Table 4.3: {loss}/{quartile}: got {len(vals)} values, "
                f"expected {len(TABLE_4_3_COLUMNS)}\n  line: {line!r}"
            )
        for (collection, metric), (value, letters) in zip(TABLE_4_3_COLUMNS, vals):
            rows.append(
                {
                    "table": "4.3",
                    "architecture": "CE",
                    "loss": loss,
                    "miner": quartile,
                    "collection": collection,
                    "metric": metric,
                    "reported_value": value,
                    "reported_equivalent_to": _letters_to_domains(letters),
                }
            )
    if not rows:
        raise ValueError("Table 4.3: no rows parsed")
    return pd.DataFrame(rows)


def parse_beir_tables(lines: list[str]) -> pd.DataFrame:
    """Tables 4.4-4.9: per-dataset BEIR nDCG@10, uppercase equivalence letters."""
    specs = [
        ("4.4", "BE"),
        ("4.5", "BE"),
        ("4.6", "BE"),
        ("4.7", "CE"),
        ("4.8", "CE"),
        ("4.9", "CE"),
    ]
    rows = []
    for table, arch in specs:
        start = next(
            (i for i, l in enumerate(lines) if l.startswith(f"Table {table}:")), None
        )
        if start is None:
            print(f"WARNING: Table {table} not found")
            continue

        # bound the window at the next table caption, so rows belonging to the
        # following table are never parsed against this table's header
        nxt = next(
            (
                i
                for i, l in enumerate(lines[start + 1 :], start=start + 1)
                if re.match(r"^Table \d+\.\d+:", l)
            ),
            len(lines),
        )
        window = lines[start:nxt]

        # the header line names the datasets; it follows the caption
        header = None
        for line in window[:8]:
            if line.lstrip().startswith("Loss"):
                header = line
                break
        if header is None:
            print(f"WARNING: Table {table}: no header line")
            continue
        datasets = header.split()[2:]  # drop "Loss" and "Domain"

        for line in window:
            m = LOSS_DOMAIN_RE.match(line)
            if not m:
                continue
            loss, domain, rest = m.groups()
            loss = LOSS_CANON.get(loss, loss)
            domain = DOMAIN_CANON[domain]
            vals = _cells(rest)
            if len(vals) != len(datasets):
                print(
                    f"WARNING: Table {table} {loss}/{domain}: "
                    f"{len(vals)} values vs {len(datasets)} datasets -- skipped"
                )
                continue
            for dataset, (value, letters) in zip(datasets, vals):
                rows.append(
                    {
                        "table": table,
                        "architecture": arch,
                        "loss": loss,
                        "miner": domain,
                        "collection": dataset,
                        "metric": "nDCG@10",
                        "reported_value": value,
                        "reported_equivalent_to": _letters_to_domains(letters),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    lines = _lines()

    t41 = parse_table_4_1(lines)
    t43 = parse_table_4_3(lines)
    tbeir = parse_beir_tables(lines)

    for name, df in [
        ("table_4_1", t41),
        ("table_4_3", t43),
        ("table_beir", tbeir),
    ]:
        dest = os.path.join(OUTDIR, f"{name}.csv")
        df.to_csv(dest, index=False)
        annotated = (df["reported_equivalent_to"] != "").sum() if len(df) else 0
        print(f"{name}: {len(df):4d} cells, {annotated:4d} carry equivalence letters")

    print(f"\nwrote {os.path.relpath(OUTDIR, REPO)}/")


if __name__ == "__main__":
    main()
