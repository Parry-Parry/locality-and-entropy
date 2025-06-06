# teacher_student_distribution_analysis.py
"""
Robust utilities for analysing **teacher–student retrieval relationships** and
complementary dataset‑level statistics.

Analyses provided
-----------------
1. **Entropy–performance correlation** (`corr --type entropy`)
2. **KL‑divergence–performance correlation** (`corr --type kl`)
3. **Mean supremum distance per training dataset** (`supremum`)
4. **Mean query entropy per run file** (`entropy`)

Features
--------
* **Dataset filtering**: optional substring filter so only run files whose
  names contain that substring (case‑insensitive) are processed.
* Lightweight dependencies: **pandas**, **numpy**, **scipy**, **json**, **gzip**.
* Streams JSONL training data; avoids loading large artefacts into memory.
* Single‑file module with CLI (`python teacher_student_distribution_analysis.py --help`).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

###############################################################################
# Helper utilities
###############################################################################

def _shift_to_positive(arr: np.ndarray) -> np.ndarray:
    """Shift array so that all elements become strictly positive."""
    if (arr <= 0).all():
        arr = arr + abs(arr.min()) + 1e-6
    return arr


def _safe_probability_distribution(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary score array to a valid probability distribution."""
    scores = _shift_to_positive(scores.astype(float))
    return scores / scores.sum()

###############################################################################
# Run‑file parsing
###############################################################################

def read_trec_run_scores(path: str | Path) -> Dict[str, Dict[str, float]]:
    """Parse a TREC run file (plain text **or** *.gz*).

    Returns
    -------
    dict
        Mapping `qid → {docid: score}`.
    """
    run: Dict[str, Dict[str, float]] = {}
    # Auto‑detect compression by file extension
    open_fn = gzip.open if str(path).endswith(".gz") else open
    mode = "rt" if open_fn is gzip.open else "r"
    with open_fn(path, mode, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 6:
                continue  # skip malformed lines
            qid, docid, score = parts[0], parts[2], float(parts[4])
            run.setdefault(qid, {})[docid] = score
    return run

###############################################################################
# Information‑theoretic measures
###############################################################################

def shannon_entropy(scores: List[float]) -> float:
    if not scores:
        return float("nan")
    p = _safe_probability_distribution(np.asarray(scores))
    return float(-np.sum(p * np.log2(p)))


def kl_divergence_teacher_student(
    teacher: Dict[str, float],
    student: Dict[str, float],
    eps: float = 1e-12,
) -> float:
    docs = set(teacher) | set(student)
    t = np.array([teacher.get(d, 0.0) for d in docs])
    s = np.array([student.get(d, 0.0) for d in docs])
    P = _safe_probability_distribution(t)
    Q = _safe_probability_distribution(s)
    P = np.where(P == 0, eps, P)
    Q = np.where(Q == 0, eps, Q)
    return float(np.sum(P * np.log2(P / Q)))

###############################################################################
# 1. Entropy–performance correlation
###############################################################################

def compute_entropy_dataframe(run_path: str | Path) -> pd.DataFrame:
    q2scores = read_trec_run_scores(run_path)
    rows = [(qid, shannon_entropy(list(scores.values()))) for qid, scores in q2scores.items()]
    return pd.DataFrame(rows, columns=["qid", "entropy"])


def _ensure_long_format(results_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a *wide* metrics table (one row per run, metric columns) to *long*.

    If `results_df` already has columns [`name`, `qid`, `measure`, `value`] it is
    returned unchanged. Otherwise we expect a format like:
        name, AP(rel=2), nDCG@10, ...
    with **one row per run** (aggregated metrics). We reshape it to long so that
    downstream code expecting [`name`, `measure`, `value`] continues to work.
    Note: *query‑level* analyses still require a `qid` column.
    """
    if {"measure", "value"}.issubset(results_df.columns):
        return results_df  # already long

    if "name" not in results_df.columns:
        raise ValueError("results_df must contain a 'name' column.")

    metric_cols = [c for c in results_df.columns if c != "name"]
    long_df = results_df.melt(id_vars="name", value_vars=metric_cols,
                              var_name="measure", value_name="value")
    return long_df


def correlate_student_with_entropy(
    results_df: pd.DataFrame,
    teacher_name: str,
    measure: str,
    teacher_run_path: str | Path,
    output_tsv: str | Path | None = None,
) -> pd.DataFrame:
    """Correlate teacher entropy with *query‑level* student effectiveness.

    Requirements
    ------------
    * `results_df` **must** contain per‑query metrics – i.e. columns [`qid`,
      `name`, `measure`, `value`]. If your CSV is *aggregated per run* (one row
      per model), you need to provide a per‑query metrics file instead.
    """
    # Detect wide vs long format
    if "qid" not in results_df.columns:
        print(f"{results_df.columns}")
        raise ValueError("`correlate_student_with_entropy` requires a per‑query "
                         "metrics table with a 'qid' column. Your results.csv "
                         "appears aggregated per run. Provide per‑query metrics "
                         "or use other analyses that work on run‑level data.")

    results_df = _ensure_long_format(results_df)

    df_metric = results_df[(results_df["measure"] == measure)].copy()
    ent_df = compute_entropy_dataframe(teacher_run_path).set_index("qid")

    teacher_scores = (
        df_metric[df_metric["name"] == teacher_name][["qid", "value"]]
        .rename(columns={"value": "teacher_value"})
        .set_index("qid")
    )
    teacher_aug = teacher_scores.join(ent_df, how="inner")

    rows: List[Tuple[str, float, float, float, float]] = []
    for student_name, grp in df_metric[df_metric["name"] != teacher_name].groupby("name"):
        student_scores = grp[["qid", "value"]].set_index("qid")
        merged = teacher_aug.join(student_scores, how="inner").rename(columns={"value": "student_value"})
        if merged.shape[0] < 2:
            print(f"Warning: not enough data for {student_name} (only {merged.shape[0]} rows)")
            continue
        r_p, p_p = stats.pearsonr(merged["entropy"], merged["student_value"])
        r_s, p_s = stats.spearmanr(merged["entropy"], merged["student_value"])
        rows.append((student_name, r_p, p_p, r_s, p_s))

    out_df = pd.DataFrame(rows,
                          columns=["student_name", "r_pearson", "p_pearson", "r_spearman", "p_spearman"])\
              .sort_values("r_pearson", ascending=False, ignore_index=True)
    if output_tsv is not None:
        out_df.to_csv(output_tsv, sep="	", index=False)
    return out_df

###############################################################################
# 2. KL‑divergence–performance correlation
###############################################################################

def _locate_student_run(model_name: str, run_dir: Path, dataset_filter: str | None) -> Path | None:
    pat = re.compile(re.escape(model_name), re.IGNORECASE)
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        fname = p.name
        if dataset_filter and dataset_filter.lower() not in fname.lower():
            print(f"Warning: skipping {fname} because it does not match the dataset filter '{dataset_filter}'")
            continue
        if pat.search(fname):
            return p
    return None


def compute_kl_dataframe(teacher_run: str | Path, student_run: str | Path) -> pd.DataFrame:
    teacher = read_trec_run_scores(teacher_run)
    student = read_trec_run_scores(student_run)
    rows = [
        (qid, kl_divergence_teacher_student(teacher[qid], student[qid]))
        for qid in (set(teacher) & set(student))
    ]
    return pd.DataFrame(rows, columns=["qid", "KL"]) if rows else pd.DataFrame(columns=["qid", "KL"])


def correlate_student_with_kl(
    results_df: pd.DataFrame,
    teacher_name: str,
    measure: str,
    run_dir: str | Path,
    teacher_run_path: str | Path,
    output_tsv: str | Path | None = None,
    dataset_filter: str | None = None,
) -> pd.DataFrame:
    df_metric = results_df[results_df["measure"] == measure]
    run_dir = Path(run_dir)

    rows: List[Tuple[str, float, float, float, float, float]] = []
    for student_name, grp in df_metric[df_metric["name"] != teacher_name].groupby("name"):
        student_run = _locate_student_run(student_name, run_dir, dataset_filter)
        if student_run is None:
            print(f"Warning: no run file found for {student_name} in {run_dir}")
            continue
        kl_df = compute_kl_dataframe(teacher_run_path, student_run)
        if kl_df.empty:
            continue
        student_scores = grp[["qid", "value"]].set_index("qid")
        merged = kl_df.set_index("qid").join(student_scores, how="inner").rename(columns={"value": "student_value"})
        if merged.shape[0] < 2:
            continue
        mean_kl = merged["KL"].mean()
        r_p, p_p = stats.pearsonr(merged["KL"], merged["student_value"])
        r_s, p_s = stats.spearmanr(merged["KL"], merged["student_value"])
        rows.append((student_name, mean_kl, r_p, p_p, r_s, p_s))

    out_df = pd.DataFrame(rows, columns=["student_name", "mean_KL", "r_pearson", "p_pearson", "r_spearman", "p_spearman"])\
              .sort_values("mean_KL", ignore_index=True)
    if output_tsv is not None:
        out_df.to_csv(output_tsv, sep="\t", index=False)
    return out_df

###############################################################################
# 3. Mean supremum distance per training dataset
###############################################################################

def _load_distance_matrix(path: str | Path) -> Dict[str, Dict[str, float]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _stream_training_instances(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if (line := line.strip()):
                obj = json.loads(line)
                yield obj["doc_id_a"], obj.get("doc_id_b", [])


def compute_mean_supremum(training_path: str | Path, score_path: str | Path) -> float:
    distances = _load_distance_matrix(score_path)
    suprema: List[float] = []
    for doc_a, doc_b_list in _stream_training_instances(training_path):
        if not doc_b_list:
            continue
        vals = [distances.get(doc_a, {}).get(db, float("nan")) for db in doc_b_list]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            suprema.append(max(vals))
    return float(np.mean(suprema)) if suprema else float("nan")


def batch_mean_supremum(
    training_paths: Iterable[str | Path],
    score_paths: Iterable[str | Path],
    output_tsv: str | Path | None = None,
) -> pd.DataFrame:
    """Compute mean supremum distance for *parallel* lists of training/score files.

    Parameters
    ----------
    training_paths : iterable of paths to JSONL files
    score_paths    : iterable of paths to gz‑JSON matrices (same length)
    output_tsv     : optional file path – write the result as TSV if given

    Returns
    -------
    DataFrame with columns [dataset, mean_supremum]
    """
    training_paths = list(training_paths)
    score_paths = list(score_paths)
    if len(training_paths) != len(score_paths):
        raise ValueError("training_paths and score_paths must be equal length")

    rows: List[Tuple[str, float]] = []
    for train_p, score_p in zip(training_paths, score_paths):
        mean_sup = compute_mean_supremum(train_p, score_p)
        rows.append((Path(train_p).stem, mean_sup))

    out_df = pd.DataFrame(rows, columns=["dataset", "mean_supremum"])
    if output_tsv is not None:
        out_df.to_csv(output_tsv, sep="	", index=False)
    return out_df

###############################################################################
# 4. Mean query entropy per run file
###############################################################################

def compute_mean_entropy(run_path: str | Path) -> float:
    """Mean Shannon entropy across all queries in a single run file."""
    ent_df = compute_entropy_dataframe(run_path)
    return float(ent_df["entropy"].mean())

def compute_variance_entropy(run_path: str | Path) -> float:
    """Variance of Shannon entropy across all queries in a single run file."""
    ent_df = compute_entropy_dataframe(run_path)
    return float(ent_df["entropy"].var())


def batch_mean_entropy(
    run_paths: Iterable[str | Path],
    output_tsv: str | Path | None = None,
) -> pd.DataFrame:
    rows = [(Path(rp).stem, compute_mean_entropy(rp)) for rp in run_paths]
    out_df = pd.DataFrame(rows, columns=["run", "mean_entropy"])
    if output_tsv is not None:
        out_df.to_csv(output_tsv, sep="	", index=False)
    return out_df

def batch_variance_entropy(
    run_paths: Iterable[str | Path],
    output_tsv: str | Path | None = None,
) -> pd.DataFrame:
    rows = [(Path(rp).stem, compute_variance_entropy(rp)) for rp in run_paths]
    out_df = pd.DataFrame(rows, columns=["run", "variance_entropy"])
    if output_tsv is not None:
        out_df.to_csv(output_tsv, sep="	", index=False)
    return out_df

###############################################################################
# CLI entry‑point
###############################################################################

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Teacher–student analysis toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Correlation subcommand
    p_corr = sub.add_parser("corr", help="Entropy or KL correlation analysis")
    p_corr.add_argument("results_csv")
    p_corr.add_argument("teacher_run")
    p_corr.add_argument("teacher_name")
    p_corr.add_argument("run_dir")
    p_corr.add_argument("--measure", default="nDCG@10")
    p_corr.add_argument("--type", choices=["entropy", "kl"], default="entropy")
    p_corr.add_argument("--dataset", help="Substring to filter run files (KL only)")
    p_corr.add_argument("--out", help="TSV output path")

    # Supremum subcommand
    p_sup = sub.add_parser("supremum", help="Mean supremum distance analysis")
    p_sup.add_argument("training_files", nargs="+")
    p_sup.add_argument("--score_files", nargs="+", required=True)
    p_sup.add_argument("--out", help="TSV output path")

    # Entropy subcommand
    p_ent = sub.add_parser("entropy", help="Mean query entropy per run file")
    p_ent.add_argument("run_files", nargs="+")
    p_ent.add_argument("--out", help="TSV output path")

    args = parser.parse_args()

    if args.cmd == "corr":
        df = pd.read_csv(args.results_csv, sep='\t')
        if args.type == "entropy":
            res = correlate_student_with_entropy(
                df, args.teacher_name, args.measure, args.teacher_run, args.out
            )
        else:  # KL
            res = correlate_student_with_kl(
                df,
                args.teacher_name,
                args.measure,
                args.run_dir,
                args.teacher_run,
                args.out,
                dataset_filter=args.dataset,
            )
        print(res.to_string(index=False))

    elif args.cmd == "supremum":
        res = batch_mean_supremum(args.training_files, args.score_files, args.out)
        print(res.to_string(index=False))

    else:  # entropy
        res = batch_variance_entropy(args.run_files, args.out)
        print(res.to_string(index=False))


if __name__ == "__main__":
    _cli()
