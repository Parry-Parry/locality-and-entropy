# teacher_student_distribution_analysis.py
"""
Comprehensive utilities for analysing **teacher–student retrieval relationships**
plus complementary dataset‑level statistics.

Included analyses
-----------------
1. **Entropy–performance correlation** (optionally writes TSV)
2. **KL‑divergence–performance correlation** (optionally writes TSV)
3. **Mean supremum distance per training dataset** (optionally writes TSV)
4. **Mean query entropy per run file** (optionally writes TSV)

Only lightweight dependencies are used: **pandas**, **numpy**, **scipy**,
**json**, and **gzip**.
All functions avoid loading large artefacts fully into memory where practical.
Each analysis can be invoked programmatically or via the built‑in CLI.
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
# Helpers
###############################################################################

def _shift_to_positive(arr: np.ndarray) -> np.ndarray:
    """Shift array so that all values become strictly positive."""
    if (arr <= 0).all():
        arr = arr + abs(arr.min()) + 1e-6
    return arr


def _safe_probability_distribution(scores: np.ndarray) -> np.ndarray:
    """Convert an array of (possibly non‑positive) scores to probabilities."""
    scores = _shift_to_positive(scores.astype(float))
    return scores / scores.sum()

###############################################################################
# TREC‑run parsing
###############################################################################

def read_trec_run_scores(path: str | Path) -> Dict[str, Dict[str, float]]:
    """Return {qid: {docid: score}} parsed from a TREC run file."""
    run: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 6:
                continue
            qid, docid, score = parts[0], parts[2], float(parts[4])
            run.setdefault(qid, {})[docid] = score
    return run

###############################################################################
# Information‑theoretic measures
###############################################################################

def shannon_entropy(scores: List[float]) -> float:
    """Shannon entropy H(P) (base‑2) for a single query score list."""
    if not scores:
        return float("nan")
    p = _safe_probability_distribution(np.asarray(scores))
    return float(-np.sum(p * np.log2(p)))


def kl_divergence_teacher_student(
    teacher: Dict[str, float],
    student: Dict[str, float],
    eps: float = 1e-12,
) -> float:
    """D_KL(P‖Q) where P=teacher, Q=student (base‑2)."""
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
    data = [(qid, shannon_entropy(list(scores.values()))) for qid, scores in q2scores.items()]
    return pd.DataFrame(data, columns=["qid", "entropy"])


def correlate_student_with_entropy(
    results_df: pd.DataFrame,
    teacher_name: str,
    measure: str,
    teacher_run_path: str | Path,
    output_tsv: str | Path | None = None,
) -> pd.DataFrame:
    """Correlate each student model's query performance with teacher entropy."""
    df_metric = results_df[results_df["measure"] == measure].copy()
    ent_df = compute_entropy_dataframe(teacher_run_path)

    teacher_scores = (
        df_metric[df_metric["name"] == teacher_name][["qid", "value"]]
        .rename(columns={"value": "teacher_value"})
        .set_index("qid")
    )
    teacher_aug = teacher_scores.join(ent_df.set_index("qid"), how="inner")

    rows: List[Tuple[str, float, float, float, float]] = []
    for student_name, grp in df_metric[df_metric["name"] != teacher_name].groupby("name"):
        student_scores = grp[["qid", "value"]].set_index("qid")
        merged = teacher_aug.join(student_scores, how="inner").rename(columns={"value": "student_value"})
        if merged.shape[0] < 2:
            continue
        r_p, p_p = stats.pearsonr(merged["entropy"], merged["student_value"])
        r_s, p_s = stats.spearmanr(merged["entropy"], merged["student_value"])
        rows.append((student_name, r_p, p_p, r_s, p_s))

    out_df = pd.DataFrame(rows, columns=["student_name", "r_pearson", "p_pearson", "r_spearman", "p_spearman"])
    out_df = out_df.sort_values("r_pearson", ascending=False, ignore_index=True)
    if output_tsv is not None:
        out_df.to_csv(output_tsv, sep="\t", index=False)
    return out_df

###############################################################################
# 2. KL‑divergence–performance correlation
###############################################################################

def _locate_student_run(model_name: str, run_dir: Path) -> Path | None:
    pat = re.compile(re.escape(model_name), re.IGNORECASE)
    for p in run_dir.iterdir():
        if p.is_file() and pat.search(p.name):
            return p
    return None


def compute_kl_dataframe(teacher_run: str | Path, student_run: str | Path) -> pd.DataFrame:
    teacher = read_trec_run_scores(teacher_run)
    student = read_trec_run_scores(student_run)
    qids = set(teacher) & set(student)
    data = [(qid, kl_divergence_teacher_student(teacher[qid], student[qid])) for qid in qids]
    return pd.DataFrame(data, columns=["qid", "KL"]) if data else pd.DataFrame(columns=["qid", "KL"])


def correlate_student_with_kl(
    results_df: pd.DataFrame,
    teacher_name: str,
    measure: str,
    run_dir: str | Path,
    teacher_run_path: str | Path,
    output_tsv: str | Path | None = None,
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    df_metric = results_df[results_df["measure"] == measure]

    rows: List[Tuple[str, float, float, float, float, float]] = []
    for student_name, grp in df_metric[df_metric["name"] != teacher_name].groupby("name"):
        student_run = _locate_student_run(student_name, run_dir)
        if student_run is None:
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

    out_df = pd.DataFrame(rows, columns=["student_name", "mean_KL", "r_pearson", "p_pearson", "r_spearman", "p_spearman"])
    out_df = out_df.sort_values("mean_KL", ignore_index=True)
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
            line = line.strip()
            if not line:
                continue
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
    """Compute mean supremum distance for multiple dataset/score pairs."""
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
        out_df.to_csv(output_tsv, sep="\t", index=False)
    return out_df

###############################################################################
# 4. Mean query entropy per run file
###############################################################################

def compute_mean_entropy(run_path: str | Path) -> float:
    """Return mean Shannon entropy across all queries in one run."""
    ent_df = compute_entropy_dataframe(run_path)
    return float(ent_df["entropy"].mean())


def batch_mean_entropy(run_paths: Iterable[str | Path], output_tsv: str | Path | None = None) -> pd.DataFrame:
    rows: List[Tuple[str, float]] = []
    for rp in run_paths:
        rows.append((Path(rp).stem, compute_mean_entropy(rp)))
    out_df = pd.DataFrame(rows, columns=["run", "mean_entropy"])
    if output_tsv is not None:
        out_df.to_csv(output_tsv, sep="\t", index=False)
    return out_df

###############################################################################
# CLI
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
        df = pd.read_csv(args.results_csv)
        if args.type == "entropy":
            res = correlate_student_with_entropy(
                df, args.teacher_name, args.measure, args.teacher_run, args.out
            )
        else:
            res = correlate_student_with_kl(
                df, args.teacher
