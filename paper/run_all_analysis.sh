#!/usr/bin/env bash
# run_all.sh  –  batch all teacher–student analyses
# -------------------------------------------------
# Prerequisites
#   pip install pandas numpy scipy
#
# Directory layout assumed
#   ├─ paper/analysis.py   # already on disk
#   ├─ perquery_metrics/msmarco-passage-trec-dl-2019-judged-metrics.tsv.gz              # per-query metrics from PyTerrier
#   ├─ runs/                    # *.trec run files   (teacher + students)
#   └─ data/                    # training JSONL  +  gz-JSON score matrices
#
#   TEACHER_NAME        – value of the teacher in perquery_metrics/msmarco-passage-trec-dl-2019-judged-metrics.tsv.gz/name column
#   TEACHER_RUN         – path to the teacher’s .trec inside runs/
#   TRAIN_JSONL_*       – one or more training datasets
#   SCORE_GZ_*          – matching gz-JSON distance matrices
#
# Usage
#   chmod +x run_all.sh
#   ./run_all.sh

set -euo pipefail

# ---------------------------------------------------------------------
# USER CONFIGURATION  (EDIT THESE LINES)
# ---------------------------------------------------------------------

TEACHER_RUN="runs/msmarco_passage_trec_dl_2019_judged_crossencoder.res.gz"     # <-- the teacher's run file
DATASET="msmarco_passage_trec_dl_2019_judged" # <-- dataset name
TEACHER_NAME="$DATASET.crossencoder"        # <-- adjust to your CSV column
MEASURE="nDCG@10"                   # <-- per-query metric to analyse

# Training / score file pairs  (extend or modify as needed)
TRAIN_JSONL=(
  "data/bm25.16.jsonl"
  "data/crossencoder.16.jsonl"
  "data/random.16.jsonl"
  "data/ensemble.16.jsonl"
)
SCORE_GZ=(
  "data/bm25.scores.json.gz"
  "data/crossencoder.scores.json.gz"
  "data/random.scores.json.gz"
  "data/ensemble.all.scores.json.gz"
)
# ---------------------------------------------------------------------

shopt -s extglob nocaseglob
if [[ -n "$DATASET" ]]; then
  RUN_FILES=(runs/*"$DATASET"*.@(trec|trec.gz|res|res.gz))
else
  RUN_FILES=(runs/*.@(trec|trec.gz|res|res.gz))
fi
shopt -u extglob nocaseglob

mkdir -p analysis

echo "1. Entropy–performance correlation"
python paper/analysis.py corr \
  perquery_metrics/msmarco-passage-trec-dl-2019-judged-metrics.tsv.gz "$TEACHER_RUN" "$TEACHER_NAME" runs/ \
  --measure "$MEASURE" --type entropy \
  --out analysis/entropy_correlations.tsv

echo "2. KL-divergence–performance correlation"
python paper/analysis.py corr \
  perquery_metrics/msmarco-passage-trec-dl-2019-judged-metrics.tsv.gz "$TEACHER_RUN" "$TEACHER_NAME" runs/ \
  --measure "$MEASURE" --type kl \
  --out analysis/kl_correlations.tsv

echo "3. Mean query entropy for every run"
python paper/analysis.py entropy \
  "${RUN_FILES[@]}" \
  --out analysis/mean_entropy.tsv

echo "4. Mean supremum distances for training datasets"
python paper/analysis.py supremum \
  "${TRAIN_JSONL[@]}" \
  --score_files "${SCORE_GZ[@]}" \
  --out analysis/supremum.tsv

echo "All analyses finished. Results are in the 'analysis/' directory."
