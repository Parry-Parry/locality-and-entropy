#!/usr/bin/env bash

DATASETS=("msmarco-passage/trec-dl-2019/judged" "msmarco-passage/trec-dl-2020/judged" "beir/trec-covid")

for dataset in "${DATASETS[@]}"; do
    # Replace slashes with hyphens in the dataset name
    dataset_hyphenated="${dataset//\//-}"
    
    python -m implicit.evaluate \
    --run_dir rerank_splade_runs \
    --eval "$dataset" \
    --out_dir "metrics/${dataset_hyphenated}-metrics.rerank-splade.tsv.gz" \
    --rel 2 \
    --filter
done