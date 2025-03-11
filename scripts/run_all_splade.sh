#!/bin/bash

OUTPUT_DIRECTORY="data"
BATCH_SIZE=128

mkdir -p $OUTPUT_DIRECTORY

DATASETS=("msmarco-passage/trec-dl-2019/judged" "msmarco-passage/trec-dl-2020/judged" "beir/trec-covid")

for dataset in "${DATASETS[@]}"; do
    # Replace slashes with hyphens in the dataset name
    # Build base command
    CMD="python -m implicit.data_processing.run_splade \
    --ir_dataset $dataset \
    --output_directory $OUTPUT_DIRECTORY"

    # depending on dataset set topics_or_res
    if [[ $dataset == *"trec-dl-2019"* ]]; then
        INDEX_PATH="/nfs/indices/msmarco-passage.splade-ensemble-distill.pisa"
    elif [[ $dataset == *"trec-dl-2020"* ]]; then
        INDEX_PATH="/nfs/indices/msmarco-passage.splade-ensemble-distill.pisa"
    elif [[ $dataset == *"trec-covid"* ]]; then
        INDEX_PATH="/nfs/indices/BEIR/trec-covid/trec-covid.splade-ensemble-distill.pisa"
    fi

    CMD="$CMD --index_path $INDEX_PATH"

    if [ ! -z "$BATCH_SIZE" ]; then
        CMD="$CMD --batch_size $BATCH_SIZE"
    fi

    # Execute the command
    eval $CMD
done