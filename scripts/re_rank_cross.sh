#!/bin/bash

MODEL_DIRECTORY="checkpoints"
OUTPUT_DIRECTORY="runs"
BATCH_SIZE=512
TEXT_FIELD="text"
# Required arguments
# Optional arguments with empty defaults
OVERWRITE=$1

mkdir -p $OUTPUT_DIRECTORY

DATASETS=("msmarco-passage/trec-dl-2019/judged" "msmarco-passage/trec-dl-2020/judged" "beir/trec-covid")

for dataset in "${DATASETS[@]}"; do
    # Replace slashes with hyphens in the dataset name
    # Build base command
    CMD="python -m implicit.batch_run_topics \
    --ir_dataset $dataset \
    --model_directory $MODEL_DIRECTORY \
    --output_directory $OUTPUT_DIRECTORY"

    # depending on dataset set topics_or_res
    if [[ $dataset == *"trec-dl-2019"* ]]; then
        TOPICS_OR_RES="runs/msmarco_passage_trec_dl_2019_judged_crossencoder.res.gz"
    elif [[ $dataset == *"trec-dl-2020"* ]]; then
        TOPICS_OR_RES="runs/msmarco_passage_trec_dl_2020_judged_crossencoder.res.gz"
    elif [[ $dataset == *"trec-covid"* ]]; then
        TOPICS_OR_RES="runs/beir_trec_covid_crossencoder.res.gz"
    fi

    CMD="$CMD --topics_or_res $TOPICS_OR_RES"
   
    if [ ! -z "$BATCH_SIZE" ]; then
        CMD="$CMD --batch_size $BATCH_SIZE"
    fi

    if [ ! -z "$TEXT_FIELD" ]; then
        CMD="$CMD --text_field $TEXT_FIELD"
    fi

    if [ ! -z "$OVERWRITE" ]; then
        CMD="$CMD --dont_overwrite $OVERWRITE"
    fi

    # Execute the command
    eval $CMD
done