#!/bin/bash

MODEL_DIRECTORY="checkpoints"
OUTPUT_DIRECTORY="runs"
BATCH_SIZE=512
TEXT_FIELD="text"
# Required arguments
IR_DATASET=$1
# Optional arguments with empty defaults
TOPICS_OR_RES=$2
OVERWRITE=$3

mkdir -p $OUTPUT_DIRECTORY

# Build base command
CMD="python -m implicit.batch_run_topics \
--ir_dataset $IR_DATASET \
--model_directory $MODEL_DIRECTORY \
--output_directory $OUTPUT_DIRECTORY"

# Add optional arguments only if they're defined
if [ ! -z "$TOPICS_OR_RES" ]; then
    CMD="$CMD --topics_or_res $TOPICS_OR_RES"
fi

if [ ! -z "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

if [ ! -z "$TEXT_FIELD" ]; then
    CMD="$CMD --text_field $TEXT_FIELD"
fi

# Execute the command
eval $CMD