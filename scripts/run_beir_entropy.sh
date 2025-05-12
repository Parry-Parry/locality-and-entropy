#!/bin/bash

MODEL_DIRECTORY="entropy_checkpoints"
OUTPUT_DIRECTORY="entropy_beir_runs"
BATCH_SIZE=512
FILE=$1

mkdir -p $OUTPUT_DIRECTORY

CMD="python -m implicit.batch_beir \
--file $FILE \
--model_directory $MODEL_DIRECTORY \
--output_directory $OUTPUT_DIRECTORY"

# Execute the command
eval $CMD