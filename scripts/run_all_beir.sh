#!/bin/bash

MODEL_DIRECTORY="checkpoints"
OUTPUT_DIRECTORY="beir_runs"
BATCH_SIZE=512

mkdir -p $OUTPUT_DIRECTORY

CMD="python -m implicit.batch_run_beir \
--model_directory $MODEL_DIRECTORY \
--output_directory $OUTPUT_DIRECTORY"

# Execute the command
eval $CMD