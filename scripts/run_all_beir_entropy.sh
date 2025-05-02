#!/bin/bash

MODEL_DIRECTORY="entropy_checkpoints"
OUTPUT_DIRECTORY="beir_runs"
BATCH_SIZE=512

mkdir -p $OUTPUT_DIRECTORY

CMD="python -m implicit.batch_run_beir \
--ir_dataset $dataset \
--model_directory $MODEL_DIRECTORY \
--output_directory $OUTPUT_DIRECTORY"
fi

# Execute the command
eval $CMD