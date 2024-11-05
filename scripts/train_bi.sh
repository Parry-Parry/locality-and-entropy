MODEL_NAME="bert-base-uncased"
OUTPUT_DIR="checkpoints"
WANDB_PROJECT="implicit-distillation"
WARMUP_RATIO=0.1
LR=5e-5
FP16=true
SAVE_LIMIT=1
LOSS=$1
TRIPLE_FILE=$2
GROUP_SIZE=$3
BATCH_SIZE=$4
GRAD_ACCUM=$5
# optional teacher file is now last argument
TEACHER_FILE=$6

# Build base command
CMD="python -m implicit.train \
--model_name_or_path $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--wandb_project $WANDB_PROJECT \
--warmup_ratio $WARMUP_RATIO \
--lr $LR \
--save_total_limit $SAVE_LIMIT \
--loss_fn $LOSS \
--triple_file $TRIPLE_FILE \
--group_size $GROUP_SIZE \
--batch_size $BATCH_SIZE \
--gradient_accumulation_steps $GRAD_ACCUM \
--fp16"

# Add teacher file argument only if it's defined
if [ ! -z "$TEACHER_FILE" ]; then
    CMD="$CMD --teacher_file $TEACHER_FILE"
fi

# Execute the command
eval $CMD