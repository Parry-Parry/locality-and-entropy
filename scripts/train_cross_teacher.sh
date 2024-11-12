MODEL_NAME="google/electra-base-discriminator"
OUTPUT_DIR="checkpoints"
WANDB_PROJECT="implicit-distillation"
WARMUP_RATIO=0.1
LR=5e-5
FP16=true
SAVE_LIMIT=1
LOSS=$1
GROUP_SIZE=$2
TRIPLE_FILE="data/crossencoder.${GROUP_SIZE}.jsonl"
BATCH_SIZE=$3
GRAD_ACCUM=$4
MAX_STEPS=$5
# optional teacher file is now last argument
TEACHER_FILE=$6

# Build base command
CMD="python -m implicit.train_cat \
--model_name_or_path $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--wandb_project $WANDB_PROJECT \
--warmup_ratio $WARMUP_RATIO \
--learning_rate $LR \
--save_total_limit $SAVE_LIMIT \
--loss_fn $LOSS \
--num_train_epochs 1 \
--training_dataset_file $TRIPLE_FILE \
--group_size $GROUP_SIZE \
--per_device_train_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $GRAD_ACCUM \
--ir_dataset "msmarco-passage/train/triples-small" \
--logging_steps 1000 \
--save_steps 100000 \
--dataloader_num_workers 4 \
--fp16 t
--report_to wandb"

# Add max steps argument only if it's defined
if [ ! -z "$MAX_STEPS" ]; then
    CMD="$CMD --max_steps $MAX_STEPS"
fi

# Add teacher file argument only if it's defined
if [ ! -z "$TEACHER_FILE" ]; then
    CMD="$CMD --teacher_file $TEACHER_FILE"
fi

# Execute the command
eval $CMD