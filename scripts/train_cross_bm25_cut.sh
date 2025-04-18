MODEL_NAME="google/electra-base-discriminator"
OUTPUT_DIR="entropy_checkpoints"
WANDB_PROJECT="negatives"
WARMUP_RATIO=0.1
LR=1e-5
FP16=true
SAVE_LIMIT=1
LOSS=$1
GROUP_SIZE=$2
MODE=$3
TEACHER_FILE="data/bm25.crossencoder.scores.json.gz"
TRIPLES_FILE="data/$MODE.jsonl"

# Define constant
TOTAL_DOCS=12000000
BASE_BATCH_SIZE=8

# how many steps to get, TOTAL_DOCS / (BATCH_SIZE * GROUP_SIZE)
PER_BATCH_DOCS=$((BASE_BATCH_SIZE * GROUP_SIZE))
TOTAL_STEPS=$((TOTAL_DOCS / PER_BATCH_DOCS))

# Build base command
CMD="python -m implicit.train_cat \
--model_name_or_path $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--wandb_project $WANDB_PROJECT \
--warmup_ratio $WARMUP_RATIO \
--max_steps $TOTAL_STEPS \
--learning_rate $LR \
--save_total_limit $SAVE_LIMIT \
--loss_fn $LOSS \
--num_train_epochs 1 \
--training_dataset_file $TRIPLE_FILE \
--group_size $GROUP_SIZE \
--per_device_train_batch_size $BASE_BATCH_SIZE \
--gradient_accumulation_steps 2 \
--ir_dataset "msmarco-passage/train/triples-small" \
--logging_steps 1000 \
--save_steps 100000 \
--dataloader_num_workers 4 \
--fp16 t \
--report_to wandb
--test_dataset_file "data/msmarco-passage-trec-dl-2019-judged-bm25.run.gz" \
--test_ir_dataset "msmarco-passage/trec-dl-2019/judged" \
"

# if loss function != lce, add teacher file

if [[ "$LOSS" != "lce" ]]; then
    CMD="$CMD --teacher_file $TEACHER_FILE"
fi

# Execute the command
eval $CMD