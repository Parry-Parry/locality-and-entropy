OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"

FILE=$OUTPUT_DIR/msmarco-hard-negatives.jsonl.gz

python -m implicit.data_processing.get_scores \
    --file $FILE \
    --model_name_or_path $CHECKPOINT \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \
