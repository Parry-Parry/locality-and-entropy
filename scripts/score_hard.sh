OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"
FILE=$OUTPUT_DIR/ensemble.16.jsonl

python -m implicit.data_processing.get_scores \
    --file $FILE \
    --model_name_or_path $CHECKPOINT \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \
    --name_override "ensemble"
