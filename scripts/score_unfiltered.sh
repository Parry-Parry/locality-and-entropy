OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"
FILE="data_train/ensemble.unfiltered.16.jsonl"

python -m implicit.data_processing.get_scores \
    --file $FILE \
    --model_name_or_path $CHECKPOINT \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \
    --name_override "ensemble.unfiltered"
