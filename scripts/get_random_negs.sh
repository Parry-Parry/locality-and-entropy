OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"
# Create output directory
mkdir -p $OUTPUT_DIR

# Mine negatives
python src/implicit/data_processing/mine_random_negatives.py \
    --file "${OUTPUT_DIR}/triples.jsonl.gz" \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \
    --model_name_or_path $CHECKPOINT 