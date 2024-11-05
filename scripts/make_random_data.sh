OUTPUT_DIR='sigir25-implicit-diff/data'
CHECKPOINT='sigir25-implicit-diff/checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"
# Create output directory
mkdir -p $OUTPUT_DIR

# Mine negatives
python sigir25-implicit-diff/src/implicit/data_processing/mine_random_negatives.py \
    --file "${OUTPUT_DIR}/triples.jsonl.gz" \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \
    --model_name_or_path $CHECKPOINT \
    --cache "${OUTPUT_DIR}/crossencoder-cache"