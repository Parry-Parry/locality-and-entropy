OUTPUT_DIR='sigir25-implicit-diff/random_data'
CHECKPOINT='sigir25-implicit-diff/checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"
# Create output directory
mkdir -p $OUTPUT_DIR

# Mine negatives
python sigir25-implicit-diff/src/implicit/data_processing/mine_negatives.py \
    --file "${OUTPUT_DIR}/triples.jsonl.gz" \
    --dataset $DATASET \
    --depth 10000 \
    --out_dir $OUTPUT_DIR \
    --model_name_or_path $CHECKPOINT \
    --random  