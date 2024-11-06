OUTPUT_DIR='sigir25-implicit-diff/data'
CHECKPOINT='sigir25-implicit-diff/checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"
# Create output directory
mkdir -p $OUTPUT_DIR

# Dump triples to file
python sigir25-implicit-diff/src/implicit/data_processing/get_triples.py \
    --dataset $DATASET \
    --out_file "${OUTPUT_DIR}/triples.jsonl.gz"

# Mine negatives
python sigir25-implicit-diff/src/implicit/data_processing/mine_negatives.py \
    --file "${OUTPUT_DIR}/triples.jsonl.gz" \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \
    --model_name_or_path $CHECKPOINT 