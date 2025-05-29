OUTPUT_DIR='random_data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"
# Create output directory
mkdir -p $OUTPUT_DIR

# Dump triples to file
python src/implicit/data_processing/get_triples.py \
    --dataset $DATASET \
    --out_file "${OUTPUT_DIR}/triples.jsonl.gz"

# Mine negatives
python src/implicit/data_processing/mine_negatives.py \
    --file "${OUTPUT_DIR}/triples.jsonl.gz" \
    --dataset $DATASET \
    --depth 10000 \
    --out_dir $OUTPUT_DIR \
    --model_name_or_path $CHECKPOINT \
    --use_random  