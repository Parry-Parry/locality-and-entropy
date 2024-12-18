OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"

# Mine negatives
python src/implicit/data_processing/mine_negatives_from_graph.py \
    --file "${OUTPUT_DIR}/triples.jsonl.gz" \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \