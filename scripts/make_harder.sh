DATASET="msmarco-passage/train/triples-small"
OUTPUT_DIR='data'

# Mine negatives
python src/implicit/data_processing/create_harder_negatives.py \
    --triples_file "${OUTPUT_DIR}/triples.jsonl" \
    --data_folder $OUTPUT_DIR 