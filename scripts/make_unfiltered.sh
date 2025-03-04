DATASET="msmarco-passage/train/triples-small"
DATA_DIR='sigir25-implicit-diff/data_train'
OUTPUT_DIR='sigir25-implicit-diff/data'

# Mine negatives
python sigir25-implicit-diff/src/implicit/data_processing/create_harder_nofilter.py \
    --triples_file "${OUTPUT_DIR}/triples.jsonl" \
    --data_folder $DATA_DIR 