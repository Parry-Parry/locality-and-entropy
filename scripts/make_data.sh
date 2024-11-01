OUTPUT_DIR = 'sigir25-implicit-diff/data'
CHECKPOINT = 'sigir25-implicit-diff/checkpoints/crossencoder'
DATASET = "msmarco-passage/train/triples-small"
# Create output directory
mkdir -p $OUTPUT_DIR

# Dump triples to file
python -m implicit.data_processing.get_triples \
    --dataset $DATASET \
    --out_file "${OUTPUT_DIR}/triples.tsv.gz"

# Mine negatives
python -m implicit.data_processing.mine_negatives \
    --file "${OUTPUT_DIR}/triples.tsv.gz" \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $CHECKPOINT \
    --cache "${OUTPUT_DIR}/crossencoder-cache"