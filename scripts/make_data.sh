OUTPUT_DIR = 'sigir25-implicit-diff/data'
CHECKPOINT = 'checkpoints/crossencoder'
DATASET = "msmarco-passage/train/triples-small"
# Create output directory
mkdir -p $OUTPUT_DIR

python -m implicit.data.get_triples \
    --dataset $DATASET \
    --out_file "${OUTPUT_DIR}/triples.tsv.gz"

python -m implicit.data.mine_negatives \
    --file "${OUTPUT_DIR}/triples.tsv.gz" \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $CHECKPOINT \
    --cache "${OUTPUT_DIR}/crossencoder-cache"