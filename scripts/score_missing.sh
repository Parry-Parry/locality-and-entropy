OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"

GROUP_SIZE=$1
MODEL=$2
FILE=$OUTPUT_DIR/$MODEL.$GROUP_SIZE.triples.jsonl

python -m implicit.data_processing.get_scores \
    --file $FILE \
    --model_name_or_path $CHECKPOINT \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \
