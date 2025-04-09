OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"

FILE=$OUTPUT_DIR/$MODEL.16.jsonl

MODELS=("bm25" "crossencoder" "random")

for MODEL in ${MODELS[@]}; do
    python -m implicit.run_entropy \
        --file $OUTPUT_DIR/$MODEL.16.jsonl \
        --model_name_or_path $CHECKPOINT \
        --dataset $DATASET \
        --out_dir $OUTPUT_DIR \
        --name_override $MODEL
done

python -m implicit.run_entropy \
    --file $OUTPUT_DIR/ensemble.16.jsonl \
    --model_name_or_path $CHECKPOINT \
    --dataset $DATASET \
    --out_dir $OUTPUT_DIR \
    --name_override ensemble.all