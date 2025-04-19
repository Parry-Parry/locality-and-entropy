OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"

MODES=(
    "upper_quartile"
    "lower_quartile"
    "above_median"
    "below_median"
    "outlier_quartiles"
    "inner_quartiles"
)

for MODEL in ${MODELS[@]}; do
    python -m implicit.run_entropy \
        --file $OUTPUT_DIR/$MODEL.jsonl \
        --teacher_file $OUTPUT_DIR/bm25.scores.json.gz \
        --model_name_or_path $CHECKPOINT \
        --dataset $DATASET \
        --out_dir $OUTPUT_DIR \
        --name_override $MODEL
done