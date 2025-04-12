OUTPUT_DIR='data'
CHECKPOINT='checkpoints/crossencoder'
DATASET="msmarco-passage/train/triples-small"

FILE=$1
TEACHER_FILE=$2

python -m implicit.data_processing.get_scores \
    --file $FILE \
    --teacher_file $TEACHER_FILE \
    --model_name_or_path $CHECKPOINT \
    --dataset $DATASET \
