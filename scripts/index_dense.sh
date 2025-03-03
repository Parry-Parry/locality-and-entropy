CHECKPOINT=$1
INDEX_PATH=$2
DATASET=$3

python -m implicit.data_processing.index_dense \
    --checkpoint $CHECKPOINT \
    --index_path $INDEX_PATH \
    --dataset $DATASET 