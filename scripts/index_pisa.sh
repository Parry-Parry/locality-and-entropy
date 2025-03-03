INDEX_PATH=$1
DATASET=$2

python -m implicit.data_processing.index_pisa \
    --index_path $INDEX_PATH \
    --dataset $DATASET 