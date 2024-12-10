DIRECTORY='checkpoints'
OUTPUT='data'

python -m implicit.data_processing.get_inventory \
    --directory $DIRECTORY \
    --output $OUTPUT