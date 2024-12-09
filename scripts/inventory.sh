DIRECTORY='checkpoints'
OUTPUT='data/inventory.csv'

python -m implicit.data_processing.get_inventory \
    --directory $DIRECTORY \
    --output $OUTPUT