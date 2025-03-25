DIRECTORY='checkpoints'
OUTPUT='data'
IGNORE=$1

CMD="python -m implicit.data_processing.get_inventory \
    --directory $DIRECTORY \
    --output $OUTPUT"

if [ ! -z "$IGNORE" ]; then
    CMD="$CMD --ignore_complete"
fi

eval $CMD