TRIPLES='data/bm25.16.jsonl'
OUTPUT_DIRECTORY='data'
TEACHER_FILE='data/bm25.scores.json.gz'

MODES=(
    "upper_quartile"
    "lower_quartile"
    "above_median"
    "below_median"
    "outlier_quartiles"
    "inner_quartiles"
)

# for mode in modes

for mode in "${MODES[@]}"; do
    # Call the Python script with the appropriate arguments

    python -m implicit.entropy.generate_entropy \
        --teacher_file ${TEACHER_FILE} \
        --triples ${TRIPLES} \
        --output_directory ${OUTPUT_DIRECTORY} \
        --mode ${mode} \
        --use_positive

    python -m implicit.entropy.filter_triples \
        --qids $OUTPUT_DIRECTORY/$mode.positive.txt \
        --triples $TRIPLES \
        --mode $mode \
        --output_directory $OUTPUT_DIRECTORY
        
done

