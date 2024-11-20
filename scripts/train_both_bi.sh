LOSS=$1
GROUP_SIZE=$2
BATCH_SIZE=$3
GRAD_ACCUM=$4
CROSS_TEACHER_FILE="data/crossencoder.scores.json.gz"
BM25_TEACHER_FILE="data/bm25.crossencoder.scores.json.gz"

CMD = f"./scripts/train_bi_bm25 {LOSS} {GROUP_SIZE} {BATCH_SIZE} {GRAD_ACCUM} {BM25_TEACHER_FILE}"
eval $CMD

CMD = f"./scripts/train_bi_teacher {LOSS} {GROUP_SIZE} {BATCH_SIZE} {GRAD_ACCUM} {CROSS_TEACHER_FILE}"
eval $CMD

