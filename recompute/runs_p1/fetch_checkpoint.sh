#!/bin/zsh
# Fetch one checkpoint's inference files (no optimizer state), or delete it.
#
# Each checkpoint is ~440M for inference (model.safetensors + tokenizer); the
# 1.7G on the cluster is the checkpoint-NNNNN/ optimizer state, which is
# excluded here. Fetch one at a time, use it, then delete, so disk never holds
# more than one model.
#
#   ./fetch_checkpoint.sh get  cat-KL_Divergence-ensemble.16-16
#   ./fetch_checkpoint.sh drop cat-KL_Divergence-ensemble.16-16
#   ./fetch_checkpoint.sh du
set -u

HOST=${IMPLICIT_HOST:-concerned-raccoon-50.os}
SRC=/mnt/primary/implicit/sigir25-implicit-diff/checkpoints
DST="$(cd "$(dirname "$0")" && pwd)/checkpoints"

mkdir -p "$DST"

case "${1:-}" in
  get)
    name=${2:?checkpoint name required}
    if [[ -f "$DST/$name/model.safetensors" ]]; then
      echo "already present: $name"; exit 0
    fi
    echo ">>> fetching $name"
    rsync -a --exclude='checkpoint-*/' "$HOST:$SRC/$name/" "$DST/$name/" \
      || { echo "FAILED: $name"; exit 1; }
    du -sh "$DST/$name"
    ;;
  drop)
    name=${2:?checkpoint name required}
    rm -rf "${DST:?}/${name:?}"
    echo "deleted $name"
    ;;
  du)
    du -sh "$DST" 2>/dev/null || echo "0B"
    ;;
  *)
    echo "usage: $0 {get|drop|du} [checkpoint-name]"; exit 2
    ;;
esac
