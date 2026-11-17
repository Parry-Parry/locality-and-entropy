#!/bin/zsh
# P1 Step 6: determinism check, then regenerate the missing hotpotqa runs.
#
# Disk discipline: one checkpoint on disk at a time (~440M). Each is fetched,
# used for BOTH its determinism check (nfcorpus, which already exists and has a
# published value) and its missing run (hotpotqa), then deleted.
#
# Step 6.2 is a gate: if a regenerated nfcorpus run does not match the
# published nDCG@10 within 0.001, STOP -- nothing generated here is trustworthy.
set -u

cd "$(dirname "$0")/../.." || exit 1
FETCH=recompute/runs_p1/fetch_checkpoint.sh
OUT=recompute/runs_p1/out
CAND=recompute/runs_p1/candidates
mkdir -p "$OUT"

# checkpoint | cat? | published nfcorpus nDCG@10 (determinism target)
MODELS=(
  "cat-LCE-ensemble.16-16|1|0.327"
  "cat-RankNet-ensemble.16-16|1|0.342"
  "cat-KL_Divergence-ensemble.16-16|1|0.341"
)

for entry in "${MODELS[@]}"; do
  name="${entry%%|*}"; rest="${entry#*|}"
  is_cat="${rest%%|*}"; target="${rest#*|}"
  catflag=""; [[ "$is_cat" == "1" ]] && catflag="--cat"

  echo "================================================================"
  echo ">>> $name  (determinism target nfcorpus nDCG@10 = $target)"
  echo "================================================================"

  zsh "$FETCH" get "$name" || exit 1

  # --- determinism check on nfcorpus (run already exists) ---
  python3 -u recompute/runs_p1/rerank.py \
    --candidates "$CAND/run.beir.bm25-flat.nfcorpus.txt.gz" \
    --checkpoint "recompute/runs_p1/checkpoints/$name" \
    --out "$OUT" $catflag --batch-size 64 2>&1 | grep -v "^Cat scoring"

  # --- the missing run: hotpotqa ---
  python3 -u recompute/runs_p1/rerank.py \
    --candidates "$CAND/run.beir.bm25-flat.hotpotqa.txt.gz" \
    --checkpoint "recompute/runs_p1/checkpoints/$name" \
    --out "$OUT" $catflag --batch-size 64 2>&1 | grep -v "^Cat scoring"

  zsh "$FETCH" drop "$name"
  echo ">>> disk after $name: $(zsh "$FETCH" du)"
done

echo
echo "=== verifying determinism ==="
python3 recompute/runs_p1/verify_determinism.py
