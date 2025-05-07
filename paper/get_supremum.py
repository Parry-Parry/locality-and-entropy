import json
import gzip
import itertools
import argparse
import numpy as np

# Training / score file pairs  (extend or modify as needed)
TRAIN_JSONL=(
  "data/bm25.16.jsonl"
  "data/crossencoder.16.jsonl"
  "data/random.16.jsonl"
  "data/ensemble.16.jsonl"
)
SCORE_GZ=(
  "data/bm25.scores.json.gz"
  "data/crossencoder.scores.json.gz"
  "data/random.scores.json.gz"
  "data/ensemble.all.scores.json.gz"
)

LOOKUP = ['bm25', 'crossencoder', 'random', 'ensemble']


def supremum(scores):
    all_combinations = itertools.combinations(scores, 2)
    distances = map(lambda x, y: np.abs(x - y), all_combinations)

    max_distance = max(distances)
    return max_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Output file to save the results",

    )
    args = parser.parse_args()

    # Load the scores from the gz files
    scores = []
    for score_file in SCORE_GZ:
        with gzip.open(score_file, 'rt') as f:
            data = json.load(f)
            scores.append(data)
    # Calculate the supremum
    all_qids = set()

    for score in scores:
        all_qids.update(score.keys())
    
    supremum_results = {}
    for qid in tqdm(all_qids):
        q_scores = [score[qid] for score in scores if qid in score else None]
        if len(q_scores) > 1:
            supremum_results[qid] = [supremum(q_scores) if x is not None else None for x in q_scores]
        else:
            supremum_results[qid] = None
    # Save the results to a JSON file
    with open(args.output, 'w') as f:
        json.dump(supremum_results, f, indent=4)


if __name__ == "__main__":
    main()