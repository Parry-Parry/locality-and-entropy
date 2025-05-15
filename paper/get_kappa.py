import json
import argparse
import numpy as np
import gzip
from tqdm import tqdm

TRAIN_JSONL = [
    "data/bm25.16.jsonl",
    "data/crossencoder.16.jsonl",
    "data/random.16.jsonl",
    "data/ensemble.16.jsonl",
]

SCORE_GZ = [
    "data/bm25.scores.json.gz",
    "data/crossencoder.scores.json.gz",
    "data/random.scores.json.gz",
    "data/ensemble.all.scores.json.gz",
]

def compute_kappa_for_file(jsonl_path, scores_path):
    """Compute κ_q for each query in the JSONL file."""
    with gzip.open(scores_path, 'rt', encoding='utf-8') as f:
        scores_lookup = json.load(f)
    kappa_per_query = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {jsonl_path}"):
            item = json.loads(line)
            qid = item['query_id']
            doc_id_a = item['doc_id_a']
            doc_id_b = item['doc_id_b']

            docs = [doc_id_a] + doc_id_b

            scores = np.array([
                scores_lookup[str(qid)].get(doc_id, 0.0) for doc_id in docs
            ])


            scores = np.array(item['scores'], dtype=float)
            # biased sampling distribution ν_q
            nu = scores / scores.sum()
            # uniform target distribution μ_q
            mu = np.full_like(nu, 1.0 / len(nu))
            # κ_q = max_x μ(x)/ν(x)
            kappa = np.max(mu / nu)
            kappa_per_query[qid] = float(kappa)
    return kappa_per_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, default="kappa_output.json",
        help="Output JSON file to save κ_q statistics"
    )
    args = parser.parse_args()

    overall_stats = {}
    for path, score_path in zip(TRAIN_JSONL, SCORE_GZ):
        name = path.split('/')[-1].split('.')[0]
        kappa_map = compute_kappa_for_file(path, score_path)
        values = np.array(list(kappa_map.values()), dtype=float)
        overall_stats[name] = {
            'kappa_max': float(values.max()),
            'kappa_median': float(np.median(values)),
            'kappa_per_query': kappa_map
        }

    with open(args.output, 'w', encoding='utf-8') as out:
        json.dump(overall_stats, out, indent=2)


if __name__ == '__main__':
    main()
