import math
from rankers._util import load_json, save_json
import os
import ir_datasets as irds
from . import cuts as cuts
import json
from fire import Fire
from tqdm import tqdm


def entropy_func(scores):
    """
    Compute the entropy of a list of scores after normalizing to obtain a probability distribution.
    If the sum of scores is zero, returns 0.
    """
    total = sum(scores)
    if total == 0:
        return 0.0
    norm_scores = [s / total for s in scores]
    return -sum(p * math.log(p) for p in norm_scores if p > 0)


def compute_entropy_for_subranking(subranking, score_lookup, query_id):
    """
    Given a subranking (list of doc_ids) for a query,
    look up the score for each doc via score_lookup and compute the entropy.
    """
    scores = [score_lookup[query_id].get(doc_id, 0) for doc_id in subranking]
    return entropy_func(scores)


def main(
        teacher_file: str,
        output_directory: str,
        mode: str,
        triples: str,
        use_positive: bool = False,
        ir_dataset: str = 'msmarco-passage/train/triples-small'
):
    if mode not in cuts.__all__:
        raise ValueError(f"Invalid mode. Available modes are: {cuts.__all__}")
    cut_function = getattr(cuts, mode)
    ir_dataset = irds.load(ir_dataset)
    teacher_scores = load_json(teacher_file)
    positive = '.positive' if use_positive else ''
    output_file = f'{cut_function}{positive}.txt'
    entropy_lookup_file = f'{cut_function}{positive}.json.gz'
    output_file = os.path.join(output_directory, output_file)
    entropy_lookup_file = os.path.join(output_directory, entropy_lookup_file)
    entropy_lookup = {}
    if use_positive:
        with open(triples) as f:
            for line in tqdm(f):
                line = json.loads(line)
                qid = str(line['query_id'])
                doc_id_a = str(line['doc_id_a'])
                doc_id_b = [str(x) for x in line['doc_id_b']]

                if qid not in teacher_scores:
                    continue
                if doc_id_a not in teacher_scores[qid]:
                    continue
                if any(doc_id not in teacher_scores[qid] for doc_id in doc_id_b):
                    continue
                
                ranking = [doc_id_a] + doc_id_b
                entropy = compute_entropy_for_subranking(ranking, teacher_scores, qid)
                entropy_lookup[qid] = entropy
    else:
        with open(triples) as f:
            for line in tqdm(f):
                line = json.loads(line)
                qid = str(line['query_id'])
                doc_id_b = [str(x) for x in line['doc_id_b']]

                if qid not in teacher_scores:
                    continue
                if any(doc_id not in teacher_scores[qid] for doc_id in doc_id_b):
                    continue

                ranking = [doc_id_a] + doc_id_b
                entropy = compute_entropy_for_subranking(ranking, teacher_scores, qid)
                entropy_lookup[qid] = entropy

    save_json(entropy_lookup, entropy_lookup_file)
    cut_ids = cut_function(entropy_lookup)

    with open(output_file, 'w') as f:
        for qid in cut_ids:
            f.write(qid + '\n')


if __name__ == "__main__":
    Fire(main)
