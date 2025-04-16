import math
from rankers._util import load_json
import pyterrier as pt
import os
import ir_datasets as irds
from . import cuts as cuts
import json
import random
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
        rankings: str,
        teacher_file: str,
        triples: str = None,
        output_directory: str,
        group_size: int = 16,
        use_positive: bool = False,
        mode: str,
        ir_dataset: str = 'msmarco-passage/train/triples-small'
):
    if use_positive:
        assert triples is not None, "Please provide a triples file for positive sampling."
    if mode not in cuts.__all__:
        raise ValueError(f"Invalid mode. Available modes are: {cuts.__all__}")
    cut_function = getattr(cuts, mode)
    ir_dataset = irds.load(ir_dataset)
    total_docs = ir_dataset.docs_count()
    rankings = pt.io.read_results(rankings)
    rankings = rankings.groupby("qid").docno.apply(list).to_dict()
    teacher_scores = load_json(teacher_file)
    positive = '.positive' if use_positive else ''
    output_file = f'{cut_function}.{group_size}{positive}.jsonl'
    output_file = os.path.join(output_directory, output_file)
    entropy_lookup = {}
    new_rankings = {}
    if use_positive:
        n_sample = group_size - 1
        with open(triples) as f, open():
            for line in tqdm(f):
                line = json.loads(line)
                qid = str(line['query_id'])
                doc_id_a = str(line['doc_id_a'])

                if qid not in teacher_scores:
                    continue
                if doc_id_a not in teacher_scores[qid]:
                    continue

                if qid not in rankings:
                    samples = []
                else:
                    ranking = rankings[qid]
                    if len(ranking) > n_sample:
                        samples = random.sample(ranking, n_sample)

                if len(samples) < n_sample:
                    sample_len = len(samples)
                    missing = n_sample - sample_len
                    indexes = random.sample(range(total_docs), missing)
                    doc_ids = [ir_dataset.docs[i].doc_id for i in indexes]
                    samples.extend(doc_ids)

                ranking = samples.append(doc_id_a)
                entropy = compute_entropy_for_subranking(ranking, teacher_scores, qid)
                entropy_lookup[qid] = entropy
                new_rankings[qid] = ranking
    else:
        for qid, ranking in tqdm(rankings.items()):
            if qid not in teacher_scores:
                continue
            samples = ranking
            if len(samples) > group_size:
                samples = random.sample(samples, group_size)
            if len(samples) < group_size:
                sample_len = len(samples)
                missing = group_size - sample_len
                indexes = random.sample(range(total_docs), missing)
                doc_ids = [ir_dataset.docs[i].doc_id for i in indexes]
                samples.extend(doc_ids)
            entropy = compute_entropy_for_subranking(samples, teacher_scores, qid)
            entropy_lookup[qid] = entropy
            new_rankings[qid] = samples
    
    qids = cut_function(entropy_lookup)

    with open(output_file, 'w') as f:
        for qid in tqdm(qids):
            if qid not in new_rankings:
                continue
            ranking = new_rankings[qid]
            if use_positive:
                ranking.remove(doc_id_a)
                f.write(json.dumps({"query_id": qid, "doc_id_a": doc_id_a, "doc_id_b": ranking}) + "\n")
            else:
                ranking = sorted(ranking, key=lambda x: teacher_scores[qid].get(x, 0), reverse=True)
                f.write(json.dumps({"query_id": qid, "doc_id_a": ranking[0], "doc_id_b": ranking[1:]}) + "\n")

if __name__ == "__main__":
    Fire(main)
