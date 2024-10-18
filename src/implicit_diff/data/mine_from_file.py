import pandas as pd
import random
from tqdm import tqdm
from fire import Fire
import numpy as np
from rankers._util import load_json
import ir_datasets as irds

MSMARCO = "msmarco-passage/train/triples-small"

def get_negatives(triples_file : str, score_file : str, out_file : str, depth : int = 100, group_size : int=2):
    n_neg = group_size - 1
    triples = pd.read_json(triples_file, orient='records', lines=True)
    docs = pd.DataFrame(irds.load(MSMARCO).docs_iter()).doc_id.to_list()
    
    scores = load_json(score_file)

    def sample_n(values):
        delta = n_neg - len(values)
        if delta > 0:
            return values + random.sample(docs, k=delta)
        return random.sample(values, k=n_neg)
    
    # scores is of the form {query_id: {doc_id: score}} turn into ordered query_id : [doc_id] sorted by score
    scores = {str(k): [str(doc_id) for doc_id, _ in sorted(v.items(), key=lambda x: x[1], reverse=True)][:depth] for k, v in scores.items()}
    print(len(scores))
    print(np.mean([len(v) for v in scores.values()]))
    triples['doc_id_b'] = [list(sample_n(scores[str(qid)])) for qid in tqdm(list(triples.query_id.to_list()))]

    triples.to_json(out_file, orient='records', lines=True)

    return "Done"

if __name__ == '__main__':
    Fire(get_negatives)