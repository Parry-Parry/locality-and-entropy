import pandas as pd
import pyterrier as pt 
if not pt.started(): pt.init()
import ir_datasets as irds
import random
from fire import Fire
from tqdm import tqdm

def mine_from_file(file : str, out_path : str, ir_dataset : str, qids : str = None, depth : int = 100, group_size : int = -1, output_qids : bool = False):
    res = pt.io.read_results(file)
    if qids is not None:
        with open(qids, 'r') as f:
            qids = f.read().splitlines()
        res = res[res['qid'].isin(qids)]
    
    ir_dataset = irds.load(ir_dataset)
    doc_id_a = pd.DataFrame(ir_dataset.docpairs_iter()).set_index('query_id')['doc_id_a'].to_dict()
    query_ids = list(set(res['qid'].tolist()))
    doc_ids = res.groupby('qid').apply(lambda x: x['docno'].tolist()[:depth]).to_list()

    out = {
        'query_id': [],
        'doc_id_a': [],
        'doc_id_b': []
    }

    for qid, doc in tqdm(zip(query_ids, doc_ids), desc='Creating triples'):
        if qid not in doc_id_a or len(doc) < group_size -1:
            continue
        out['query_id'].append(qid)
        out['doc_id_a'].append(doc_id_a[qid])
        out['doc_id_b'].append(random.sample(doc, k=group_size-1) if group_size > 0 else doc)

    final_frame = pd.DataFrame(out)

    frame_out_path = f"{out_path}/triples.{depth}.{group_size}.jsonl.gz"
    final_frame.to_json(frame_out_path, orient='records', lines=True, compression='gzip')

    if group_size==-1:
        group_size = depth

    if output_qids:
        qids_out_path = f"{out_path}/qids.{depth}.{group_size}.txt"
        with open(qids_out_path, 'w') as f:
            f.write('\n'.join(list(set(query_ids))))
    
    return "Done!"

if __name__ == '__main__':
    Fire(mine_from_file)