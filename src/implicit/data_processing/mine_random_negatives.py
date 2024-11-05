from collections import defaultdict
from fire import Fire
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
if not pt.started(): pt.init()
import logging
import os
import random
from pyterrier_caching import ScorerCache
from implicit.util import save_json, load_json

def bm25(index_dir : str, k1 : float = 1.2, b : float = 0.75, threads : int = 1, **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyterrier_pisa import PisaIndex

    if os.path.exists(index_dir):
        logging.info(f"Loading index from {index_dir}")
        return PisaIndex(index_dir, threads=threads, **kwargs).bm25(k1=k1, b=b, verbose=True)
    else:
        logging.info("assuming pre-built index")
        return PisaIndex.from_dataset(index_dir, threads=threads, **kwargs).bm25(k1=k1, b=b, verbose=True)

def load_crossencoder(model_name_or_path : str, batch_size : int = 512, verbose : bool = True, cache : str = None):
    from rankers import CatTransformer
    model = CatTransformer.from_pretrained(model_name_or_path, batch_size=batch_size, verbose=verbose)
    if cache is not None:
        cached_scorer = ScorerCache(cache, model)
        if not cached_scorer.built():
            dataset = pt.get_dataset('irds:msmarco-passage')
            cached_scorer.build(dataset.get_corpus_iter())
        return cached_scorer
    return model

def mine(file,
         dataset : str,
         out_dir : str, 
         index_path : str = None, 
         model_name_or_path : str = None,
         subset_depth : int = 100,
         depth : int = 200,
         batch_size : int = 512,
         n_neg : int = None,
         n_negs : list = [31, 15, 7, 1],
         cache : str = None,
         ):
    logging.info(f"Index: {index_path}")
    logging.info(f"Output Directory: {out_dir}")
    logging.info("Loading model...")
    if index_path is None: index_path = 'msmarco_passage'
    name = model_name_or_path.replace("/", "-") if model_name_or_path else 'bm25'

    dataset = irds.load(dataset)
    logging.info("Loading dataset...")
    docs = pd.DataFrame(dataset.docs_iter())
    docs_lookup = docs.set_index('doc_id')['text'].to_dict()
    docs = docs.doc_id.to_list()  
    query_lookup = pd.DataFrame(dataset.queries_iter()).set_index('query_id')['text'].to_dict()
    triples = pd.read_json(file, orient='records', lines=True)
    doc_id_a_lookup = triples.set_index("query_id").doc_id_a.to_dict()
    doc_id_a_lookup = {str(k): v for k, v in doc_id_a_lookup.items()}

    if n_neg is not None: n_negs = [n_neg]
    output_lookup = out_dir + f'/{name}.scores.json.gz'
    if os.path.exists(output_lookup):
        lookup = defaultdict(dict)
        logging.info(f"Loading scores from {output_lookup}")
        _lookup = load_json(output_lookup)
        for qid in _lookup.keys():
            lookup[qid] = _lookup[qid]
    else:
        lookup = defaultdict(dict)
    for n_neg in n_negs:
        negs = {qid : random.sample(docs, k=depth) for qid in list(triples.query_id.unique())}
        group_size = n_neg + 1
        
        def pivot_negs(negs):
            frame = {
                'qid': [],
                'docno': [],
                'query': [],
                'text': [],
            }
            for _qid in tqdm(negs.keys(), desc="Pivoting negatives"):
                if len(negs[_qid]) < 1: continue
                qid = str(_qid)
                try:
                    query_text = query_lookup[qid]
                except:
                    logging.error(f"Query {qid} not found in queries")
                    continue
                doc_id_a = str(doc_id_a_lookup[qid])
                frame['qid'].append(qid)
                frame['docno'].append(doc_id_a)
                frame['text'].append(docs_lookup[doc_id_a])
                frame['query'].append(query_text)
                for _doc_id in negs[_qid]:
                    doc_id = str(_doc_id)
                    frame['qid'].append(qid)
                    frame['docno'].append(doc_id)
                    frame['text'].append(docs_lookup[doc_id])
                    frame['query'].append(query_text)
            frame['score'] = [0.0] * len(frame['qid'])
            return pd.DataFrame(frame).drop_duplicates(subset=['qid', 'docno'])

        if model_name_or_path:
            logging.info("Loading crossencoder...")
            crossencoder = load_crossencoder(model_name_or_path, batch_size=batch_size, cache=cache) % subset_depth
            frame = pivot_negs(negs)
            logging.info(f"Getting teacher scores for {len(frame)} pairs...")
            
            # get length of lookup and cut frame 
            res = crossencoder.transform(frame)
            
            for row in tqdm(res.itertuples()):
                lookup[row.qid][row.docno] = row.score
            
            res = res.groupby('qid')['docno'].apply(list).reset_index().set_index('qid')['docno'].to_dict()
            negs = {str(qid) : random.sample(res[qid], k=n_neg) for qid in res.keys()}
            triples['doc_id_b'] = triples['query_id'].map(lambda x: negs[str(x)])
            triples.to_json(out_dir + f'/{name}.{group_size}.random.triples.jsonl.gz', orient='records', lines=True)
        
    save_json(lookup, out_dir + f'/{name}.scores.json.gz')

    return f"Successfully saved to {out_dir}"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(mine)