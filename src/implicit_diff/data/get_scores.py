from rankers import CatTransformer, DotTransformer
from rankers._util import save_json
from collections import defaultdict
from fire import Fire
import logging
import pandas as pd
import ir_datasets as irds
from tqdm import tqdm

def get_scores(model_name_or_path : str, 
               triples_dir : str, 
               ir_dataset : str, 
               output_file : str, 
               batch_size : int = 512, 
               architecture : str = 'cat',
               ):
    if architecture == 'cat':
        model = CatTransformer.from_pretrained(model_name_or_path, batch_size=batch_size, verbose=True)
    elif architecture == 'dot':
        model = DotTransformer.from_pretrained(model_name_or_path, batch_size=batch_size, verbose=True)
    else:
        raise ValueError("Architecture must be either 'cat' or 'dot'")
    
    dataset = irds.load(ir_dataset)
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()

    def pivot_triples(triples):
        frame = {
            'qid': [],
            'docno': [],
            'query': [],
            'text': [],
        }
        for row in tqdm(triples.itertuples(), desc="Pivoting triples"):
            qid = str(row.query_id)
            doc_id_a = str(row.doc_id_a)
            doc_id_b = [str(x) for x in row.doc_id_b] if type(row.doc_id_b) == list else [str(row.doc_id_b)]
            query_text = queries[qid]
            frame['qid'].append(qid)
            frame['docno'].append(doc_id_a)
            frame['text'].append(docs[doc_id_a])
            frame['query'].append(query_text)

            for id in doc_id_b:
                frame['qid'].append(qid)
                frame['docno'].append(id)
                frame['text'].append(docs[id])
                frame['query'].append(query_text)
                
        frame['score'] = [0.0] * len(frame['qid'])
        return pd.DataFrame(frame)

    triples = pd.read_json(triples_dir, lines=True, orient='records')
    frame = pivot_triples(triples)

    scores = model.transform(frame)

    lookup = defaultdict(dict)

    for row in scores.itertuples():
        lookup[row.qid][row.docno] = row.score

    save_json(lookup, output_file)

    return f"Successfully saved scores to {output_file}"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(get_scores)
    
    