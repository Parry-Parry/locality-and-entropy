import pyterrier as pt 
if not pt.started(): pt.init()
import ir_datasets as irds 
import pandas as pd 
from fire import Fire 
import os
import logging

def load_bi_encoder(checkpoint : str ='sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco', batch_size : int = 64, **kwargs):
    from transformers import AutoModel, AutoTokenizer
    from pyterrier_dr import HgfBiEncoder, BiScorer

    model = AutoModel.from_pretrained(checkpoint).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    backbone = HgfBiEncoder(model, tokenizer, {}, device=model.device)
    return BiScorer(backbone, batch_size=batch_size)

def load_cross_encoder(checkpoint : str ='cross-encoder/ms-marco-TinyBERT-L-6', batch_size : int = 64, **kwargs):
    from rankers import CatTransformer
    return CatTransformer.from_pretrained(checkpoint, batch_size=batch_size)

def get_bm25(dataset_id):
    from tira.rest_api_client import Client
    task = dataset_id.split("/")[0]
    if task != 'reneuir-2024':
        task = 'ir-benchmarks'

    tira = Client()
    return tira.pd.from_retriever_submission(approach=f'{task}/tira-ir-starter/BM25 (tira-ir-starter-pyterrier)', dataset=dataset_id)


def load_run(system_name, dataset_id, trec_file = None):
    from tira.rest_api_client import Client
    if trec_file:
        return pt.io.read_results(trec_file)
    trec_file = f'data/{dataset_id.replace("/", "-")}-{system_name}.run.gz'

    if dataset_id == 'msmarco-passage/trec-dl-2020/judged':
        dataset_shortcut = 'dl20'
    elif dataset_id == 'msmarco-passage/trec-dl-2019/judged':
        dataset_shortcut = 'dl19'

    if system_name == 'truth':
        trec_file = f'data/trec_{dataset_shortcut}.rez.gz'

    if os.path.exists(trec_file):
        return pt.io.read_results(trec_file)

    print('run not available locally, load from tira...')
    task = dataset_id.split("/")[0]
    if task != 'reneuir-2024':
        task = 'ir-benchmarks'

    tira = Client('https://api.tira.io')
    if system_name == 'bm25':
        return tira.pd.from_retriever_submission(approach=f'{task}/tira-ir-starter/BM25 (tira-ir-starter-pyterrier)', dataset=dataset_id)
    
    if system_name == 'rank-zephyr':
        return tira.pd.from_retriever_submission(approach=f'{task}/fschlatt/rank-zephyr', dataset=dataset_id)
    
    if system_name == 'mono-t5':
        return tira.pd.from_retriever_submission(approach=f'{task}/tira-ir-starter/MonoT5 3b (tira-ir-starter-gygaggle)', dataset=dataset_id)

def run_topics(ir_dataset : str, 
               model_name_or_path : str, 
               out_path : str,
               topics_or_res : str = None,
               index : str = None,
               batch_size : int = 256, 
               text_field : str = 'text', 
               cat : bool = False,
               overwrite : bool = False):
    if not overwrite and os.path.exists(out_path): return "File already exists!"
    try:
        topics_or_res = pt.io.read_results(topics_or_res) if topics_or_res else load_run('bm25', ir_dataset)
    except AttributeError:
        return f"Tira is down, couldnt load {ir_dataset}!"
    ir_dataset = irds.load(ir_dataset)
    queries = pd.DataFrame(ir_dataset.queries_iter()).set_index('query_id').text.to_dict()
    topics_or_res['query'] = topics_or_res['qid'].map(lambda qid: queries[qid])
    if not os.path.exists(f"{model_name_or_path}/config.json"):
        return f"Model not found at specified path {model_name_or_path}!"
    model = load_bi_encoder(model_name_or_path, batch_size=batch_size) if not cat else load_cross_encoder(model_name_or_path, batch_size=batch_size)

    if not index: 
        docs = pd.DataFrame(ir_dataset.docs_iter()).set_index('doc_id').text.to_dict()
        topics_or_res['text'] = topics_or_res['docno'].map(lambda docno: docs[docno])
        model = model
    else:
        index = pt.IndexFactory.of(index, memory=True)
        model = pt.text.get_text(index, text_field) >> model
    print(topics_or_res.head())
    res = model.transform(topics_or_res)
    pt.io.write_results(res, out_path)

    return "Done!"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(run_topics)