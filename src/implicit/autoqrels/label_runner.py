from autoqrels.oneshot import DuoPrompt
import pandas as pd
from typing import Dict, List
import ir_datasets as irds
from tqdm import tqdm
from collections import defaultdict


class LabelRunner:
    def __init__(self,
                 dataset,
                 backbone='google/flan-t5-xl',
                 batch_size=8,
                 verbose=False,
                 query_id_field='query_id',
                 doc_id_field='doc_id',
                 query_field='text',
                 doc_field='text',
                 max_src_len=330,
                 cache_path=None
                 ):
        if type(dataset) is str:
            dataset = irds.load(dataset)
        self.dataset = dataset
        self.backbone = backbone
        self.verbose = verbose
        self.query_id_field = query_id_field
        self.doc_id_field = doc_id_field
        self.labeller = DuoPrompt(dataset,
                                  backbone,
                                  'cuda',
                                  batch_size,
                                  False,
                                  query_field,
                                  doc_field,
                                  max_src_len,
                                  cache_path)

    def transform(self, triples: List[dict]) -> Dict[str, pd.DataFrame]:
        output = defaultdict(dict)

        for element in tqdm(triples):
            qid = element[self.query_id_field]
            doc_id_a = element['doc_id_a']
            doc_id_b = element['doc_id_b']

            labels = self.labeller.infer_oneshot(str(qid), doc_id_a, doc_id_b)

            output[qid][doc_id_a] = 1
            for doc_id, label in zip(doc_id_b, labels):
                output[qid][doc_id] = label

        return output
