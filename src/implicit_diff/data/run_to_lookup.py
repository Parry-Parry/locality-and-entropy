import pyterrier as pt 
from fire import Fire
from collections import defaultdict
import json
import gzip


def convert(run : str, out : str):
    run = pt.io.read_results(run)

    run['qid'] = run['qid'].astype(str)
    run['docno'] = run['docno'].astype(str)
    run['score'] = run['score'].astype(float)

    output = defaultdict(dict)
    for i, row in run.iterrows():
        output[row['qid']][row['docno']] = row['score']
    
    with gzip.open(out, 'wt') as f:
        json.dump(output, f)

if __name__ == '__main__':
    Fire(convert)