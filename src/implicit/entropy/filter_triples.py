import os
import json
from fire import Fire
from tqdm import tqdm


def main(triples: str,
         qids: str,
         mode: str,
         output_directory: str):
    output_file = os.path.join(output_directory, f'{mode}.jsonl')

    with open(qids) as f:
        qids = set(line.strip() for line in f)

    with open(triples) as f, open(output_file, 'w') as out_f:
        for line in tqdm(f):
            record = json.loads(line)
            qid = str(record['query_id'])
            if qid not in qids:
                continue
            out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    Fire(main)
