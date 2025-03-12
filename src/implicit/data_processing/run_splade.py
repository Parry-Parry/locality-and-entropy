import os
import pandas as pd
import ir_datasets as irds
import pyterrier as pt
from fire import Fire


def sparse_retriever(index_path: str, checkpoint: str = 'naver/splade-cocondenser-ensembledistil', batch_size: int = 128, threads: int = 4, **kwargs):
    from pyt_splade import Splade
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(index_path, threads=threads, stemmer='none').quantized()
    splade = Splade(model=checkpoint)

    return splade.query_encoder(verbose=True, batch_size=batch_size) >> index


def main(
        index_path: str,
        output_directory: str,
        ir_dataset: str,
        batch_size: int = 128,
        threads: int = 4,
        depth: int = 1000,
):
    retriever_obj = sparse_retriever(
        index_path=index_path,
        batch_size=batch_size,
        threads=threads
        )
    pipe = retriever_obj

    dataset = irds.load(ir_dataset)
    queries = pd.DataFrame(dataset.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})

    result = pipe.transform(queries)
    if len(result) == 0:
        print("No results to write")
        return
    formatted_dataset = ir_dataset.replace("/", "-")
    output_file = os.path.join(output_directory, f"splade.{formatted_dataset}.{depth}.tsv.gz")
    pt.io.write_results(result, output_file)
    print(f"Results written to {output_file}")
    return 0


if __name__ == '__main__':
    Fire(main)
