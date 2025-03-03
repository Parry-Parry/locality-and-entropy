from pyterrier_pisa import PisaIndex
import pyterrier as pt
from fire import Fire


def main(
        index_path: str,
        dataset: str,
        threads: int = 4,
):
    index = PisaIndex(index_path, threads=threads)
    dataset = pt.get_dataset(f'irds:{dataset}')

    index.index(dataset.get_corpus_iter())


if __name__ == "__main__":
    Fire(main)
