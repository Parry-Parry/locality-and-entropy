from pyterrier_dr import FlexIndex
import pyterrier as pt
from fire import Fire

def load_bi_encoder(
    checkpoint: str = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    batch_size: int = 64,
    **kwargs,
):
    from transformers import AutoModel, AutoTokenizer
    from pyterrier_dr import HgfBiEncoder, BiScorer

    model = AutoModel.from_pretrained(checkpoint).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    backbone = HgfBiEncoder(model, tokenizer, {}, device=model.device)
    return BiScorer(backbone, batch_size=batch_size)


def main(
        checkpoint: str,
        index_path: str,
        dataset: str,
        batch_size: int = 128,
):
    index = FlexIndex(index_path)
    model = load_bi_encoder(checkpoint, batch_size=batch_size)
    pipe = model >> index

    dataset = pt.get_dataset(f'irds:{dataset}')

    pipe.index(dataset.get_corpus_iter())


if __name__ == "__main__":
    Fire(main)
