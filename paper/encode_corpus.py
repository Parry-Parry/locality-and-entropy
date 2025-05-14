from dataclasses import dataclass, field
import ir_datasets as irds
import pandas as pd
from transformers import HfArgumentParser
import pyterrier as pt
from pyterrier_dr import FlexIndex, HgfBiEncoder


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="bert-base-uncased", metadata={"help": "Path to the model"})


@dataclass
class DataArguments:
    ir_dataset: str = field(metadata={"help": "IR dataset"})
    output_directory: str = field(metadata={"help": "Path to the output directory"})
    batch_size: int = field(default=512, metadata={"help": "Batch size"})


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()
    model = HgfBiEncoder.from_pretrained(
        model_args.model_name_or_path, batch_size=data_args.batch_size
    )

    doc_out = f"{data_args.output_directory}/doc_embeddings"
    doc_index = FlexIndex(doc_out)

    docs_iter = pt.get_dataset(f"irds:{data_args.ir_dataset}").get_corpus_iter()
    doc_pipe = model >> doc_index

    doc_pipe.index(docs_iter, batch_size=data_args.batch_size)


if __name__ == "__main__":
    main()
