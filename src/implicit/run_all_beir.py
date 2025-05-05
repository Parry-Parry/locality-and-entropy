import pyterrier as pt

if not pt.started():
    pt.init()
import ir_datasets as irds
import pandas as pd
from fire import Fire
import os
import logging
import glob


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
    return BiScorer(backbone, batch_size=batch_size, verbose=True)


def load_cross_encoder(
    checkpoint: str = "cross-encoder/ms-marco-TinyBERT-L-6",
    batch_size: int = 64,
    **kwargs,
):
    from rankers import CatTransformer

    return CatTransformer.from_pretrained(checkpoint, batch_size=batch_size, verbose=True)


def load_run(system_name, dataset_id):
    filename = f"data/beir.run.{system_name}.{dataset_id}.txt"
    return pt.io.read_results(filename)


def get_latest_checkpoint(model_name_or_path):
    checkpoints = [
        d for d in os.listdir(model_name_or_path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_name_or_path, d))
    ]

    if not checkpoints:
        return f"Model not found at specified path {model_name_or_path}!"

    # Sort by checkpoint number in descending order (newest first)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    # Find the latest valid checkpoint
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(model_name_or_path, checkpoint)
        if os.path.exists(os.path.join(checkpoint_path, "config.json")) and os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
            return checkpoint_path

    return f"Model not found at specified path {model_name_or_path}!"


def run_topics(
    model_name_or_path: str,
    out_path: str,
    batch_size: int = 256,
    cat: bool = False,
    dont_overwrite: bool = False,
):
    if not os.path.exists(f"{model_name_or_path}/config.json"):
        # find most recent checkpoint
        model_name_or_path = get_latest_checkpoint(model_name_or_path)
        if 'not found' in model_name_or_path:
            return f"Model not found at specified path {model_name_or_path}!"
        output_directory = os.path.dirname(out_path)
        basename = os.path.basename(out_path)
        # get checkpoint number
        checkpoint_number = model_name_or_path.split("-")[-1]
        basename = f"tmp-{checkpoint_number}-{basename}"
        out_path = os.path.join(output_directory, basename)

    files = glob.glob("data/run.beir*.txt")
    
    for file in files:
        print(f"Processing {file}")
        # format is data/beir.run.{system_name}.{dataset_id}.txt
        system_name = file.split(".")[2]
        dataset_id = file.split(".")[3]
        formatted_dataset = dataset_id.replace("/", "-")
        model_name = os.path.basename(model_name_or_path)
        out_file = os.path.join(out_path, f"beir_{formatted_dataset}_{model_name}_{system_name}.res.gz")
        print(f"Output file: {out_file}")
        if 'cqadupstack' in dataset_id:
            dataset_id = '/'.join(dataset_id.split('-'))

        try:
            dataset = irds.load(f"beir/{dataset_id}")
        except Exception as e:
            logging.info(f"Error loading dataset {dataset_id}: {e}")
            continue

        if os.path.exists(out_file) and not dont_overwrite:
            logging.info(f"File already exists at {out_file}")
            continue

        run = pt.io.read_results(file)
        try:
            queries = (
                pd.DataFrame(dataset.queries_iter()).set_index("query_id").text.to_dict()
            )
        except Exception as e:
            logging.info(f"Error loading queries for dataset {dataset_id}: {e}")
            continue
        run["query"] = run["qid"].map(lambda qid: queries[qid])
        docstore = dataset.docs_store()
        run['text'] = run['docno'].map(lambda docno: docstore.get(docno).text)

        try:
            model = (
                load_bi_encoder(model_name_or_path, batch_size=batch_size)
                if not cat
                else load_cross_encoder(model_name_or_path, batch_size=batch_size)
            )
        except Exception as e:
            logging.info(f"Error loading model {model_name_or_path}: {e}")
            return f"Error loading model: {e}"

        res = model.transform(run)

        pt.io.write_results(res, out_file)

        logging.info(f"Results written to {out_file}")
    return f"Results written to {output_directory}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(run_topics)
