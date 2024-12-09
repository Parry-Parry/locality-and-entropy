import os
import random
import pandas as pd
from typing import Any
from dataclasses import dataclass
from fire import Fire

# Constants
DATASETS = {
    'TREC Deep Learning 2019': 'msmarco-passage/trec-dl-2019',
    'TREC Deep Learning 2020': 'msmarco-passage/trec-dl-2020',
}
MODELS = {'Cross-Encoder': 'cat', 'Bi-Encoder': 'dot'}
N_NEG = [2, 4, 8, 16]
LOSS_FUNCTIONS = ['LCE', 'MarginMSE', 'RankNet', 'KL_Divergence']
SOURCES = ['crossencoder', 'random', 'bm25', 'ensemble']
TARGET_METRIC = 'nDCG@10'


@dataclass
class Item:
    dataset: str
    model: str
    n_neg: int
    loss_function: str
    source: str
    content: Any


def parse_name(file_name: str, row):
    name = os.path.basename(file_name)
    dataset = None
    for k, v in DATASETS.items():
        if v in name:
            dataset = k
    if dataset is None:
        return None
    model_type = 'cat' if 'cat' in name else 'dot'
    loss = None
    for loss_function in LOSS_FUNCTIONS:
        if loss_function in name:
            loss = loss_function
    if loss is None:
        return None
    n_count = None
    for negative_count in N_NEG[::-1]:
        if str(negative_count) in name:
            n_count = negative_count
    negative_source = None
    for source in SOURCES:
        if source in name:
            negative_source = source
    if negative_source is None:
        return None
    return Item(dataset=dataset, model_type=model_type, loss_function=loss, source=negative_source, n_neg=n_count, content=getattr(row, TARGET_METRIC))


def generate_table(output: str, run_file: str = None, use_dummy_data=False):
    assert run_file is not None or use_dummy_data, "Either a run file or the use_dummy_data flag must be set"
    # Use a test dataframe if the flag is True
    if use_dummy_data:
        # Create a dummy dataframe for testing
        data = {
            'run': [],
            'nDCG@10': []
        }
        for loss_function in LOSS_FUNCTIONS:
            for n_neg in N_NEG:
                for dataset, ref in DATASETS.items():
                    for source in SOURCES:
                        data['run'].append(f"{ref}-{MODELS['Cross-Encoder']}-{loss_function}-{source}.{n_neg}-distilled")
                        data['nDCG@10'].append(round(random.uniform(0.0, 1.0), 4))
        runs = pd.DataFrame(data)
    else:
        runs = pd.read_csv(run_file, sep='\t')

    items = []

    # Parse the runs and create items
    for run_name, row in runs.groupby('run'):
        item = parse_name(run_name, row)
        if item is None:
            continue
        items.append(item)

    # Group by loss_function and n_neg
    grouped_items = {}
    for item in items:
        key = (item.loss_function, item.n_neg)
        if key not in grouped_items:
            grouped_items[key] = []
        grouped_items[key].append(item)

    # Generate LaTeX table
    latex_code = []

    # Table preamble
    latex_code.append(r"\begin{table}[h!]")
    latex_code.append(r"\centering")
    latex_code.append(r"\begin{tabular}{|c|c|" + "c" * (len(DATASETS) * len(SOURCES)) + r"|}")
    latex_code.append(r"\hline")

    # Table Header with \multicolumn for datasets
    latex_code.append(r"Loss Function & Group Size & " +
                      " & ".join([f"\\multicolumn{{{len(SOURCES)}}}{{|c|}}{{{dataset}}}" for dataset in DATASETS]) + r" \\")
    latex_code.append(r" & & " + " & ".join(SOURCES * len(DATASETS)) + r" \\")
    latex_code.append(r"\hline")

    # Table Data
    for (loss_function, n_neg), group in grouped_items.items():
        row = [f"{loss_function}", f"{n_neg}"]
        for dataset in DATASETS:
            for source in SOURCES:
                matching_items = [item for item in group if item.dataset == dataset and item.source == source]
                row.append(str(matching_items[0].content) if matching_items else "N/A")
        latex_code.append(" & ".join(row) + r" \\")

    latex_code.append(r"\hline")
    latex_code.append(r"\end{tabular}")
    latex_code.append(r"\caption{Performance on various datasets and loss functions}")
    latex_code.append(r"\end{table}")

    # Join all lines into a single LaTeX string
    latex_str = "\n".join(latex_code)
    with open(output, 'w') as f:
        f.write(latex_str)


if __name__ == "__main__":
    Fire(generate_table)
