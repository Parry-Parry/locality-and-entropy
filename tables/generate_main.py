import os
import random
import pandas as pd
from typing import Any
from dataclasses import dataclass
from fire import Fire

# Constants
DATASETS = {
    'TREC Deep Learning 2019': 'trec-dl-2019',
    'TREC Deep Learning 2020': 'trec-dl-2020',
}
MODELS = {'Cross-Encoder': 'cat', 'Bi-Encoder': 'dot'}
N_NEG = [2, 4, 8, 16]
LOSS_FUNCTIONS = ['LCE', 'MarginMSE', 'RankNet', 'KL_Divergence']
SOURCES = ['random', 'bm25', 'crossencoder', 'ensemble']
TARGET_METRIC = 'nDCG@10'
ROUNDING = 3


@dataclass
class Item:
    dataset: str
    model_type: str
    n_neg: int
    loss_function: str
    source: str
    content: Any
    sig : bool = False


def parse_name(file_name: str, row, sig):
    name = os.path.basename(file_name)
    dataset = None
    for k, v in DATASETS.items():
        if v in name:
            dataset = k
    if dataset is None:
        raise ValueError(f"Dataset not found in {name}")
        return None
    model_type = 'cat' if 'cat' in name else 'dot'
    loss = None
    for loss_function in LOSS_FUNCTIONS:
        if loss_function in name:
            loss = loss_function
    if loss is None:
        raise ValueError(f"Loss function not found in {name}")
        return None
    n_count = None
    for negative_count in N_NEG[::-1]:
        if f".{negative_count}" in name:
            n_count = negative_count
    negative_source = None
    for source in SOURCES:
        if source in name:
            negative_source = source
    if negative_source is None:
        raise ValueError(f"Negative source not found in {name}")
        return None
    return Item(dataset=dataset, model_type=model_type, loss_function=loss, source=negative_source, n_neg=n_count, content=row, sig=sig)


def generate_table(output: str, run_file: str = None, use_dummy_data=False):
    assert run_file is not None or use_dummy_data, "Either a run file or the use_dummy_data flag must be set"
    # Use a test dataframe if the flag is True
    if use_dummy_data:
        # Create a dummy dataframe for testing
        data = {
            'run': [],
            'nDCG@10': [],
            'sig_nDCG@10': []
        }
        for loss_function in LOSS_FUNCTIONS:
            for n_neg in N_NEG:
                for dataset, ref in DATASETS.items():
                    for source in SOURCES:
                        data['run'].append(f"{ref}-{MODELS['Cross-Encoder']}-{loss_function}-{source}.{n_neg}-distilled")
                        data['nDCG@10'].append(round(random.uniform(0.0, 1.0), 3))
                        data['sig_nDCG@10'].append(random.choice([True, False]))
        runs = pd.DataFrame(data)
    else:
        runs = pd.read_csv(run_file, sep='\t')

    items = []

    # Parse the runs and create items
    for i, row in runs.iterrows():
        run_name = row['run']
        val = getattr(row, TARGET_METRIC)
        val = round(val, ROUNDING)
        sig = getattr(row, f'sig_{TARGET_METRIC}')
        item = parse_name(run_name, val, sig)
        if item is None:
            continue
        items.append(item)

    # Group by loss_function and n_neg
    grouped_items = {}
    for item in items:
        key = item.loss_function
        if key not in grouped_items:
            grouped_items[key] = []
        grouped_items[key].append(item)
        # sort by n_neg
        grouped_items[key] = sorted(grouped_items[key], key=lambda x: x.n_neg)

    # Generate LaTeX table
    latex_code = []

    # Table preamble
    latex_code.append(r"\begin{table*}[tb]")
    latex_code.append(r"\centering")
    latex_code.append(r"\begin{tabular}{ll|" + "c" * (len(DATASETS) * len(SOURCES)) + r"}")
    latex_code.append(r"\toprule")

    # Table Header with \multicolumn for datasets
    latex_code.append(r"&& " +
                      " & ".join([f"\\multicolumn{{{len(SOURCES)}}}{{c}}{{{dataset}}}" for dataset in DATASETS]) + r" \\")
    latex_code.append(r"Loss Function & Group Size & " + " & ".join(SOURCES * len(DATASETS)) + r" \\")
    latex_code.append(r"\midrule")

    # Table Data
    for loss_function, group in grouped_items.items():
        for n_neg in N_NEG:
            row = [f"{loss_function}"]
            row.append(str(n_neg))
            for dataset in DATASETS:
                for source in SOURCES:
                    matching_items = [item for item in group if item.dataset == dataset and item.source == source and item.n_neg == n_neg]
                    val = str(matching_items[0].content) if matching_items else "N/A"
                    sig = matching_items[0].sig if matching_items else False
                    if sig:
                        val = val + r"^*"
                    else:
                        val = val + r'\phantom{^*}'
                    val = f"${val}$"
                    row.append(val)
            latex_code.append(" & ".join(row) + r" \\")
        latex_code.append(r"\midrule")

    latex_code.append(r"\bottomrule")
    latex_code.append(r"\end{tabular}")
    latex_code.append(r"\caption{Performance on various datasets and loss functions}")
    latex_code.append(r"\end{table*}")

    # Join all lines into a single LaTeX string
    latex_str = "\n".join(latex_code)
    with open(output, 'w') as f:
        f.write(latex_str)


if __name__ == "__main__":
    Fire(generate_table)
