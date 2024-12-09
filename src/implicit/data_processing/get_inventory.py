from fire import Fire
import os
from dataclasses import dataclass

NEGATIVES = [2, 4, 8, 16]
LOSS_FUNCTIONS = ['LCE', 'MarginMSE', 'RankNet', 'KL_Divergence']
SOURCES = {
        'crossencoder' : [],
        'random' : [],
        'bm25' : [],
        'ensemble' : []
    }


@dataclass
class Item:
    model_type : str
    loss_function : str
    negative_source : str
    negative_count : int


def parse_name(name : str):
    # dot-bert-base-uncased-RankNet-crossencoder.16-2-distilled
    model_type = 'cat' if 'cat' in name else 'dot'
    loss = None
    for loss_function in LOSS_FUNCTIONS:
        if loss_function in name:
            loss = loss_function
    if loss is None:
        return None
    n_count = None
    for negative_count in NEGATIVES:
        if str(negative_count) in name:
            n_count = negative_count
    negative_source = None
    for source in SOURCES:
        if source in name:
            negative_source = source
    if negative_source is None:
        return None
    return Item(model_type=model_type, loss_function=loss, negative_source=negative_source, negative_count=n_count)


def inventory(directory : str):
    SOURCES = {
        'crossencoder' : [],
        'random' : [],
        'bm25' : [],
        'ensemble' : []
    }

    for dir in os.listdir(directory):
        item = parse_name(dir)
        if item is None:
            continue
        SOURCES[item.negative_source].append(item)

    # print all relevant information, for example if there are missing models for any combination of sources, negative counts, etc.
    print(['-']*20)
    for source in SOURCES:
        print(f"Source: {source}")
        for item in SOURCES[source]:
            print(f"Model Type: {item.model_type}, Loss Function: {item.loss_function}, Negative Source: {item.negative_source}, Negative Count: {item.negative_count}")
    print(['-']*20)
    for n in NEGATIVES:
        for loss in LOSS_FUNCTIONS:
            for source in SOURCES:
                for model_type in ['cat', 'dot']:
                    item = Item(model_type=model_type, loss_function=loss, negative_source=source, negative_count=n)
                    if item not in SOURCES[source]:
                        print(f"Missing: {item}")
    print(['-']*20)
    # print overall percentage of expected models for each source
    for source in SOURCES:
        total = len(SOURCES[source])
        expected = len(LOSS_FUNCTIONS) * len(NEGATIVES) * 2
        print(f"Source: {source}, Total: {total}, Expected: {expected}, Percentage: {total/expected}")