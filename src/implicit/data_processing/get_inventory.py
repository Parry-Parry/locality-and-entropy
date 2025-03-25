from fire import Fire
import os
from dataclasses import dataclass

NEGATIVES = [16]
batch_size_mapping = {
    16 : 8,
    8 : 8,
    4 : 16,
    2 : 16
}
LOSS_FUNCTIONS = ['LCE', 'MarginMSE', 'RankNet', 'KL_Divergence']
loss_mapping = {
    'LCE' : 'lce',
    'MarginMSE' : 'margin_mse',
    'RankNet' : 'ranknet',
    'KL_Divergence' : 'kl_div'
}
SOURCES = {
        'crossencoder' : [],
        'random' : [],
        'bm25' : [],
     #   'ensemble' : []
    }
source_mapping = {
    'crossencoder' : 'teacher',
    'random' : 'random',
    'bm25' : 'bm25',
 #   'ensemble' : 'ensemble'
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


def inventory(directory : str, output : str = None, ignore_complete : bool = False):
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

    if output is not None:
        # create a csv file with a boolean for if a required model is missing
        out = {
            'source' : [],
            'model_type' : [],
            'loss_function' : [],
            'negative_count' : [],
            'complete' : []
        }
        for source in SOURCES:
            for item in SOURCES[source]:
                out['source'].append(source)
                out['model_type'].append(item.model_type)
                out['loss_function'].append(item.loss_function)
                out['negative_count'].append(item.negative_count)
                out['complete'].append(True)

        for n in NEGATIVES:
            for loss in LOSS_FUNCTIONS:
                for source in SOURCES:
                    for model_type in ['cat', 'dot']:
                        item = Item(model_type=model_type, loss_function=loss, negative_source=source, negative_count=n)
                        out['source'].append(source)
                        out['model_type'].append(item.model_type)
                        out['loss_function'].append(item.loss_function)
                        out['negative_count'].append(item.negative_count)
                        if item not in SOURCES[source]:
                            out['complete'].append(False)
                        else:
                            out['complete'].append(True)
        import pandas as pd
        df = pd.DataFrame(out)
        out_file = 'checklist.csv'
        df.to_csv(os.path.join(output, out_file), index=False)
        print(f"CSV written to {os.path.join(output, out_file)}")
        out_string = ''
        for row in df.itertuples():
            if not row.complete or ignore_complete:
                parsed_source = source_mapping[row.source]
                parsed_batch = batch_size_mapping[row.negative_count]
                parsed_loss = loss_mapping[row.loss_function]
                model_type = 'bi' if row.model_type == 'dot' else 'cross'
                script = f'./scripts/train_{model_type}_{parsed_source}.sh'

                args = [script, parsed_loss, str(row.negative_count)]
                out_string += ' '.join(args) + '\n'
        out_file = 'missing_models.txt'
        with open(os.path.join(output, out_file), 'w') as f:
            f.write(out_string)
        print(f"CMD written to {os.path.join(output, out_file)}")


if __name__ == "__main__":
    Fire(inventory)
