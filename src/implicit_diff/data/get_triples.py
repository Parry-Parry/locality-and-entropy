from fire import Fire 
import ir_datasets as irds
import pandas as pd

def get_triples(dataset : str, out_file : str, subset : int = -1):
    irdataset = irds.load(dataset)
    triples = pd.DataFrame(irdataset.docpairs_iter())
    if subset > 0:
        triples = triples.sample(subset)
    triples.to_csv(out_file, index=False)

    return f"Saved {len(triples)} triples of {dataset} to {out_file}"

if __name__ == '__main__':
    Fire(get_triples)