from fire import Fire 
import os 
from os.path import join
import pyterrier as pt 
if not pt.started(): pt.init()
from ir_measures import *

def main(eval : str, run_dir : str, out_dir : str, rel : int = 1, filter : bool = False, baseline : str = "standard_teach"):
    files = [f for f in os.listdir(run_dir) if os.path.isfile(join(run_dir, f))]
    formatted_eval = eval.replace('-', '_').replace('/', '_')
    if filter: files = [f for f in files if formatted_eval in f]
    pt_dataset = pt.get_dataset(f"irds:{eval}")
    metrics = [AP(rel=rel), NDCG(cutoff=10), R(rel=rel)@100, P(rel=rel, cutoff=10), RR(rel=rel), RR(rel=rel, cutoff=10)]
    runs = []
    names = []
    baseline_run = None
    for file in files:
        if file.endswith(".gz"):
            name = file.strip('.res.gz')
            run = pt.transformer.get_transformer(pt.io.read_results(join(run_dir, file)))
            print([name, f'{formatted_eval}_{baseline}'])
            if name == f'{formatted_eval}_{baseline}':
                baseline_run = run
                continue
            runs.append(run)
            names.append(name)
    assert baseline_run is not None, f"Baseline {baseline} not found!"
    # prepend the baseline run
    print(runs)
    runs = [baseline_run] + runs
    names = [baseline] + names

    print(len(runs))
    
    df_eval = pt.Experiment(
            runs,
            pt_dataset.get_topics('text'),
            pt_dataset.get_qrels(),
            eval_metrics=metrics, 
            names=names,
            baseline=0,
        )
    df_eval.to_csv(out_dir, sep='\t')
            
    return "Success!"

if __name__ == '__main__':
    Fire(main)