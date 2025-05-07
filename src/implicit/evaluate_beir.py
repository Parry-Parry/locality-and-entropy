from fire import Fire
import os
from os.path import join
import pyterrier as pt

if not pt.started():
    pt.init()
from ir_measures import *


def main(
    run_dir: str,
    out_dir: str,
    rel: int = 1,
    per_query: bool = False,
    baseline: str = None,
):
    files = [f for f in os.listdir(run_dir) if os.path.isfile(join(run_dir, f))]
    all_dataset_ids = set([f.split('_')[1] for f in files])

    # iterate over all dataset ids
    for dataset_id in all_dataset_ids:
        subset = [f for f in files if dataset_id in f]
        if len(subset) < 1:
            continue
        formatted_eval = dataset_id.replace("/", "-")
        _beir_with_test = {
            "dbpedia-entity",
            "fever",
            "fiqa",
            "hotpotqa",
            "nfcorpus",
            "quora",
            "scifact"
        }
        if dataset_id in _beir_with_test:
            dataset_id = f"{dataset_id}/test"
        if 'cqadupstack' in dataset_id:
            dataset_id = '/'.join(dataset_id.split('-'))
        if 'webis-touche2020' in dataset_id:
            dataset_id = 'webis-touche2020/v2'
        pt_dataset = pt.get_dataset(f"irds:beir/{dataset_id}")
        metrics = [
            AP(rel=rel),
            NDCG(cutoff=10),
            R(rel=rel) @ 100,
            P(rel=rel, cutoff=10),
            RR(rel=rel),
            RR(rel=rel, cutoff=10),
        ]
        runs = []
        names = []
        baseline_run = None
        for file in subset:
            if file.endswith(".gz"):
                name = file.strip(".res.gz")
                run = pt.transformer.get_transformer(
                    pt.io.read_results(join(run_dir, file))
                )
                print([name, f"{formatted_eval}_{baseline}"])
                if baseline is not None and name == f"{formatted_eval}_{baseline}":
                    baseline_run = run
                    continue
                runs.append(run)
                names.append(name)
        if baseline is not None:
            runs = [baseline_run] + runs
            names = [baseline] + names

        df_eval = pt.Experiment(
            runs,
            pt_dataset.get_topics(),
            pt_dataset.get_qrels(),
            perquery=per_query,
            eval_metrics=metrics,
            names=names,
            baseline=0 if baseline is not None else None,
        )
        out_file = f"beir_{formatted_eval}.tsv.gz"
        out_file = os.path.join(out_dir, out_file)
        df_eval.to_csv(out_file, sep="\t")

    return "Success!"


if __name__ == "__main__":
    Fire(main)
