from fire import Fire
import os
from os.path import join
import pyterrier as pt
import pandas as pd
import statsmodels.stats.weightstats as ws
from ir_measures import AP, NDCG

if not pt.started():
    pt.init()


import numpy as np

def tost(x, y, bound_pct=1.0):
    """
    Perform two one-sided t-tests (TOST) for equivalence, with bounds set
    to ±bound_pct% of the observed difference in sample means.

    Parameters
    ----------
    x : array-like
        Sample data for group 1.
    y : array-like
        Sample data for group 2.
    bound_pct : float, optional
        Equivalence margin as a percentage of the observed mean difference
        (default is 1.0, i.e. ±1 %).

    Returns
    -------
    p_val : float
        Combined p-value for the equivalence test.
    p_low : float
        P-value for the lower‐bound test (H0: μ1−μ2 ≤ −bound).
    p_high : float
        P-value for the upper‐bound test (H0: μ1−μ2 ≥ +bound).
    bound : float
        Absolute equivalence margin used (±bound).
    diff_mean : float
        Observed difference in sample means (mean(x) − mean(y)).
    """
    # observed difference in means
    diff_mean = np.mean(x) - np.mean(y)
    # equivalence margin = bound_pct% of |diff_mean|
    bound = abs(diff_mean) * (bound_pct / 100.0)
    low_eq, high_eq = -bound, bound

    p_val, (t_low, p_low, df_low), (t_high, p_high, df_high) = \
        ws.ttost_ind(x, y, low_eq, high_eq)

    return p_val, p_low, p_high

def parse_run_meta(fname):
    base = fname.replace(".res.gz","")
    if '16' not in base:
        return {'loss':'LCE','arch':'CE-Teacher','domain':'BM25'}
    meta = {'loss':'UNK','arch':'UNK','domain':'UNK'}
    for loss in ("LCE","marginMSE","RankNet","KL"):
        if loss.lower() in base.lower():
            meta["loss"] = loss
            break
    meta["arch"] = "CE" if "cat" in base.lower() else "BE"
    dom = None
    for tag, name in [
        ("random","Random"),
        ("ensemble","Ensemble"),
        ("crossencoder","Cross-Encoder"),
        ("bm25","BM25")
    ]:
        if tag in base.lower():
            dom = name
            break
    meta["domain"] = dom or "UNK"
    return meta


def main(run_dir: str, out_dir: str, rel: int = 1, baseline: str = None):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(run_dir) if f.endswith(".res.gz")]
    beir_files = [f for f in files if "beir" in f]
    dataset_ids = sorted({f.split("_")[1] for f in beir_files})
    groups = {
        "dl19": ["dl_2019"],
        "dl20": ["dl_2020"],
        "beir": [ds for ds in dataset_ids if ds not in {"dl_2019","dl_2020"}],
    }

    for group_name, ds_list in groups.items():
        all_perdfs = []
        for ds in ds_list:
            # standardize the PT dataset id
            ds_key = ds
            if ds in {"dbpedia-entity","fever","fiqa","hotpotqa","nfcorpus","quora","scifact"}:
                ds_key = f"{ds}/test"
            if 'cqadupstack' in ds_key:
                ds_key = '/'.join(ds_key.split('-'))
            if 'webis-touche2020' in ds_key:
                ds_key = 'webis-touche2020/v2'
            if 'dl_' in ds_key:
                if '19' in ds_key:
                    ds_key = 'msmarco-passage/trec-dl-2019/judged'
                elif '20' in ds_key:
                    ds_key = 'msmarco-passage/trec-dl-2020/judged'
            # load the dataset
                pt_ds = pt.get_dataset(f"irds:{ds_key}")
                rel = 2
            else:
                try:
                    pt_ds = pt.get_dataset(f"irds:beir/{ds_key}")
                except Exception as e:
                    print(f"Error loading dataset {ds_key}: {e}")
                    continue
            topics, qrels = pt_ds.get_topics("text"), pt_ds.get_qrels()
            metrics = [AP(rel=rel), NDCG(cutoff=10)]
            metric_names = [str(m) for m in metrics]

            if "dl-" in ds_key:
                if '19' in ds_key:
                    formatted_ds_key = "dl_2019"
                elif '20' in ds_key:
                    formatted_ds_key = "dl_2020"
                
                subset = [f for f in files if formatted_ds_key in f]
            else:
                subset = [f for f in files if f.split("_")[1] == ds]
            runs, names = [], []
            baseline_run = None
            for fn in subset:
                runname = fn.replace(".res.gz","")
                if any(x in runname for x in ("median","quartile")):
                    continue
                run = pt.transformer.get_transformer(pt.io.read_results(join(run_dir, fn)))
                if baseline and runname.endswith(f"_{baseline}"):
                    baseline_run = run
                else:
                    runs.append(run)
                    names.append(runname)
            if baseline and baseline_run:
                runs.insert(0, baseline_run)
                names.insert(0, baseline)

            # per-query eval (wide form)
            df_per = pt.Experiment(
                runs, topics, qrels,
                perquery=True,
                eval_metrics=metrics,
                names=names,
                baseline=0 if baseline else None
            )

            if len(df_per) == 0:
                # print several diagnostics
                print(f"Warning: no results for {ds}")
                print(f"  run_dir: {run_dir}")
                print(f"  out_dir: {out_dir}")
                print(f"  ds: {ds}")
                print(f"  files: {subset}")
                print(f"  baseline: {baseline}")
                print(topics.head())
                print(qrels.head())

            # attach run metadata
            df_per['domain'] = df_per.name.map(lambda x: parse_run_meta(x)["domain"])
            df_per['loss']   = df_per.name.map(lambda x: parse_run_meta(x)["loss"])
            df_per['arch']   = df_per.name.map(lambda x: parse_run_meta(x)["arch"])
            df_per['measure'] = df_per.measure.map(str)

            all_perdfs.append(df_per)

        df_all = pd.concat(all_perdfs, ignore_index=True)
        df_all.to_csv(join(out_dir, f"perquery_{group_name}.tsv.gz"), 
                      sep="\t", index=False)

        # ——— mean of each measure by loss/domain/arch ———
        df_means = (
            df_all
            .groupby(['loss','domain','arch','measure'])['value']
            .mean()
            .reset_index()
        )
        print(f"Means for {group_name}:")
        print(len(df_means))
        df_means.to_csv(join(out_dir, f"means_{group_name}.tsv"), 
                        sep="\t", index=False)

        # ——— TOST across domains for each loss/arch/measure ———
        records = []
        for (loss, arch), sub_la in df_all.groupby(['loss','arch']):
            if arch == "CE-Teacher":
                records.append({
                    "group":   group_name,
                    "loss":    loss,
                    "arch":    arch,
                    "domain1": "BM25",
                    "domain2": "BM25",
                    "measure": "N/A",
                    "p_lower": 1.0,
                    "p_upper": 1.0
                })
                continue
            doms = sub_la['domain'].unique()
            measures = sub_la['measure'].unique()

            for m in measures:
                sub = sub_la[sub_la['measure'] == m]
                for i, d1 in enumerate(doms):
                    for d2 in doms[i+1:]:
                        # get qids present in both domains for this measure
                        ids1 = set(sub.loc[sub['domain'] == d1, 'qid'])
                        ids2 = set(sub.loc[sub['domain'] == d2, 'qid'])
                        common_qids = sorted(ids1 & ids2)
                        if not common_qids:
                            continue

                        # build lookup dicts for fast value access
                        val_map1 = dict(zip(
                            sub.loc[sub['domain'] == d1, 'qid'],
                            sub.loc[sub['domain'] == d1, 'value']
                        ))
                        val_map2 = dict(zip(
                            sub.loc[sub['domain'] == d2, 'qid'],
                            sub.loc[sub['domain'] == d2, 'value']
                        ))

                        # extract aligned arrays
                        x = np.array([val_map1[q] for q in common_qids])
                        y = np.array([val_map2[q] for q in common_qids])

                        p, p_lo, p_hi = tost(x, y)
                        records.append({
                            "group":   group_name,
                            "loss":    loss,
                            "arch":    arch,
                            "domain1": d1,
                            "domain2": d2,
                            "measure": m,
                            "p_value": p,
                            "p_lower": p_lo,
                            "p_upper": p_hi
                        })
        df_tost = pd.DataFrame.from_records(records)
        df_tost.to_csv(join(out_dir, f"tost_{group_name}.tsv"), 
                       sep="\t", index=False)

    return "Success!"


if __name__ == "__main__":
    Fire(main)