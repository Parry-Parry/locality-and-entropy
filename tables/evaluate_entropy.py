#!/usr/bin/env python3
from fire import Fire
import os
import re
from os.path import join
import numpy as np
import pyterrier as pt
import pandas as pd
import statsmodels.stats.weightstats as ws
from ir_measures import AP, NDCG

if not pt.started():
    pt.init()


def tost(x, y, bound_pct=0.5):
    """
    Two‐one‐sided t‐test with margin = ±bound_pct% of observed mean difference.
    Returns (p_overall, p_low, p_high).
    """
    diff = np.mean(x) - np.mean(y)
    margin = abs(diff) * (bound_pct / 100.0)
    low_eq, high_eq = -margin, margin
    pval, (t_low, p_low, _), (t_high, p_high, _) = ws.ttost_ind(x, y, low_eq, high_eq)
    return pval, p_low, p_high


def parse_run_meta(fname):
    """
    Parses loss, arch, domain from run filename.
    Treats 'cat-' runs as CE / Cross-Encoder.
    """
    base = fname.replace(".res.gz", "")
    # default BM25 baseline
    if "16" not in base:
        return {"loss": "LCE", "arch": "CE-Teacher", "domain": "BM25"}
    meta = {"loss": "UNK", "arch": "UNK", "domain": "UNK"}
    # loss
    for loss in ("LCE", "marginMSE", "RankNet", "KL"):
        if loss.lower() in base.lower():
            meta["loss"] = loss
            break
    # arch
    if "cat-" in base.lower():
        meta["arch"] = "CE"
    else:
        meta["arch"] = "BE"
    # domain

    for tag, name in [
        ("lower_quartile", "lower_quartile"),
        ("upper_quartile", "upper_quartile"),
        ("below_median", "below_median"),
        ("above_median", "above_median"),
        ("inner_quartiles", "inner_quartiles"),
    ]:
        if tag in base.lower():
            meta["domain"] = name
            break
    return meta


def main(run_dir: str, out_dir: str, rel: int = 1):
    """
    Evaluate only CAT runs with quartile/median splits:
      lower_quartile, upper_quartile, below_median,
      above_median, inner_quartiles.
   """
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(run_dir) if f.endswith(".res.gz")]

    # allowed subsets
    ALLOWED = {
        "lower_quartile",
        "upper_quartile",
        "below_median",
        "above_median",
        "inner_quartiles"
    }
    files = [f for f in files if any(s in f for s in ALLOWED)]

    # discover datasets from BEIR files
    beir_files = [f for f in files if "beir" in f]
    dataset_ids = sorted({f.split("_")[1] for f in beir_files})
    groups = {
        "dl19": ["dl_2019"],
        "dl20": ["dl_2020"],
        "beir": [ds for ds in dataset_ids if ds not in {"dl_2019", "dl_2020"}],
    }

    metrics = [AP(rel=rel), NDCG(cutoff=10)]
    metric_names = [str(m) for m in metrics]

    for group_name, ds_list in groups.items():
        all_per = []
        # for each dataset in group
        for ds in ds_list:
            # standardize PT id
            ds_key = ds
            if ds in {
                "dbpedia-entity", "fever", "fiqa", "hotpotqa",
                "nfcorpus", "quora", "scifact"
            }:
                ds_key = f"{ds}/test"
            if "cqadupstack" in ds_key:
                ds_key = "/".join(ds_key.split("-"))
            if "webis-touche2020" in ds_key:
                ds_key = "webis-touche2020/v2"
            if "dl_" in ds_key:
                if "19" in ds_key:
                    pt_ds = pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged")
                    rel = 2
                else:
                    pt_ds = pt.get_dataset("irds:msmarco-passage/trec-dl-2020/judged")
                    rel = 2
            else:
                try:
                    pt_ds = pt.get_dataset(f"irds:beir/{ds_key}")
                except Exception as e:
                    print(f"Error loading dataset {ds_key}: {e}")
                    continue
            topics, qrels = pt_ds.get_topics("text"), pt_ds.get_qrels()
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
            for fn in sorted(subset):
                runname = fn.replace(".res.gz", "")
                run = pt.transformer.get_transformer(pt.io.read_results(join(run_dir, fn)))
                runs.append(run)
                names.append(runname)

            if not runs:
                continue

            # evaluate per‐query
            df_per = pt.Experiment(
                runs, topics, qrels,
                perquery=True,
                eval_metrics=metrics,
                names=names
            )
            # attach metadata
            df_per["dataset_id"] = ds
            df_per["loss"]       = df_per.name.map(lambda x: parse_run_meta(x)["loss"])
            df_per["arch"]       = df_per.name.map(lambda x: parse_run_meta(x)["arch"])
            df_per["domain"]     = df_per.name.map(lambda x: parse_run_meta(x)["domain"])
            df_per["subset"]     = df_per.name.str.split("-").str[2]  # the split name
            df_per["measure"]    = df_per.measure.map(str)

            all_per.append(df_per)

        # combine all per‐query
        if not all_per:
            continue
        df_all = pd.concat(all_per, ignore_index=True)
        df_all.to_csv(join(out_dir, f"perquery_cat_{group_name}.tsv.gz"),
                      sep="\t", index=False)

        # compute means
        df_means = (
            df_all
            .groupby(["dataset_id","loss","subset","arch","domain","measure"])["value"]
            .mean()
            .reset_index()
        )
        df_means.to_csv(join(out_dir, f"means_cat_{group_name}.tsv"),
                        sep="\t", index=False)

        # TOST across subsets
        records = []
        for (loss, arch), sub in df_all.groupby(["loss","arch"]):
            subsets = sub["subset"].unique().tolist()
            for i in range(len(subsets)):
                for j in range(i+1, len(subsets)):
                    s1, s2 = subsets[i], subsets[j]
                    for m in metric_names:
                        x = sub[(sub.subset==s1)&(sub.measure==m)]["value"]
                        y = sub[(sub.subset==s2)&(sub.measure==m)]["value"]
                        if len(x)==0 or len(y)==0:
                            continue
                        p, pl, pu = tost(x, y)
                        records.append({
                            "dataset_id": ds,
                            "group":      group_name,
                            "loss":       loss,
                            "arch":       arch,
                            "subset1":    s1,
                            "subset2":    s2,
                            "measure":    m,
                            "p_value":    p,
                            "p_lower":    pl,
                            "p_upper":    pu
                        })
        df_tost = pd.DataFrame.from_records(records)
        df_tost.to_csv(join(out_dir, f"tost_cat_{group_name}.tsv"),
                       sep="\t", index=False)

    return "Success!"


if __name__ == "__main__":
    Fire(main)