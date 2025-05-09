from fire import Fire
import os
from os.path import join
import pyterrier as pt
import pandas as pd
import statsmodels.stats.weightstats as ws
from ir_measures import AP, NDCG

if not pt.started():
    pt.init()


# helper to perform TOST equivalence
def tost(x, y, low_eq=-0.01, high_eq=0.01):
    p_low, p_high, _, _ = ws.ttost_ind(x, y, low_eq, high_eq)
    return p_low, p_high


# parse metadata from filename
def parse_run_meta(fname):
    base = fname.replace(".res.gz","")
    meta = {}
    # loss
    for loss in ("LCE","marginMSE","RankNet","KL"):
        if loss.lower() in base.lower():
            meta["loss"] = loss
            break
    # arch
    meta["arch"] = "CE" if "crossencoder" in base.lower() else "BE"
    # domain
    dom = None
    if "random" in base.lower():
        dom = "Random"
    elif "ensemble" in base.lower():
        dom = "Ensemble"
    elif "crossencoder" in base.lower():
        dom = "Cross-Encoder"
    elif "bm25" in base.lower():
        dom = "BM25"
    meta["domain"] = dom or "UNK"
    return meta


def main(run_dir: str, out_dir: str, rel: int = 1, baseline: str = None):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(run_dir) if f.endswith(".res.gz")]
    beir_files = [f for f in files if "beir" in f]
    # define the three benchmark groups
    dataset_ids = sorted({f.split("_")[1] for f in beir_files})
    groups = {
        "dl19": ["dl-2019"],
        "dl20": ["dl-2020"],
        "beir": [ds for ds in dataset_ids if ds not in {"dl-2019", "dl-2020"}],
    }

    for group_name, ds_list in groups.items():
        all_perdfs = []
        # for DL groups, there’s only one dataset; for 'beir' we loop all sub‐datasets
        for ds in ds_list:
            # standardize the PT dataset id
            ds_key = ds
            if ds in {"dbpedia-entity","fever","fiqa","hotpotqa","nfcorpus","quora","scifact"}:
                ds_key = f"{ds}/test"
            if 'cqadupstack' in ds_key:
                ds_key = '/'.join(ds_key.split('-'))
            if 'webis-touche2020' in ds_key:
                ds_key = 'webis-touche2020/v2'
            if 'dl-' in ds_key:
                if '19' in ds_key:
                    ds_key = 'msmarco-passage/msmarco-passage/trec-dl-2019/judged'
                elif '20' in ds_key:
                    ds_key = 'msmarco-passage/msmarco-passage/trec-dl-2020/judged'
            # load the dataset
                pt_ds = pt.get_dataset(f"irds:{ds_key}")
                rel = 2
            else:
                pt_ds = pt.get_dataset(f"irds:beir/{ds_key}")
            topics, qrels = pt_ds.get_topics("text"), pt_ds.get_qrels()

            metrics = [AP(rel=rel), NDCG(cutoff=10)]
            metric_names = [str(m) for m in metrics]

            # select run‐files for this dataset
            subset = [f for f in files if f.split("_")[1] == ds]
            runs, names = [], []
            baseline_run = None
            for fn in subset:
                runname = fn.replace(".res.gz","")
                run = pt.transformer.get_transformer(pt.io.read_results(join(run_dir, fn)))
                if baseline and runname.endswith(f"_{baseline}"):
                    baseline_run = run
                else:
                    runs.append(run)
                    names.append(runname)
            if baseline and baseline_run:
                runs.insert(0, baseline_run)
                names.insert(0, baseline)

            # run per‐query evaluation
            df_per = pt.Experiment(
                runs, topics, qrels,
                perquery=True,
                eval_metrics=metrics,
                names=names,
                baseline=0 if baseline else None
            )
            # attach parsed metadata
            metas = [parse_run_meta(n) for n in df_per["run"]]
            df_meta = pd.DataFrame(metas)
            df_per = pd.concat([df_per.reset_index(drop=True), df_meta], axis=1)

            all_perdfs.append(df_per)

        # combine (for BEIR) or single‐ds (for DL)
        df_all = pd.concat(all_perdfs, ignore_index=True)

        # save full per‐query for debugging
        df_all.to_csv(join(out_dir, f"perquery_{group_name}.tsv.gz"), sep="\t", index=False)

        # compute mean metrics by loss/domain/arch
        df_means = (
            df_all
            .groupby(["loss","domain","arch"])[metric_names]
            .mean()
            .reset_index()
        )
        df_means.to_csv(join(out_dir, f"means_{group_name}.tsv"), sep="\t", index=False)

        # TOST across domains for each loss/arch/metric
        records = []
        for (loss, arch), sub in df_all.groupby(["loss","arch"]):
            doms = sub["domain"].unique().tolist()
            for i in range(len(doms)):
                for j in range(i+1, len(doms)):
                    d1, d2 = doms[i], doms[j]
                    for m in metric_names:
                        x = sub.loc[sub["domain"]==d1, m]
                        y = sub.loc[sub["domain"]==d2, m]
                        p_lo, p_hi = tost(x, y)
                        records.append({
                            "group": group_name,
                            "loss": loss,
                            "arch": arch,
                            "domain1": d1,
                            "domain2": d2,
                            "metric": m,
                            "p_lower": p_lo,
                            "p_upper": p_hi
                        })
        df_tost = pd.DataFrame.from_records(records)
        df_tost.to_csv(join(out_dir, f"tost_{group_name}.tsv"), sep="\t", index=False)

    return "Success!"


if __name__ == "__main__":
    Fire(main)