import os
import pandas as pd
import networkx as nx
import string

def annotate_equivalence(df_tost, alpha=0.05, metric="nDCG@10"):  # use NDCG@10 for header
    records = []
    # group by group and loss
    for (group, loss), sub in df_tost.groupby(["group", "loss"]):
        # within each, build graph on domain nodes
        G = nx.Graph()
        # get unique domains
        doms = pd.unique(sub[["domain1", "domain2"]].values.ravel())
        G.add_nodes_from(doms)
        for _, row in sub.iterrows():
            if row['metric'] == metric and row['p_lower'] > alpha and row['p_upper'] > alpha:
                G.add_edge(row['domain1'], row['domain2'])
        comps = list(nx.connected_components(G))
        comps.sort(key=lambda comp: sorted(comp)[0])
        for idx, comp in enumerate(comps):
            label = string.ascii_uppercase[idx]
            for d in comp:
                records.append({
                    "group": group,
                    "loss": loss,
                    "domain": d,
                    "eq_class": label
                })
    return pd.DataFrame.from_records(records)


def generate_table(out_dir, alpha=0.05):
    # load mean tables and tost tables
    groups = ['dl19', 'dl20', 'beir']
    means = {}
    tosts = {}
    for g in groups:
        means[g] = pd.read_csv(os.path.join(out_dir, f"means_{g}.tsv"), sep="\t")
        tosts[g] = pd.read_csv(os.path.join(out_dir, f"tost_{g}.tsv"), sep="\t")

    # annotate eq classes for each group
    eq_frames = []
    for g in groups:
        df_eq = annotate_equivalence(tosts[g], alpha=alpha)
        df_eq['group'] = g
        eq_frames.append(df_eq)
    df_eq_all = pd.concat(eq_frames, ignore_index=True)

    # merge eq into means
    df_merged = []
    for g in groups:
        df = means[g].copy()
        df['group'] = g
        df = df.merge(
            df_eq_all[df_eq_all['group'] == g],
            on=['group','loss','domain'],
            how='left'
        )
        df_merged.append(df)
    df_all = pd.concat(df_merged, ignore_index=True)

    # pivot for latex
    # rows: loss, domain
    # columns: group -> arch -> metric
    df_all = df_all.rename(columns={'ndcg_cut_10': 'NDCG@10', 'map': 'MAP'})
    table = df_all.pivot_table(
        index=['loss','domain'],
        columns=['group','arch'],
        values=['NDCG@10','MAP','eq_class'],
        aggfunc='first'
    )

    # begin latex output
    latex = []
    latex.append('% requires \usepackage{booktabs,multirow}')
    latex.append('\begin{table}[t]')
    latex.append('  \centering')
    latex.append('  \footnotesize')
    latex.append('  \setlength{\tabcolsep}{3pt}')
    header = ('  \begin{tabular}{ll' + 'cccc' * len(groups) + '}')
    latex.append(header)
    latex.append('  \toprule')
    # first header row with superscripts
    sp = []
    for g in groups:
        # get eq classes for BE and CE for this group (first loss suffices)
        subs = df_eq_all[df_eq_all['group']==g]
        # pick any loss domain combination
        row = subs.iloc[0]
        # for BE and CE
        be_cl = subs[subs['domain']=='BM25']['eq_class'].iloc[0] if 'BM25' in subs['domain'].values else ""
        ce_cl = subs[subs['domain']=='Cross-Encoder']['eq_class'].iloc[0] if 'Cross-Encoder' in subs['domain'].values else ""
        sp.append(f"{be_cl},{ce_cl}" if ce_cl else be_cl)
    latex.append(f"    &  & \multicolumn{{4}}{{c}}{{TREC DL’19\textsuperscript{{{sp[0]}}}}} & \multicolumn{{4}}{{c}}{{TREC DL’20\textsuperscript{{{sp[1]}}}}} & \multicolumn{{4}}{{c}}{{BEIR mean\textsuperscript{{{sp[2]}}}}} \\")
    latex.append('  \cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}')
    # second header row
    latex.append('    &  & \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} & \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} & \multicolumn{2}{c}{BE} & \multicolumn{2}{c}{CE} \\')
    latex.append('  \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}')
    # metric row
    latex.append('    Loss & Domain & nDCG & MAP & nDCG & MAP & nDCG & MAP & nDCG & MAP & nDCG & MAP & nDCG & MAP \\')
    latex.append('  \midrule')

    # body rows
    for loss in table.index.levels[0]:
        for dom in table.loc[loss].index:
            row = table.loc[(loss, dom)]
            vals = []
            for g in groups:
                for arch in ['BE','CE']:
                    ndcg = row['NDCG@10', g, arch]
                    m = row['MAP', g, arch]
                    vals.append(f"{ndcg:.2f}")
                    vals.append(f"{m:.2f}")
            latex.append(f"  {loss} & {dom} & " + " & ".join(vals) + " \\")
    latex.append('  \bottomrule')
    latex.append('\end{tabular}')
    latex.append('\end{table}')

    # print or save
    output = '\n'.join(latex)
    print(output)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    generate_table(args.out_dir)
