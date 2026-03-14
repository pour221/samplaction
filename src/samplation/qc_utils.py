import subprocess
import pandas as pd
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def run_seqkit(in_path, out_path):
    fastas = [i.as_posix() for i in Path(in_path).glob('*')
              if i.suffix in ('.fa','.fna','.fasta')]
    if not fastas:
        raise ValueError(f'No FASTA files found in {in_path}')

    with open(f'{out_path}/assembly_stats.tsv', 'w') as out:
        subprocess.run(['seqkit', 'stats', '-a', *fastas],
                       stdout=out,
                       check=True)

    return Path(f'{out_path}/assembly_stats.tsv')


def prepare_seqkit_res(in_quality_df):
    quality_df = in_quality_df.copy()
    quality_df['file'] = quality_df['file'].apply(lambda p: Path(p).name)

    if (quality_df['format'] != 'FASTA').any() or (quality_df['type'] != 'DNA').any():
        raise ValueError('Some files not in FASTA format or not DNA seq')

    quality_df = quality_df.set_index('file').drop(columns=['format', 'type'])

    return quality_df.replace(',', '', regex=True).astype(float)

def drop_outsiders(in_quality_df, target_size=None, threshold_num_seqs=1000, threshold_n50=5000):
    quality_df = in_quality_df.copy()

    if target_size is None:
        target_size = quality_df['sum_len'].mean()

    processing_df = quality_df.copy()
    processing_df =  processing_df[(processing_df['num_seqs'] < threshold_num_seqs) &
                                   (processing_df['N50'] > threshold_n50) &
                                   (processing_df['sum_len'] < target_size*1.2) &
                                   (target_size*0.8 < processing_df['sum_len'])]

    processing_df = processing_df[(processing_df['N50'] > processing_df['N50'].quantile(0.05)) &
                                  (processing_df['N50'] < processing_df['N50'].quantile(0.95)) &
                                  (processing_df['num_seqs'] < processing_df['num_seqs'].quantile(0.95))]

    return processing_df

def drop_by_quality_score(in_quality_df):
    quality_df = in_quality_df.copy()
    metrics = ['N50', 'max_len', 'num_seqs']

    z_score_normalization = StandardScaler()
    normalized_metrics = pd.DataFrame(z_score_normalization.fit_transform(quality_df[metrics]),
                                      columns=metrics, index=quality_df.index)

    quality_df['quality_score'] = normalized_metrics['N50'] + normalized_metrics['max_len'] - normalized_metrics['num_seqs']
    quality_df = quality_df[quality_df['quality_score'] >= quality_df['quality_score'].quantile(0.1)]

    return quality_df.drop('quality_score', axis=1)

def do_pca(in_quality_df: pd.DataFrame, metrics=None) -> tuple[pd.DataFrame, list]:
    quality_df = in_quality_df.copy()
    if metrics is None:
        metrics = ['N50','sum_len','num_seqs','max_len']

    pc_cols = [f'PC{i+1}' for i in range(len(metrics))]
    scaler = StandardScaler()
    prep_metrics = scaler.fit_transform(quality_df[metrics])
    pca = PCA()
    quality_pca = pca.fit_transform(prep_metrics)

    return pd.DataFrame(columns=pc_cols, index=quality_df.index, data=quality_pca), pca.explained_variance_ratio_

def do_dbscan(pca_df):
    working_pca = pca_df.copy()
    dbscan = DBSCAN()
    working_pca['cluster'] = dbscan.fit_predict(working_pca[['PC1','PC2']])

    return working_pca

def visualize_res(dbscan_df, output_file: Path):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=120)
    axs = axs.ravel()
    xmin = dbscan_df['PC1'].min() - 1
    xmax = dbscan_df['PC1'].max() + 1

    ymin = dbscan_df['PC2'].min() - 1
    ymax = dbscan_df['PC2'].max() + 1

    sns.scatterplot(data=dbscan_df, x='PC1', y='PC2', ax=axs[0], hue='cluster', palette='coolwarm')
    axs[0].set_title("PCA (all assemblies)")
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)

    sns.scatterplot(data=dbscan_df[dbscan_df['cluster'] != -1], x='PC1', y='PC2', ax=axs[1])
    axs[1].set_title("PCA (filtered)")
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(ymin, ymax)

    plt.savefig(f'{output_file.parent}/{output_file.stem}.png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
