import subprocess
import pandas as pd
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def run_seqkit(in_path: Path, out_path: Path) -> Path:
    fastas = [i.as_posix() for i in Path(in_path).glob('*')
              if i.suffix in ('.fa','.fna','.fasta')]
    if not fastas:
        raise ValueError(f'No FASTA files found in {in_path}')

    with open(out_path / 'assembly_stats.tsv', 'w') as out:
        subprocess.run(['seqkit', 'stats', '-a', *fastas],
                       stdout=out,
                       check=True)

    return out_path / 'assembly_stats.tsv'


def prepare_seqkit_res(in_quality_df: pd.DataFrame) -> pd.DataFrame:
    quality_df = in_quality_df.copy()
    quality_df['file'] = quality_df['file'].apply(lambda p: Path(p).name)

    if (quality_df['format'] != 'FASTA').any() or (quality_df['type'] != 'DNA').any():
        raise ValueError('Some input files are not in FASTA format or do not contain DNA sequences')

    quality_df = quality_df.set_index('file').drop(columns=['format', 'type'])

    return quality_df.replace(',', '', regex=True).astype(float)

def apply_threshold_filters(in_quality_df: pd.DataFrame, target_size: int | None =None,
                            threshold_num_seqs: int = 1000, threshold_n50: int = 5000) -> pd.DataFrame:
    quality_df = in_quality_df.copy()

    if target_size is None:
        target_size = quality_df['sum_len'].mean()

    processing_df = quality_df.copy()
    processing_df =  processing_df[(processing_df['num_seqs'] < threshold_num_seqs) &
                                   (processing_df['N50'] > threshold_n50) &
                                   (processing_df['sum_len'] < target_size*1.2) &
                                   (target_size*0.8 < processing_df['sum_len'])]
    if processing_df.empty:
        raise ValueError(f'No samples left after hard filtering')

    processing_df = processing_df[(processing_df['N50'] > processing_df['N50'].quantile(0.05)) &
                                  (processing_df['N50'] < processing_df['N50'].quantile(0.95)) &
                                  (processing_df['num_seqs'] < processing_df['num_seqs'].quantile(0.95))]
    if processing_df.empty:
        raise ValueError(f'No samples left after quantile-based filtering')

    return processing_df

def apply_quality_score_filter(in_quality_df: pd.DataFrame) -> pd.DataFrame:
    quality_df = in_quality_df.copy()

    if quality_df.shape[0] < 3:
        return quality_df

    metrics = ['N50', 'max_len', 'num_seqs']

    missing_metrics = [m for m in metrics if m not in quality_df.columns]
    if missing_metrics:
        raise ValueError(f'Missing metrics for quality score calculation: {missing_metrics}')

    z_score_normalization = StandardScaler()
    normalized_metrics = pd.DataFrame(z_score_normalization.fit_transform(quality_df[metrics]),
                                      columns=metrics, index=quality_df.index)

    quality_df['quality_score'] = normalized_metrics['N50'] + normalized_metrics['max_len'] - normalized_metrics['num_seqs']
    quality_df = quality_df[quality_df['quality_score'] >= quality_df['quality_score'].quantile(0.1)]

    return quality_df.drop('quality_score', axis=1)

def do_pca(in_quality_df: pd.DataFrame, metrics: list[str] | None = None) -> tuple[pd.DataFrame, list]:
    quality_df = in_quality_df.copy()
    if metrics is None:
        metrics = ['N50','sum_len','num_seqs','max_len']

    if quality_df.shape[0] < 2:
        raise ValueError('At least 2 samples are required for PCA')

    missing_metrics = [m for m in metrics if m not in quality_df.columns]
    if missing_metrics:
        raise ValueError(f'Missing metrics for PCA: {missing_metrics}')

    pc_cols = [f'PC{i+1}' for i in range(len(metrics))]
    scaler = StandardScaler()
    prep_metrics = scaler.fit_transform(quality_df[metrics])
    pca = PCA()
    quality_pca = pca.fit_transform(prep_metrics)

    return pd.DataFrame(columns=pc_cols, index=quality_df.index, data=quality_pca), pca.explained_variance_ratio_

def do_dbscan(pca_df: pd.DataFrame, eps: float = 0.8, min_samples: int = 5) -> pd.DataFrame:
    working_pca = pca_df.copy()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    working_pca['cluster'] = dbscan.fit_predict(working_pca[['PC1','PC2']])

    return working_pca

def visualize_res(dbscan_df: pd.DataFrame, output_file: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=120)
    axs = axs.ravel()
    xmin = dbscan_df['PC1'].min() - 1
    xmax = dbscan_df['PC1'].max() + 1

    ymin = dbscan_df['PC2'].min() - 1
    ymax = dbscan_df['PC2'].max() + 1

    sns.scatterplot(data=dbscan_df, x='PC1', y='PC2', ax=axs[0], hue='cluster', palette='tab10')
    axs[0].set_title("PCA (all assemblies)")
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)

    sns.scatterplot(data=dbscan_df[dbscan_df['cluster'] != -1], x='PC1', y='PC2', ax=axs[1])
    axs[1].set_title("PCA (filtered)")
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(ymin, ymax)

    plt.savefig(output_file, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
