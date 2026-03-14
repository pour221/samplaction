from pathlib import Path
import pandas as pd

from .qc_utils import (run_seqkit, prepare_seqkit_res, drop_outsiders, drop_by_quality_score,
                      do_pca, do_dbscan, visualize_res)

def analyze(input_data: Path, output_file: Path, target_size=None, threshold_num_seqs=1000, threshold_n50=5000,
            metrics=None):

    if input_data.is_dir():
        quality_df_path = run_seqkit(input_data, output_file.parent)
        quality_df = pd.read_csv(quality_df_path, sep=r'\s+')
    else:
        quality_df = pd.read_csv(input_data,  sep=r'\s+')

    quality_df = prepare_seqkit_res(quality_df)
    quality_df = drop_outsiders(quality_df, target_size, threshold_num_seqs, threshold_n50)
    quality_df = drop_by_quality_score(quality_df)

    pca_df, explained_ratio = do_pca(quality_df, metrics)
    pca_df = do_dbscan(pca_df)

    filtered_df = pca_df[pca_df['cluster'] != -1]

    visualize_res(pca_df, output_file)
    quality_df.loc[filtered_df.index, :].to_csv(output_file)
    print(filtered_df)


    return 0





