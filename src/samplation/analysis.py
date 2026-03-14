from pathlib import Path
import pandas as pd

from .qc_utils import (run_seqkit, prepare_seqkit_res, apply_threshold_filters, apply_quality_score_filter,
                      do_pca, do_dbscan, visualize_res)

def analyze(input_data: Path, output_file: Path,
            target_size: int | None = None,
            threshold_num_seqs: int = 1000, threshold_n50: int = 5000,
            metrics: list[str] | None = None) -> int:

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f'[STEP 0] Check inputs')

    if input_data.is_dir():
        print(f'[INFO] Input is folder with assemblies. Running seqkit')
        quality_df_path = run_seqkit(input_data, output_file.parent)
        original_quality_df = pd.read_csv(quality_df_path, sep=r'\s+')
    else:
        original_quality_df = pd.read_csv(input_data,  sep=r'\s+')

    print(f'[STEP 1] Applying threshold-based filtering')
    quality_df = prepare_seqkit_res(original_quality_df)
    quality_df = apply_threshold_filters(quality_df, target_size, threshold_num_seqs, threshold_n50)
    quality_df = apply_quality_score_filter(quality_df)
    print(f'[INFO] Samples dropped after filtering: {original_quality_df.shape[0] - quality_df.shape[0]}')

    print(f'[STEP 2] Running PCA')
    pca_df, explained_ratio = do_pca(quality_df, metrics)
    for num, ratio in enumerate(explained_ratio):
        print(f'[INFO] PС{num+1} explains {ratio*100:.2f}% of variance')

    print(f'[STEP 3] Running DBSCAN')
    pca_df = do_dbscan(pca_df)
    filtered_df = pca_df[pca_df['cluster'] != -1]
    quality_df['cluster'] = pca_df['cluster']
    quality_df['selected'] = quality_df.index.isin(filtered_df.index)

    print(f'[INFO] Visualizing results')
    plot_path = output_file.parent / f'{output_file.stem}.png'
    all_sample_output = output_file.parent / f'{output_file.stem}_all_samples{output_file.suffix}'

    visualize_res(pca_df, plot_path)
    quality_df.to_csv(all_sample_output)
    quality_df.loc[filtered_df.index, :].to_csv(output_file)

    print(f'[INFO] Total samples flagged as outliers: {original_quality_df.shape[0] - filtered_df.shape[0]}')
    print(f'[INFO] Final selected set contains {filtered_df.shape[0]} samples')

    print(f'[INFO] Filtered table saved at: {output_file}')
    print(f'[INFO] Full table with cluster labels saved at: {all_sample_output}')
    print(f'[INFO] Plot saved at: {plot_path}')

    print(f'[INFO] Done')
    return 0