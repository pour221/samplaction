import sys
import argparse

from pathlib import Path

from .analysis import analyze

def args_parser():
    parser = argparse.ArgumentParser(prog='samplaction',
                                     description='CLI tool for filtering genomic assemblies outliers based on seqkit metrics')

    parser.add_argument('--version', action='version', version='samplaction 0.2')

    parser.add_argument('-i', '--input', required=True,
                        help='tsv file with assembly stats (output from seqkit stats -a) or path to folder with genomes to run seqkit')
    parser.add_argument('-o', '--output_file', required=True,
                        help='path to store result table with selected samples')

    parser.add_argument('-s','--size', default=None, type=int,
                        help='The reference (or approximate) genome size in bp (e.g., 6000000). If not specified, the average value based on the table will be used.')
    parser.add_argument('--max_num_seqs', default=1000, type=int,
                        help='Threshold for num_seqs metric to drop. Default=1000')
    parser.add_argument('--min_n50', default=5000, type=int,
                        help='Threshold for N50 metric to drop. Default=5000')
    parser.add_argument('--metrics', default='N50 sum_len num_seqs max_len',
                        help='Space-separated list of metrics used for PCA. Default: "N50 sum_len num_seqs max_len"')

    parser.add_argument('--eps', default=0.8, type=float,
                        help='EPS for DBSCAN')

    parser.add_argument('--min_samples', default=5, type=int,
                        help='Minimum samples for DBSCAN cluster')

    return parser.parse_args()

def main():
    args = args_parser()
    # Mandatory params
    in_data = Path(args.input)
    out_data = Path(args.output_file)
    # Threshold based filtering params
    target_size = args.size
    threshold_num_seqs = args.max_num_seqs
    threshold_n50 = args.min_n50
    metrics = args.metrics.split()
    # DBSCAN params
    eps = args.eps
    min_samples = args.min_samples
    try:
        analyze(in_data, out_data,
                target_size=target_size,
                threshold_num_seqs=threshold_num_seqs,
                threshold_n50=threshold_n50,
                metrics=metrics,
                eps=eps,
                min_samples=min_samples)

    except (FileNotFoundError, ValueError, IsADirectoryError) as e:
        print(f'[ERROR] {e}', file=sys.stderr)
        raise SystemExit(1)
    except Exception as e:
        print(f'[ERROR] Unexpected error: {e}', file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
