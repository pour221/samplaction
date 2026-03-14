import argparse

from pathlib import Path

from .analysis import analyze

def args_parser():
    parser = argparse.ArgumentParser(prog='samplation', 
                                     description='CLI tool for filtering genomic assemblies outliers based on seqkit metrics')

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
    return parser.parse_args()

def main():
    args = args_parser()

    in_data = Path(args.input)
    out_data = Path(args.output_file)

    target_size = args.size
    threshold_num_seqs = args.max_num_seqs
    threshold_n50 = args.min_n50
    metrics = args.metrics.split()

    analyze(in_data, out_data,
            target_size=target_size,
            threshold_num_seqs=threshold_num_seqs,
            threshold_n50=threshold_n50,
            metrics=metrics)

if __name__ == "__main__":
    main()
