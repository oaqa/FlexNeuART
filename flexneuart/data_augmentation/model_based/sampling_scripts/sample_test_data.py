import argparse
import os
import numpy as np

"""
This script is used to generate a random sample from the test set of a given file
"""

def perform_checks(args):
    if os.path.exists(args.input_dir)==False:
        raise ValueError("Input path does not exist")
    if os.path.exists(args.output_dir):
        raise ValueError("Output path exists.")
    else:
        os.system("mkdir -p {0}".format(args.output_dir))
    
    expected_files = ["data_docs.tsv", "data_query.tsv", "qrels.txt", "test_run.txt", "train_pairs.tsv"]
    files = os.listdir(args.input_dir)

    for f in files:
        if f not in expected_files:
            raise ValueError("File {0} not found in input directory.".format(f))
    
    if args.fraction is not None and (args.fraction<=0.0 or args.fraction>1.0):
        raise ValueError("--fraction needs to be > 0.0 and <= 1.0")

def copy_files(args):
    to_move = ["data_docs.tsv", "data_query.tsv", "qrels.txt", "train_pairs.tsv"]
    for f in to_move:
        os.system("cp {0} {1}".format(os.path.join(args.input_dir, f), 
                                    os.path.join(args.output_dir, f)))

def read_data_as_dict(data_file):
    orig_dict = {}
    with open(data_file) as f:
        for line in f:
            qid = line[:-1].split()[0]
            if qid not in orig_dict:
                orig_dict[qid] = [line[:-1]]
            else:
                orig_dict[qid].append(line[:-1])
    return orig_dict

def main(args):
    test_path = os.path.join(args.input_dir, "test_run.txt")
    test_dict = read_data_as_dict(test_path)

    num_queries = len(test_dict)
    if args.count!=None:
        sample_size = min(num_queries, args.count)
    else:
        sample_size = num_queries * args.fraction
    
    cands = list(test_dict.keys())
    sample_qids = set(np.random.choice(cands, sample_size, replace=False))

    out_file = os.path.join(args.output_dir, "test_run.txt")

    op_file = open(out_file, "w")

    with open(test_path) as ip_file:
        for line in ip_file:
            sample_qid = line.split()[0]
            if sample_qid in sample_qids:
                op_file.write(line)
    op_file.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True,
                        help='Path to input directory with FlexNeuART data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to directory where output files will be written')
    
    sampling_criteria = parser.add_mutually_exclusive_group(required=True)
    sampling_criteria.add_argument('--count', type=int, help='Number of queries to sample')
    sampling_criteria.add_argument('--fraction', type=float, help='Fraction of total queries to sample')

    args = parser.parse_args()

    perform_checks(args)
    main(args)