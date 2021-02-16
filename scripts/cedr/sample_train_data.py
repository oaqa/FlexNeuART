#!/usr/bin/env python

# A simple script that takes original training data and creates a set of folders containing:
# 1. Full data/query text files as well as QRELSs
# 2. One folder contains all training data, but a sample of test queries.
# 3. Several folders containing training data samples of the given size.
# The sampled set of queries is the same in all folders (they symlink)
import sys

sys.path.append('.')

import os
import argparse
import numpy as np
import shutil

from scripts.common_eval import readRunDict, writeRunDict
from scripts.cedr.data import read_pairs_dict, write_pairs_dict
from scripts.config import QREL_FILE
import scripts.utils as utils

parser = argparse.ArgumentParser('Training data sampling script')

parser.add_argument('--seed', metavar='random seed', help='random seed',
                    type=int, default=42)

parser.add_argument('--top_level_dir', metavar='source top level dir',
                    type=str, required=True,
                    help='Source top-level data directory')

parser.add_argument('--field_name', metavar='field name',
                    type=str, required=True,
                    help='field name (sub-directory)')

parser.add_argument('--src_subdir', metavar='source subdir',
                    type=str, required=True,
                    help='A source subdirectory: no slashes!')

parser.add_argument('--dst_subdir_pref', metavar='target path pref',
                    type=str, required=True,
                    help='A target subdirectory prefix: no slashes')

parser.add_argument('--test_query_sample_qty', metavar='# of sampled queries',
                    type=int, required=int,
                    help='A # of queries to sample')


parser.add_argument('--train_set_sample_qty', metavar='# of sampled trained sets (for each size)',
                    type=int, required=True,
                    help='A # of times we sample each training subset (for each query set size)')

parser.add_argument('--train_query_sample_qty', metavar='# of sampled training queries',
                    type=int, nargs='+',
                    help='A list for the number of test queries to sample')

parser.add_argument('--datafiles', metavar='data files', help='data files: docs & queries',
                    type=str, nargs='+',
                    default=['data_docs.tsv', 'data_query.tsv'])

parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                    type=str, default=QREL_FILE)

parser.add_argument('--train_pairs', metavar='paired train data', help='paired train data',
                    type=str,
                    default='train_pairs.tsv')

parser.add_argument('--valid_run', metavar='validation file', help='validation file',
                    type=str,
                    default='test_run.txt')


args = parser.parse_args()


print(args)

utils.set_all_seeds(args.seed)

assert not os.path.split(args.src_subdir)[0], "Source sub-directory should not be a complex path!"
assert not os.path.split(args.dst_subdir_pref)[0], "Target sub-directory should not be a complex path!"

src_dir = os.path.join(args.top_level_dir, args.src_subdir, args.field_name)

# First, we create a directory with complete training data, but a sample of validation queries.
subdir_full = args.dst_subdir_pref + '_full'
dst_dir_full = os.path.join(args.top_level_dir, subdir_full, args.field_name)
os.makedirs(dst_dir_full, exist_ok=True)

# Copy data files & qrels & train pairs to the full training set directory
for data_fn in args.datafiles + [args.train_pairs, args.qrels]:
    print('Copying:', data_fn)
    shutil.copy(os.path.join(src_dir, data_fn),
                os.path.join(dst_dir_full, data_fn),
                follow_symlinks=True)

# Read and sample validation queries
full_val_run = readRunDict(os.path.join(src_dir, args.valid_run))

val_qid_lst = list(full_val_run.keys())
val_qid_lst.sort()

val_qid_sample = np.random.choice(val_qid_lst, args.test_query_sample_qty, replace=False)

sample_val_run = {qid : full_val_run[qid] for qid in val_qid_sample}

writeRunDict(sample_val_run, os.path.join(dst_dir_full, args.valid_run))

with open(os.path.join(src_dir, args.train_pairs)) as f:
    train_pairs = read_pairs_dict(f)

# Second we will sample training queries as well.
train_qid_lst = list(train_pairs.keys())


for train_sample_qty in args.train_query_sample_qty:
    for sample_id in range(args.train_set_sample_qty):
        print(f'Training query sample size: {train_sample_qty}, sample {sample_id}')
        dst_dir_sample = os.path.join(args.top_level_dir,
                                      args.dst_subdir_pref + f'_tqty={train_sample_qty}_sampid={sample_id}',
                                      args.field_name)
        os.makedirs(dst_dir_sample, exist_ok=True)

        # First symlink full-size files  TRAIN PAIRS (though)
        for data_fn in args.datafiles + [args.qrels, args.valid_run]:
            print('Symlinking:', data_fn)
            os.symlink(os.path.join('..', '..', subdir_full, args.field_name, data_fn),
                       os.path.join(dst_dir_sample, data_fn))

        train_qid_sample = np.random.choice(train_qid_lst, train_sample_qty, replace=False)
        sample_train_pairs = {qid : train_pairs[qid] for qid in train_qid_sample}
        write_pairs_dict(sample_train_pairs, os.path.join(dst_dir_sample, args.train_pairs))



