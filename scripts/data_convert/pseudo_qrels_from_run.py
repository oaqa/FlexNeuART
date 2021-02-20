#!/usr/bin/env python
# A script to generate pseudo-qrels from a TREC run
import sys

sys.path.append('.')
import argparse

from scripts.common_eval import readRunDict, writeQrels, QrelEntry, getSorteScoresFromScoreDict

parser = argparse.ArgumentParser('Generate pseudo-QRELs from a run')

parser.add_argument('--run_file',
                    metavar='run file',
                    type=str,
                    help=f'an input run file (can be compressed)',
                    required=True)

parser.add_argument('--out_qrel_file',
                    metavar='output QRELs',
                    type=str,
                    required=True,
                    help='a file to store generated pseudo-QRELs')

parser.add_argument('--top_rel_k',
                    metavar='top-k for relevant',
                    help='top relevant entries that are considered to be relevant',
                    type=int,
                    default=1)

parser.add_argument('--rel_grade',
                    metavar='relevant document grade',
                    type=int,
                    default=1,
                    help='a QREL grade for a relevant document')


args = parser.parse_args()

# Read a run dict
run_dict = readRunDict(args.run_file)


# Extract top-k entries to produce pseudo-qrels
pseudo_qrel_list = []
for qid in run_dict.keys():
    top_k_entries = getSorteScoresFromScoreDict(run_dict[qid])[0 : args.top_rel_k]
    add_qrel_list = [QrelEntry(queryId=qid, docId=did, relGrade=args.rel_grade) for did in top_k_entries.keys()]
    pseudo_qrel_list.extend(add_qrel_list)


# Finally write these QRELs
writeQrels(pseudo_qrel_list, args.out_qrel_file)
