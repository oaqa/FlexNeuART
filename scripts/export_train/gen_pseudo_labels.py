#!/usr/bin/env python
# This script generates pseudo-label QREL files from a previously generated


import argparse
import sys

sys.path.append('.')

from scripts.common_eval import read_run_dict, write_qrels, QrelEntry, get_sorted_scores_from_score_dict

parser = argparse.ArgumentParser('Generate pseudo-QRELs from a run')

parser.add_argument('--input_run', metavar='input run file',
                    required=True, type=str, help='input run file')
parser.add_argument('--out_qrels', metavar='output QREL file',
                    required=True, type=str, help='output QREL file')
parser.add_argument('--top_k', metavar='top k',
                    default=1,
                    type=int, help='top k entries to use as psedo relevant labels')
parser.add_argument('--grade', metavar='grade',
                    default=1, type=int, help='a grade for the relevance item')

args = parser.parse_args()

inp_run = read_run_dict(args.input_run)

qrels = []

for qid, run_dict in inp_run.items():
    for did, score in get_sorted_scores_from_score_dict(run_dict)[0: args.top_k]:
        qrels.append(QrelEntry(query_id=qid, doc_id=did, rel_grade=args.grade))


write_qrels(qrels, args.out_qrels)
