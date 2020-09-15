#!/usr/bin/env python
# This script generates pseudo-label QREL files from a previously generated


import argparse
import sys

sys.path.append('.')

from scripts.common_eval import readRunDict, writeQrels, QrelEntry, getSorteScoresFromScoreDict

parser = argparse.ArgumentParser('Exporting a neural Model1 model to a GIZA format (to run on CPU)')

parser.add_argument('--input_run', metavar='input run file',
                    required=True, type=str, help='input run file')
parser.add_argument('--out_qrels', metavar='output QREL file',
                    required=True, type=str, help='output QREL file')
parser.add_argument('--top_k', metavar='top k',
                    required=True, type=int, help='top k entries to use as psedo relevant labels')
parser.add_argument('--grade', metavar='grade',
                    default=1, type=int, help='a grade for the relevance item')

args = parser.parse_args()

inp_run = readRunDict(args.input_run)

qrels = []

for qid, run_dict in inp_run.items():
    for did, score in getSorteScoresFromScoreDict(run_dict)[0: args.top_k]:
        qrels.append(QrelEntry(queryId=qid, docId=did, relGrade=args.grade))


writeQrels(qrels, args.out_qrels)
