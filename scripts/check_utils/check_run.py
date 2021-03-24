#!/usr/bin/env python
# A simple script to assess validity of (TREC) runs before submission
import sys
import argparse

sys.path.append('.')

from scripts.data_convert.convert_common import read_doc_ids_from_forward_file_header, read_queries, DOCID_FIELD, FileWrapper

parser = argparse.ArgumentParser(description='Run basic run checks')
parser.add_argument('--run_file', metavar='run file',
                    help='a run file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--query_file', metavar='query file',
                    help='a query file',
                    type=str, required=True)
parser.add_argument('--run_id', metavar='run id',
                    help='optional run id to check',
                    default=None,
                    type=str)
parser.add_argument('--fwd_index_file', metavar='forward index catalog file',
                    help='the "catalog" file of the forward index',
                    type=str, required=True)
parser.add_argument('--min_exp_doc_qty',
                    metavar='min # of docs per query to expect',
                    help='min # of docs per query to expect',
                    type=int, required=True)

args = parser.parse_args()

print('Reading document IDs from the index')
all_doc_ids = read_doc_ids_from_forward_file_header(args.fwd_index_file)
print('Reading queries')
queries = read_queries(args.query_file)

query_ids = []
query_doc_qtys = {}

for e in queries:
    qid = e[DOCID_FIELD]
    query_ids.append(qid)

# Some copy-paste from common_eval.read_run_dict, but ok for now
file_name = args.run_file
with FileWrapper(file_name) as f:
    prev_query_id = None

    # Check for repeating document IDs and improperly sorted entries
    for ln, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        fld = line.split()
        if len(fld) != 6:
            ln += 1
            raise Exception(
                f'Invalid line {ln} in run file {file_name} expected 6 white-space separated fields by got: {line}')

        qid, _, docid, rank, score_str, run_id = fld
        if prev_query_id is None or qid != prev_query_id:
            seen_docs = set()
            prev_query_id = qid
            prev_score = float('inf')

        try:
            score = float(score_str)
        except:
            raise Exception(
                f'Invalid score {score_str} {ln} in run file {file_name}: {line}')

        if score > prev_score:
            raise Exception(
                f'Invalid line {ln} in run file {file_name} increasing score!')
        if docid not in all_doc_ids:
            raise Exception(
                f'Invalid line {ln} in run file {file_name} document id not found in the index: {docid}')
        if docid in seen_docs:
            raise Exception(
                f'Invalid line {ln} in run file {file_name} repeating document {docid}')

        if args.run_id is not None and run_id != args.run_id:
            raise Exception(
                f'Invalid line {ln} in run file {file_name} invalid run id {run_id}')

        prev_score = score
        seen_docs.add(docid)
        if not qid in query_doc_qtys:
            query_doc_qtys[qid] = 0
        query_doc_qtys[qid] += 1




# Finally print per-query statistics and report queries that have fewer than a give number of results generated
print('# of results per query:')
n_warn = 0
for qid in query_ids:
    qty = query_doc_qtys[qid] if qid in query_doc_qtys else 0

    print(f'{qid} {qty}')
    if qty < args.min_exp_doc_qty:
        print(f'WARNING: query {qid} has fewer results than expected!')
        n_warn += 1

print(f'Checking is complete! # of warning {n_warn}')
