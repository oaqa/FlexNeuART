#!/usr/bin/env python
# A simple script to assess validity of (TREC) runs before submission
import sys
import argparse

sys.path.append('.')

from scripts.data_convert.convert_common import readDocIdsFromForwardFileHeader, readQueries, DOCID_FIELD, FileWrapper

parser = argparse.ArgumentParser(description='Run basic run checks')
parser.add_argument('--run_file', metavar='run file',
                    help='a run file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--query_file', metavar='query file',
                    help='a query file',
                    type=str, required=True)
parser.add_argument('--fwd_index_file', metavar='forward index catalog file',
                    help='the "catalog" file of the forward index',
                    type=str, required=True)
parser.add_argument('--min_exp_doc_qty',
                    metavar='min # of docs per query to expect',
                    help='min # of docs per query to expect',
                    type=int, required=True)

args = parser.parse_args()

print('Reading document IDs from the index')
allDocIds = readDocIdsFromForwardFileHeader(args.fwd_index_file)
print('Reading queries')
queries = readQueries(args.query_file)

query_ids = []
query_doc_qtys = {}

for e in queries:
    qid = e[DOCID_FIELD]
    query_ids.append(qid)

# Some copy-paste from common_eval.readRunDict, but ok for now
fileName = args.run_file
with FileWrapper(fileName) as f:
    prevQueryId = None

    # Check for repeating document IDs and improperly sorted entries
    for ln, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        fld = line.split()
        if len(fld) != 6:
            ln += 1
            raise Exception(
                f'Invalid line {ln} in run file {fileName} expected 6 white-space separated fields by got: {line}')

        qid, _, docid, rank, scoreStr, _ = fld
        if prevQueryId is None or qid != prevQueryId:
            seenDocs = set()
            prevQueryId = qid
            prevScore = float('inf')

        try:
            score = float(scoreStr)
        except:
            raise Exception(
                f'Invalid score {scoreStr} {ln} in run file {fileName}: {line}')

        if score > prevScore:
            raise Exception(
                f'Invalid line {ln} in run file {fileName} increasing score!')
        if docid in seenDocs:
            raise Exception(
                f'Invalid line {ln} in run file {fileName} repeating document {docid}')

        prevScore = score
        seenDocs.add(docid)
        if not qid in query_doc_qtys:
            query_doc_qtys[qid] = 0
        query_doc_qtys[qid] += 1




# Finally print per-query statistics and report queries that have fewer than a give number of results generated
print('# of results per query:')
nWarn = 0
for qid in query_ids:
    qty = query_doc_qtys[qid] if qid in query_doc_qtys else 0

    print(f'{qid} {qty}')
    if qty < args.min_exp_doc_qty:
        print(f'WARNING: query {qid} has fewer results than expected!')
        nWarn += 1

print(f'Checking is complete! # of warning {nWarn}')
