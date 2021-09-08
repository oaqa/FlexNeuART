#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
    A file to generate queries & qrels from the ORCAS query collection:
    ORCAS: 18 Million Clicked Query-Document Pairs for Analyzing Search.
    Nick Craswell Daniel Campos Bhaskar Mitra Emine Yilmaz Bodo von Billerbeck.
   
"""
import os
import json
import argparse

from flexneuart.io import FileWrapper
from flexneuart.io.stopwords import read_stop_words, STOPWORD_FILE
from flexneuart.text_proc.parse import SpacyTextParser, add_retokenized_field
from flexneuart.data_convert import add_bert_tok_args, create_bert_tokenizer_if_needed

from flexneuart.data_convert import MAX_NUM_QUERY_OPT_HELP, MAX_NUM_QUERY_OPT

from flexneuart.config import TEXT_BERT_TOKENIZED_NAME, TEXT_UNLEMM_FIELD_NAME, \
    TEXT_FIELD_NAME, DOCID_FIELD, \
    TEXT_RAW_FIELD_NAME, \
    REPORT_QTY, SPACY_MODEL, QUESTION_FILE_JSON, QREL_FILE

from flexneuart.io.qrels import QrelEntry, write_qrels
from flexneuart.io.queries import read_queries

ORCAS_QID_PREF='orcas_'

parser = argparse.ArgumentParser(description='Convert MSMARCO-ORCAS queries & qrels.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--filter_query_dir', metavar='filtering query dir',
                    default=[],
                    help=f'all queries found in {QUESTION_FILE_JSON} files from these directories are ignored',
                    nargs='*')
parser.add_argument('--out_dir', metavar='output directory', help='output directory',
                    type=str, required=True)
parser.add_argument('--min_query_token_qty', type=int, default=0,
                    metavar='min # of query tokens', help='ignore queries that have smaller # of tokens')
parser.add_argument('--' + MAX_NUM_QUERY_OPT, type=int, default=None, help=MAX_NUM_QUERY_OPT_HELP)
add_bert_tok_args(parser)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inp_file = FileWrapper(args.input)
max_query_qty = arg_vars[MAX_NUM_QUERY_OPT]
if max_query_qty < 0 or max_query_qty is None:
    max_query_qty = float('inf')

ignore_queries = set()

for qfile_dir in args.filter_query_dir:
    qfile_name = os.path.join(qfile_dir, QUESTION_FILE_JSON)
    for e in read_queries(qfile_name):
        ignore_queries.add(e[TEXT_FIELD_NAME])
    print('Read queries from: ' + qfile_name)

print('A list of queries to ignore has %d entries' % (len(ignore_queries)))


if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

out_file_queries = FileWrapper(os.path.join(args.out_dir, QUESTION_FILE_JSON), 'w')
out_file_qrels_name = os.path.join(args.out_dir, QREL_FILE)

min_query_tok_qty = args.min_query_token_qty

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)
nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

bert_tokenizer = create_bert_tokenizer_if_needed(args)

qrel_list = []

# Input file is a TSV file
ln = 0

prev_qid = ''
gen_query_qty = 0

for line in inp_file:
    ln += 1
    line = line.strip()
    if not line:
        continue
    fields = line.split('\t')
    if len(fields) != 4:
        print('Misformated line %d ignoring:' % ln)
        print(line.replace('\t', '<field delimiter>'))
        continue

    qid_orig, query_orig, did, _  = fields
    qid = ORCAS_QID_PREF + qid_orig

    query_lemmas, query_unlemm = nlp.proc_text(query_orig)

    if query_lemmas == '':
        continue
    if query_lemmas in ignore_queries:
        print(f"Ignoring query, which is found in specified query files. Raw query: '{query_orig}' lemmatized query '{query_lemmas}'")

    query_toks = query_lemmas.split()
    if len(query_toks) >= min_query_tok_qty:

        qrel_list.append(QrelEntry(query_id=qid, doc_id=did, rel_grade=1))

        # Entries are sorted by the query ID
        if prev_qid != qid:
            doc = {DOCID_FIELD: qid,
                   TEXT_FIELD_NAME: query_lemmas,
                   TEXT_UNLEMM_FIELD_NAME: query_unlemm,
                   TEXT_RAW_FIELD_NAME: query_orig}
            add_retokenized_field(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)

            doc_str = json.dumps(doc) + '\n'
            out_file_queries.write(doc_str)
            gen_query_qty += 1
            if gen_query_qty >= max_query_qty:
                break

    prev_qid = qid

    if ln % REPORT_QTY == 0:
        print('Processed %d input line' % ln)

print('Processed %d input lines' % ln)

write_qrels(qrel_list, out_file_qrels_name)

inp_file.close()
out_file_queries.close()

