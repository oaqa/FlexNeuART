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
    Convert MSMARCO (v2) documents
"""

import json
import argparse
import multiprocessing

from tqdm import tqdm

ORIG_DOCID = 'orig_docid'

from flexneuart.io import FileWrapper, multi_file_linegen
from flexneuart.io.stopwords import read_stop_words, STOPWORD_FILE
from flexneuart.text_proc.parse import SpacyTextParser, add_retokenized_field
from flexneuart.data_convert import add_bert_tok_args, create_bert_tokenizer_if_needed

from flexneuart.data_convert import MSMARCO_PASS_V2_FILE_PATTERN

from flexneuart.config import TEXT_BERT_TOKENIZED_NAME, MAX_PASS_SIZE, \
    TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, DOCID_FIELD, \
    TEXT_RAW_FIELD_NAME, \
    IMAP_PROC_CHUNK_QTY, REPORT_QTY, SPACY_MODEL


parser = argparse.ArgumentParser(description='Convert MSMARCO (v2) passage')
parser.add_argument('--input', metavar='input directory',
                    help='input directory with un-tarred passage file',
                    type=str, required=True)
parser.add_argument('--output_main', metavar='main JSONL output',
                    help='the main output file, which contains all the data',
                    type=str, required=True)
parser.add_argument('--output_doc2pass', metavar='recoding JSONL',
                    help='JSONL containing a mapping from a document to all its passage IDs',
                    type=str, required=True)
parser.add_argument('--max_pass_size', metavar='max passage size bytes',
                    help='the threshold for the document size, if a document is larger it is truncated',
                    type=int, default=MAX_PASS_SIZE)

# Default is: Number of cores minus one for the spaning process
parser.add_argument('--proc_qty', metavar='# of processes', help='# of NLP processes to span',
                    type=int, default=multiprocessing.cpu_count() - 1)
add_bert_tok_args(parser)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inp_source = multi_file_linegen(args.input, MSMARCO_PASS_V2_FILE_PATTERN)
out_file = FileWrapper(args.output_main, 'w')
max_pass_size = args.max_pass_size

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)

bert_tokenizer = create_bert_tokenizer_if_needed(args)

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

class DocParseWorker:
    def __call__(self, line):

        if not line:
            return None

        fields = json.loads(line)
        passage = fields['passage'][:max_pass_size] # cut passages that are too long, for some reason they do occur in the corpus
        pid = fields['pid']
        docid = fields['docid']

        text, text_unlemm = nlp.proc_text(passage)

        doc = {DOCID_FIELD: pid,
               ORIG_DOCID: docid,
               TEXT_FIELD_NAME: text,
               TEXT_UNLEMM_FIELD_NAME : text_unlemm,
               TEXT_RAW_FIELD_NAME: passage}
        add_retokenized_field(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)

        return doc


proc_qty = args.proc_qty
print(f'Spanning {proc_qty} processes')
pool = multiprocessing.Pool(processes=proc_qty)
ln = 0

doc2pass_map = {}

for doc in pool.imap(DocParseWorker(), inp_source, IMAP_PROC_CHUNK_QTY):
    ln = ln + 1
    if doc is not None:
        did = doc[ORIG_DOCID]  # original document ID field
        pid = doc[DOCID_FIELD] # passage
        if not did in doc2pass_map:
            doc2pass_map[did] = []
        doc2pass_map[did].append(pid)
        doc_str = json.dumps(doc) + '\n'
        out_file.write(doc_str)

    else:
        # print('Misformatted line %d ignoring:' % ln)
        # print(line.replace('\t', '<field delimiter>'))
        print('Ignoring misformatted line %d' % ln)

    if ln % REPORT_QTY == 0:
        print('Processed %d passages' % ln)

print('Processed %d passage' % ln)

# inp_source is not a file and doesn't need closing
out_file.close()

print('Saving a document to passage ID mapping')

with FileWrapper(args.output_doc2pass, 'w') as out_file:
    for did, pid_arr in tqdm(doc2pass_map.items()):
        doc = { DOCID_FIELD : did,
                'pass_ids' : ' '.join(list(set(pid_arr)))
                }
        doc_str = json.dumps(doc) + '\n'
        out_file.write(doc_str)

