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
#
# Convert MSMARCO (v2) documents
#
import sys
import json
import argparse
import multiprocessing
import pytorch_pretrained_bert

from tqdm import tqdm

ORIG_DOCID = 'orig_docid'

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import MSMARCO_PASS_V2_FILE_PATTERN, \
                                                STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    multi_file_linegen, FileWrapper, read_stop_words, add_retokenized_field, pretokenize_url
from scripts.config import TEXT_BERT_TOKENIZED_NAME, \
    TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, DOCID_FIELD, BERT_BASE_MODEL, \
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

# Default is: Number of cores minus one for the spaning process
parser.add_argument('--proc_qty', metavar='# of processes', help='# of NLP processes to span',
                    type=int, default=multiprocessing.cpu_count() - 1)
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inp_source = multi_file_linegen(args.input, MSMARCO_PASS_V2_FILE_PATTERN)
out_file = FileWrapper(args.output_main, 'w')

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)

bert_tokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

class DocParseWorker:
    def __call__(self, line):

        if not line:
            return None

        fields = json.loads(line)
        passage = fields['passage']
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

