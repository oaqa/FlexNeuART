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
import sys
import numpy as np
import json
import argparse
import multiprocessing

#
# Convert a Wikipedia corpus used in the Facebook DPR project:
# https://github.com/facebookresearch/DPR/tree/master/data
# Optionally, one can specify a subset of the corpus by providing
# a numpy array with passage IDs to select.
#
# The input is a TAB-separated file with three columns: id, passage text, title
# This conversion script preserves original passages, but it also tokenizes them.
#

sys.path.append('.')

from scripts.data_convert.convert_common import read_stop_words, FileWrapper, add_retokenized_field, get_bert_tokenizer, \
                                                BERT_TOK_OPT, BERT_TOK_OPT_HELP
from scripts.data_convert.text_proc import SpacyTextParser
from scripts.config import STOPWORD_FILE, SPACY_MODEL, \
                        DOCID_FIELD, TEXT_RAW_FIELD_NAME, \
                        REPORT_QTY, IMAP_PROC_CHUNK_QTY, \
                        TEXT_BERT_TOKENIZED_NAME,\
                        TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME

parser = argparse.ArgumentParser(description='Convert a Wikipedia corpus downloaded from github.com/facebookresearch/DPR.')
parser.add_argument('--input_file', metavar='input file', help='input directory',
                    type=str, required=True)
parser.add_argument('--passage_ids', metavar='optional passage ids',
                    type=str, default=None, help='an optional numpy array with passage ids to select')
parser.add_argument('--out_file', metavar='output file',
                    help='output JSONL file',
                    type=str, required=True)
# Default is: Number of cores minus one for the spaning process
parser.add_argument('--proc_qty', metavar='# of processes', help='# of NLP processes to span',
                    type=int, default=multiprocessing.cpu_count() - 1)
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

args = parser.parse_args()
arg_vars = vars(args)
print(args)

bert_tokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = get_bert_tokenizer()

# Lower cased
stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)

flt_pass_ids = None
if args.passage_ids is not None:
    flt_pass_ids = set(np.load(args.passage_ids))
    print(f'Restricting parsing to {len(flt_pass_ids)} passage IDs')

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]

# Lower cased
text_processor = SpacyTextParser(SPACY_MODEL, stop_words,
                                keep_only_alpha_num=True, lower_case=True,
                                enable_pos=True)

class PassParseWorker:

    def __call__(self, line):

        if not line:
            return None

        line = line.strip()
        if not line:
            return None

        fields = line.split('\t')
        if ' '.join(fields) == 'id text title':
            return ''

        assert len(fields) == 3, f"Wrong format fline: {line}"
        # The passage text is not lower cased, please keep it this way.
        pass_id, raw_text, title = fields

        if flt_pass_ids is not None:
            if pass_id not in flt_pass_ids:
                return ''

        text_lemmas, text_unlemm = text_processor.proc_text(raw_text)
        title_lemmas, title_unlemm = text_processor.proc_text(title)

        doc = {DOCID_FIELD: pass_id,
               TEXT_FIELD_NAME: title_lemmas + ' ' + text_lemmas,
               TITLE_UNLEMM_FIELD_NAME: title_unlemm,
               TEXT_UNLEMM_FIELD_NAME: text_unlemm,
               TEXT_RAW_FIELD_NAME: title_unlemm + ' ' + raw_text}

        add_retokenized_field(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)
        return json.dumps(doc)


inp_file = FileWrapper(args.input_file)
out_file = FileWrapper(args.out_file, 'w')

proc_qty = args.proc_qty
print(f'Spanning {proc_qty} processes')
pool = multiprocessing.Pool(processes=proc_qty)
ln = 0
ln_ign = 0
for doc_str in pool.imap(PassParseWorker(), inp_file, IMAP_PROC_CHUNK_QTY):
    ln = ln + 1

    if doc_str is not None:
        if doc_str:
            out_file.write(doc_str + '\n')
        else:
            ln_ign += 1
    else:
        print('Ignoring misformatted line %d' % ln)

    if ln % REPORT_QTY == 0:
        print('Read %d passages, processed %d passages' % (ln, ln - ln_ign))

print('Processed %d passages' % ln)

inp_file.close()
out_file.close()

