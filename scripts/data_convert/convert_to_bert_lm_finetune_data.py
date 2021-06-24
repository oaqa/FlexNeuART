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
# A generic script to generate data for finetuning BERT LM model from input JSONL file.
#
import sys
import argparse
import json

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import FileWrapper, replace_chars_nl
from scripts.config import SPACY_MODEL, TEXT_RAW_FIELD_NAME, REPORT_QTY

parser = argparse.ArgumentParser(description='Convert text BERT LM finetuning data file.')
parser.add_argument('--input', metavar='input JSON(L) file, can be compressed', help='input file',
                    type=str, required=True)
parser.add_argument('--output_pref', metavar='output file prefix', help='output file prefix',
                    type=str, required=True)
parser.add_argument('--max_set_size', metavar='max # of documents in a set',
                    default=1000_000,
                    help='the maximum number of set (in documents)',
                    type=int)
parser.add_argument('--lower_case', help='lowercase text',
                    action='store_true', default=False)

args = parser.parse_args()
print(args)

doc_qty = 0
set_qty = 0
set_id = 0

inp_file = FileWrapper(args.input)

nlp = SpacyTextParser(SPACY_MODEL, [], sent_split=True)


def out_file_name(pref, num):
    return pref + str(num) + '.txt'


print('Starting set 0')
out_file = FileWrapper(out_file_name(args.output_pref, set_id), 'w')

for line in inp_file:
    doc = json.loads(line)
    text_raw = doc[TEXT_RAW_FIELD_NAME]

    doc_sents = []

    for one_sent in nlp(text_raw).sents:
        one_sent = replace_chars_nl(str(one_sent)).strip()
        if args.lower_case:
            one_sent = one_sent.lower()
        if one_sent:
            doc_sents.append(one_sent)

    # Work hard to not write empty documents, b/c it'll upset the pregenerator
    if doc_sents:
        for one_sent in doc_sents:
            out_file.write(one_sent + '\n')
        out_file.write('\n')

    doc_qty += 1
    set_qty += 1
    if doc_qty % REPORT_QTY == 0:
        print('Processed %d docs' % doc_qty)

    if set_qty >= args.max_set_size:
        set_qty = 0
        set_id += 1
        print('Starting set %d' % set_id)
        out_file.close()
        out_file = FileWrapper(out_file_name(args.output_pref, set_id), 'w')

print('Processed %d docs' % doc_qty)

inp_file.close()
out_file.close()
