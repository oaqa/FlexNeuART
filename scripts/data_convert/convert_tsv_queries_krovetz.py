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
    Generic converter for TSV-format queries that applies Krovetz stemmer.
"""
import json
import argparse

from flexneuart.io import FileWrapper
from flexneuart.io.stopwords import read_stop_words, STOPWORD_FILE
from flexneuart.text_proc.parse import KrovetzStemParser, add_retokenized_field
from flexneuart.data_convert import add_bert_tok_args, create_bert_tokenizer_if_needed

from flexneuart.config import TEXT_BERT_TOKENIZED_NAME, \
    TEXT_FIELD_NAME, DOCID_FIELD, \
    TEXT_RAW_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, \
    REPORT_QTY

parser = argparse.ArgumentParser(description='Convert TSV-format queries with Krovetz stemming.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
add_bert_tok_args(parser)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inp_file = FileWrapper(args.input)
out_file = FileWrapper(args.output, 'w')

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)
nlp : KrovetzStemParser = KrovetzStemParser(stop_words)

bert_tokenizer = create_bert_tokenizer_if_needed(args)

# Input file is a TSV file
ln = 0
for line in inp_file:
    ln += 1
    line = line.strip()
    if not line:
        continue
    fields = line.split('\t')
    if len(fields) != 2:
        print('Misformated line %d ignoring:' % ln)
        print(line.replace('\t', '<field delimiter>'))
        continue

    did, query_orig = fields

    query_stemmed = nlp(query_orig.lower())

    doc = {DOCID_FIELD: did,
           TEXT_FIELD_NAME: query_stemmed,
           TEXT_RAW_FIELD_NAME: query_orig}
    add_retokenized_field(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)
    
    doc_str = json.dumps(doc) + '\n'
    out_file.write(doc_str)

    if ln % REPORT_QTY == 0:
        print('Processed %d queries' % ln)

print('Processed %d queries' % ln)

inp_file.close()
out_file.close()
