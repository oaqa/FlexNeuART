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
# Adding predicted query fields for the MS MARCO *PASSAGE* collection.
# https://github.com/castorini/docTTTTTquery
#
# It reads all the predictions into memory
#
import sys
import argparse
import json

from tqdm import tqdm

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import read_stop_words, jsonl_gen, FileWrapper, STOPWORD_FILE
from scripts.config import SPACY_MODEL

from scripts.config import DOCID_FIELD, TEXT_FIELD_NAME

DOC2QUERY_FIELD_TEXT = 'doc2query_text'
DOC2QUERY_FIELD_TEXT_UNLEMM = 'doc2query_text_unlemm'

parser = argparse.ArgumentParser(description='Add doc2query fields to the existing JSONL data entries')

parser.add_argument('--input', metavar='input JSONL file', help='input JSONL file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--output', metavar='output JSONL file', help='output JSONL file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--target_fusion_field', metavar='target fusion field',
                    help='the name of the target field that will store concatenation of the lemmatized doc2query text and the original lemmatized text',
                    type=str, required=True)

parser.add_argument('--predictions_path',
                    required=True, metavar='doc2query predictions',
                    help='File containing predicted queries for passage data: one per each passage.')


args = parser.parse_args()
print(args)

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

doc_id_prev = None
predicted_queries = []
target_fusion_field = args.target_fusion_field

for line in tqdm(FileWrapper(args.predictions_path), desc='reading predictions'):
    line = line.strip()
    if line:
        predicted_queries.append(line)
print(f'Read predictions for {len(predicted_queries)} passages')

pass_qty = 0

with FileWrapper(args.output, 'w') as outf:
    for doce in tqdm(jsonl_gen(args.input), desc='adding doc2query fields'):
        doc_id = doce[DOCID_FIELD]

        text, text_unlemm = nlp.proc_text(predicted_queries[pass_qty])
        doce[target_fusion_field] = doce[TEXT_FIELD_NAME] + ' ' + text
        doce[DOC2QUERY_FIELD_TEXT] = text
        doce[DOC2QUERY_FIELD_TEXT_UNLEMM] = text_unlemm

        pass_qty += 1
        outf.write(json.dumps(doce) + '\n')


if pass_qty != len(predicted_queries):
    raise Exception(f'Mismatch in the number of predicted queries:  {len(predicted_queries)} ' +
                    f' and the total number of passages: {pass_qty}')
