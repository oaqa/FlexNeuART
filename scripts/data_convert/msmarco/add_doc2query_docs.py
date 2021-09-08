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
"""
    Adding predicted query fields for the MS MARCO *DOCUMENT* collection.
    https://github.com/castorini/docTTTTTquery
   
    It reads all the predictions into memory
"""
import argparse
import json

from tqdm import tqdm

from flexneuart.text_proc.parse import SpacyTextParser
from flexneuart.io.stopwords import read_stop_words, STOPWORD_FILE
from flexneuart.io import jsonl_gen, FileWrapper
from flexneuart.config import SPACY_MODEL

from flexneuart.config import DOCID_FIELD, TEXT_FIELD_NAME

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


parser.add_argument('--doc_ids_path',
                    required=True, metavar='doc2query doc IDs',
                    help='File mapping segments to doc ids.')
parser.add_argument('--predictions_path',
                    required=True, metavar='doc2query predictions',
                    help='File containing predicted queries.')

docid_to_preds = {}

args = parser.parse_args()
print(args)

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

doc_id_prev = None
predicted_queries = []
target_fusion_field = args.target_fusion_field

for doc_id, predicted_queries_partial in tqdm(zip(FileWrapper(args.doc_ids_path),
                                                  FileWrapper(args.predictions_path)),
                                              desc='reading predictions'):
    doc_id = doc_id.strip()
    if doc_id_prev is not None and doc_id_prev != doc_id:
        if predicted_queries and doc_id_prev is not None:
            docid_to_preds[doc_id_prev] = ' '.join(predicted_queries).strip()
        predicted_queries = []

    doc_id_prev = doc_id
    predicted_queries.append(predicted_queries_partial)

# Not forgetting about the last batch
if predicted_queries and doc_id_prev is not None:
    docid_to_preds[doc_id_prev] = ' '.join(predicted_queries)


with FileWrapper(args.output, 'w') as outf:
    for doce in tqdm(jsonl_gen(args.input), desc='adding doc2query fields'):
        doc_id = doce[DOCID_FIELD]
        if doc_id in docid_to_preds:
            text, text_unlemm = nlp.proc_text(docid_to_preds[doc_id])
            doce[target_fusion_field] = doce[TEXT_FIELD_NAME] + ' ' + text
            doce[DOC2QUERY_FIELD_TEXT] = text
            doce[DOC2QUERY_FIELD_TEXT_UNLEMM] = text_unlemm
        else:
            print(f'WARNING: no predictions for {doc_id}')

        outf.write(json.dumps(doce) + '\n')
