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
#

"""
    Convert Cranfield collection queries.
"""
import argparse
import json

from tqdm import tqdm

from flexneuart.data_convert.cranfield import read_cranfield_data, BODY_FIED_NAME

from flexneuart.config import DOCID_FIELD, TEXT_RAW_FIELD_NAME, TEXT_FIELD_NAME
from flexneuart.config import TEXT_BERT_TOKENIZED_NAME, SPACY_MODEL, STOPWORD_FILE
from flexneuart.data_convert import add_bert_tok_args, create_bert_tokenizer_if_needed
from flexneuart.text_proc.parse import SpacyTextParser, add_retokenized_field

from flexneuart.io import FileWrapper
from flexneuart.io.stopwords import read_stop_words


parser = argparse.ArgumentParser(description='Convert Cranfield queries.')

parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
add_bert_tok_args(parser)

args = parser.parse_args()
print(args)

inp_data = read_cranfield_data(args.input)

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
#print(stop_words)

bert_tokenizer=create_bert_tokenizer_if_needed(args)

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

with FileWrapper(args.output, 'w') as outf:
    qid=0
    for query in tqdm(inp_data, desc='converting queries'):
        # Cranfield query IDs are all wrong and don't match QRELs
        # In QRELs a query ID is simply
        qid += 1

        e = {DOCID_FIELD : str(qid),
             TEXT_RAW_FIELD_NAME : query[TEXT_RAW_FIELD_NAME]}

        body_lemmas, body_unlemm = nlp.proc_text(query[BODY_FIED_NAME])

        e[TEXT_FIELD_NAME] = body_lemmas
        e[BODY_FIED_NAME] = body_unlemm

        add_retokenized_field(e, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)

        outf.write(json.dumps(e) + '\n')



