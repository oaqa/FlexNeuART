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
# Convert Cranfield collection documents.
#

import sys
import argparse
import json

from tqdm import tqdm

sys.path.append('.')

from scripts.data_convert.cranfield.cranfield_common import *
from scripts.data_convert.text_proc import SpacyTextParser
from scripts.config import TEXT_BERT_TOKENIZED_NAME, SPACY_MODEL
from scripts.data_convert.convert_common \
    import STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    FileWrapper, read_stop_words, add_retokenized_field, get_bert_tokenizer

parser = argparse.ArgumentParser(description='Convert Cranfield documents.')

parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inp_data = read_cranfield_data(args.input)

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
#print(stop_words)

bert_tokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = get_bert_tokenizer()

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

with FileWrapper(args.output, 'w') as outf:
    for doc in tqdm(inp_data, desc='converting documents'):
        e = {DOCID_FIELD : doc[DOCID_FIELD],
             TEXT_RAW_FIELD_NAME : doc[TEXT_RAW_FIELD_NAME]}

        title_lemmas, _ = nlp.proc_text(doc[TITLE_FIELD_NAME])
        author_lemmas, _ = nlp.proc_text(doc[AUTHOR_FIELD_NAME])
        venue_lemmas, _ = nlp.proc_text(doc[VENUE_FIELD_NAME])
        body_lemmas, _ = nlp.proc_text(doc[BODY_FIED_NAME])

        e[TEXT_FIELD_NAME] = ' '.join([title_lemmas, author_lemmas, venue_lemmas, body_lemmas])
        e[TITLE_FIELD_NAME] = title_lemmas
        e[AUTHOR_FIELD_NAME] = author_lemmas
        e[VENUE_FIELD_NAME] = venue_lemmas
        e[BODY_FIED_NAME] = body_lemmas

        add_retokenized_field(e, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)

        outf.write(json.dumps(e) + '\n')



