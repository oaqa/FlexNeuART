#!/usr/bin/env python

import sys
import argparse
import json
import pytorch_pretrained_bert

from tqdm import tqdm

sys.path.append('.')

from scripts.data_convert.cranfield.cranfield_common import *
from scripts.data_convert.text_proc import SpacyTextParser
from scripts.config import BERT_BASE_MODEL, TEXT_BERT_TOKENIZED_NAME, SPACY_MODEL
from scripts.data_convert.convert_common \
    import STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    FileWrapper, readStopWords, addRetokenizedField


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

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
#print(stopWords)

bert_tokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True)

with FileWrapper(args.output, 'w') as outf:
    for doc in tqdm(inp_data, desc='converting documents'):
        e = {DOCID_FIELD : doc[DOCID_FIELD],
             TEXT_RAW_FIELD_NAME : doc[TEXT_RAW_FIELD_NAME]}

        title_lemmas, _ = nlp.procText(doc[TITLE_FIELD_NAME])
        author_lemmas, _ = nlp.procText(doc[AUTHOR_FIELD_NAME])
        venue_lemmas, _ = nlp.procText(doc[VENUE_FIELD_NAME])
        body_lemmas, _ = nlp.procText(doc[BODY_FIED_NAME])

        e[TEXT_FIELD_NAME] = ' '.join([title_lemmas, author_lemmas, venue_lemmas, body_lemmas])
        e[TITLE_FIELD_NAME] = title_lemmas
        e[AUTHOR_FIELD_NAME] = author_lemmas
        e[VENUE_FIELD_NAME] = venue_lemmas
        e[BODY_FIED_NAME] = body_lemmas

        addRetokenizedField(e, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)

        outf.write(json.dumps(e) + '\n')



