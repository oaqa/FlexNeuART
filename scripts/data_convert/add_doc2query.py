#!/usr/bin/env python
# Adding predicted query fields
# https://github.com/castorini/docTTTTTquery
# It reads all the predictions into memory
import sys
import argparse
import json

from tqdm import tqdm

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import readStopWords, jsonlGen, FileWrapper, STOPWORD_FILE
from scripts.config import SPACY_MODEL

from scripts.config import DOCID_FIELD

DOC2QUERY_FIELD_TEXT = 'doc2query_text'
DOC2QUERY_FIELD_TEXT_UNLEMM = 'doc2query_text_ulemm'

parser = argparse.ArgumentParser(description='Add doc2query fields to the existing JSONL data entries')

parser.add_argument('--input', metavar='input JSONL file', help='input JSONL file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--output', metavar='output JSONL file', help='output JSONL file (can be compressed)',
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

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)

nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True)

doc_id_prev = None
predicted_queries = []

doc_id_prev = None
predicted_queries = []
for doc_id, predicted_queries_partial in tqdm(zip(FileWrapper(args.doc_ids_path),
                                                  FileWrapper(args.predictions_path)), desc='reading predictions'):
    doc_id = doc_id.strip()
    if doc_id_prev is not None and doc_id_prev != doc_id:
        if predicted_queries and doc_id_prev is not None:
            docid_to_preds[doc_id_prev] = ' '.join(predicted_queries).strip()
        predicted_queries = []

    doc_id_prev = doc_id
    predicted_queries.append(predicted_queries_partial)

if predicted_queries and doc_id_prev is not None:
    docid_to_preds[doc_id_prev] = ' '.join(predicted_queries)


with FileWrapper(args.output, 'w') as outf:
    for doce in tqdm(jsonlGen(args.input), desc='adding doc2query fields'):
        doc_id = doce[DOCID_FIELD]
        if doc_id in docid_to_preds:
            text, text_unlemm = nlp.procText(docid_to_preds[doc_id])
            doce[DOC2QUERY_FIELD_TEXT] = text
            doce[DOC2QUERY_FIELD_TEXT_UNLEMM] = text_unlemm

            outf.write(json.dumps(doce) + '\n')

        else:
            print(f'WARNING: no predictionsf or {doc_id}')
            
