#!/usr/bin/env python
import sys
import json
import argparse
import pytorch_pretrained_bert
import tqdm

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    FileWrapper, readStopWords, addRetokenizedField
from scripts.config import TEXT_BERT_TOKENIZED_NAME, \
    TEXT_FIELD_NAME, DOCID_FIELD, BERT_BASE_MODEL, \
    TEXT_RAW_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, \
    IMAP_PROC_CHUNK_QTY, REPORT_QTY, SPACY_MODEL
from scripts.data_convert.wikipedia_dpr.utils import dpr_json_reader


def parse_args():
    parser = argparse.ArgumentParser(description='Script converts raw queries files from DPR repository '
                                                 'into FlexNeuArt internal format.')
    parser.add_argument('--input', metavar='input file', help='input file',
                        type=str, required=True)
    parser.add_argument('--output_queries', metavar='output queries file', help='output queries file',
                        type=str, required=True)
    parser.add_argument('--output_qrels', metavar='output qrels file', help='output qrels file',
                        type=str, required=True)
    parser.add_argument('--use_precomputed_negatives', type=bool, default=False, help='Use negative_ctxs field as a source for negative examples')
    parser.add_argument('--min_query_token_qty', type=int, default=0,
                        metavar='min # of query tokens', help='ignore queries that have smaller # of tokens')
    parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)
    return args


args = parse_args()
inpFile = FileWrapper(args.input)
outQueries = FileWrapper(args.output_queries, 'w')
outQrels = FileWrapper(args.output_qrels, 'w')
minQueryTokQty = args.min_query_token_qty
usePrecomputedNegatives = args.use_precomputed_negatives
stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True)

if BERT_TOK_OPT in arg_vars:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bertTokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

for query_idx, fields in tqdm.tqdm(enumerate(dpr_json_reader(inpFile))):
    query = fields["question"]
    query_lemmas, query_unlemm = nlp.procText(query)

    query_toks = query_lemmas.split()
    if len(query_toks) >= minQueryTokQty:
        doc = {
            DOCID_FIELD: query_idx,
            TEXT_FIELD_NAME: query_lemmas,
            TEXT_UNLEMM_FIELD_NAME: query_unlemm,
            TEXT_RAW_FIELD_NAME: query.lower()
        }
        addRetokenizedField(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bertTokenizer)

        docStr = json.dumps(doc) + '\n'
        outQueries.write(docStr)

        for entry in fields["positive_ctxs"]:
            outQrels.write(f'{query_idx} 0 {entry["passage_id"]} 1\n')

        if usePrecomputedNegatives:
            for entry in fields["negative_ctxs"]:
                outQrels.write(f'{query_idx} 0 {entry["passage_id"]} 0\n')

inpFile.close()
outQueries.close()
outQrels.close()
