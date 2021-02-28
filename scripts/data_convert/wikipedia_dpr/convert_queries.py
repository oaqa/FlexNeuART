#!/usr/bin/env python
import sys
import os
import json
import argparse
import pytorch_pretrained_bert
import tqdm

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser, Sentencizer
from scripts.data_convert.convert_common import getRetokenized, \
    STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    OUT_BITEXT_PATH_OPT, OUT_BITEXT_PATH_OPT_META, OUT_BITEXT_PATH_OPT_HELP, \
    FileWrapper, readStopWords, addRetokenizedField
from scripts.config import TEXT_BERT_TOKENIZED_NAME, \
    TEXT_FIELD_NAME, DOCID_FIELD, BERT_BASE_MODEL, \
    TEXT_RAW_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME, \
    SPACY_MODEL, BITEXT_QUESTION_PREFIX, BITEXT_ANSWER_PREFIX
from scripts.data_convert.wikipedia_dpr.utils import dpr_json_reader, get_passage_id


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
    parser.add_argument('--' + OUT_BITEXT_PATH_OPT, metavar=OUT_BITEXT_PATH_OPT_META,
                        help=OUT_BITEXT_PATH_OPT_HELP,
                        type=str, default=None)
    parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

    args = parser.parse_args()

    return args


args = parse_args()
arg_vars=vars(args)
inpFile = FileWrapper(args.input)
outQueries = FileWrapper(args.output_queries, 'w')
outQrels = FileWrapper(args.output_qrels, 'w')
minQueryTokQty = args.min_query_token_qty
usePrecomputedNegatives = args.use_precomputed_negatives
stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
outBitextDir = arg_vars[OUT_BITEXT_PATH_OPT]
nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True)
sentSplit = Sentencizer(SPACY_MODEL)

bitext_fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME]

bertTokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bertTokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)
    bitext_fields.append(TEXT_BERT_TOKENIZED_NAME)

biQuestFiles = {}
biAnswFiles = {}

if outBitextDir:
    if not os.path.exists(outBitextDir):
        os.makedirs(outBitextDir)

    for fn in bitext_fields:
        biQuestFiles[fn] = open(os.path.join(outBitextDir, BITEXT_QUESTION_PREFIX + fn), 'w')
        biAnswFiles[fn] = open(os.path.join(outBitextDir, BITEXT_ANSWER_PREFIX + fn), 'w')

for query_idx, fields in tqdm.tqdm(enumerate(dpr_json_reader(inpFile))):
    query = fields["question"]
    answer_list_lc = [s.lower() for s in fields["answers"]]
    query_lemmas, query_unlemm = nlp.procText(query)
    query_bert_tok = None

    query_toks = query_lemmas.split()
    if len(query_toks) >= minQueryTokQty:
        doc = {
            DOCID_FIELD: query_idx,
            TEXT_FIELD_NAME: query_lemmas,
            TEXT_UNLEMM_FIELD_NAME: query_unlemm,
            TEXT_RAW_FIELD_NAME: query.lower()
        }
        addRetokenizedField(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bertTokenizer)
        if TEXT_BERT_TOKENIZED_NAME in doc:
            query_bert_tok = doc[TEXT_BERT_TOKENIZED_NAME]

        docStr = json.dumps(doc) + '\n'
        outQueries.write(docStr)

        for entry in fields["positive_ctxs"]:
            psgId = get_passage_id(entry)
            outQrels.write(f'{query_idx} 0 {psgId} 1\n')
            if biQuestFiles and biAnswFiles:
                title_text = entry["title"]
                if title_text:
                    _, title_unlemm = nlp.procText(title_text)
                    biQuestFiles[TITLE_UNLEMM_FIELD_NAME].write(query_unlemm + '\n')
                    biAnswFiles[TITLE_UNLEMM_FIELD_NAME].write(title_unlemm + '\n')

                for ctx_sent in sentSplit(entry["text"]):
                    ctx_sent = str(ctx_sent)
                    ctx_sent_lc = ctx_sent.lower()
                    # This is sometimes can be a false positive, b/c it doesn't
                    # take sentence boundaries into account. However,
                    # we know that a positive context already contains an answer,
                    # so this shouldn't affect performance nearly at all (and these
                    # false positives would be rare)
                    has_answ = False
                    for answ in answer_list_lc:
                        if ctx_sent_lc.find(answ) >= 0:
                            has_answ = True
                            break

                    if has_answ:
                        sent_lemmas, sent_unlemm = nlp.procText(ctx_sent)

                        biQuestFiles[TEXT_FIELD_NAME].write(query_lemmas + '\n')
                        biQuestFiles[TEXT_UNLEMM_FIELD_NAME].write(query_unlemm + '\n')

                        biAnswFiles[TEXT_FIELD_NAME].write(sent_lemmas + '\n')
                        biAnswFiles[TEXT_UNLEMM_FIELD_NAME].write(sent_unlemm + '\n')

                        if bertTokenizer is not None:
                            answ_bert_tok = getRetokenized(bertTokenizer, ctx_sent_lc)
                            biQuestFiles[TEXT_BERT_TOKENIZED_NAME].write(query_bert_tok + '\n')
                            biAnswFiles[TEXT_BERT_TOKENIZED_NAME].write(answ_bert_tok + '\n')


        if usePrecomputedNegatives:
            for entry in fields["negative_ctxs"]:
                psgId = get_passage_id(entry)
                outQrels.write(f'{query_idx} 0 {psgId} 0\n')

inpFile.close()
outQueries.close()
outQrels.close()
for _, f in biQuestFiles.items():
    f.close()
for _, f in biAnswFiles.items():
    f.close()

