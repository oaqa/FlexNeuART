#!/usr/bin/env python
import sys
import os
import json
import argparse
import pytorch_pretrained_bert
import tqdm

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser, Sentencizer
from scripts.data_convert.convert_common import get_retokenized, \
    STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    OUT_BITEXT_PATH_OPT, OUT_BITEXT_PATH_OPT_META, OUT_BITEXT_PATH_OPT_HELP, \
    FileWrapper, read_stop_words, add_retokenized_field
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
inp_file = FileWrapper(args.input)
out_queries = FileWrapper(args.output_queries, 'w')
out_qrels = FileWrapper(args.output_qrels, 'w')
min_query_tok_qty = args.min_query_token_qty
use_precomputed_negatives = args.use_precomputed_negatives
stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
out_bitext_dir = arg_vars[OUT_BITEXT_PATH_OPT]
nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)
sent_split = Sentencizer(SPACY_MODEL)

bitext_fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME]

bert_tokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)
    bitext_fields.append(TEXT_BERT_TOKENIZED_NAME)

bi_quest_files = {}
bi_answ_files = {}

if out_bitext_dir:
    if not os.path.exists(out_bitext_dir):
        os.makedirs(out_bitext_dir)

    for fn in bitext_fields:
        bi_quest_files[fn] = open(os.path.join(out_bitext_dir, BITEXT_QUESTION_PREFIX + fn), 'w')
        bi_answ_files[fn] = open(os.path.join(out_bitext_dir, BITEXT_ANSWER_PREFIX + fn), 'w')

for query_idx, fields in tqdm.tqdm(enumerate(dpr_json_reader(inp_file))):
    query = fields["question"]
    answer_list_lc = [s.lower() for s in fields["answers"]]
    query_lemmas, query_unlemm = nlp.proc_text(query)
    query_bert_tok = None

    query_toks = query_lemmas.split()
    if len(query_toks) >= min_query_tok_qty:
        doc = {
            DOCID_FIELD: query_idx,
            TEXT_FIELD_NAME: query_lemmas,
            TEXT_UNLEMM_FIELD_NAME: query_unlemm,
            TEXT_RAW_FIELD_NAME: query.lower()
        }
        add_retokenized_field(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)
        if TEXT_BERT_TOKENIZED_NAME in doc:
            query_bert_tok = doc[TEXT_BERT_TOKENIZED_NAME]

        doc_str = json.dumps(doc) + '\n'
        out_queries.write(doc_str)

        for entry in fields["positive_ctxs"]:
            psg_id = get_passage_id(entry)
            out_qrels.write(f'{query_idx} 0 {psg_id} 1\n')
            if bi_quest_files and bi_answ_files:
                title_text = entry["title"]
                if title_text:
                    _, title_unlemm = nlp.proc_text(title_text)
                    bi_quest_files[TITLE_UNLEMM_FIELD_NAME].write(query_unlemm + '\n')
                    bi_answ_files[TITLE_UNLEMM_FIELD_NAME].write(title_unlemm + '\n')

                for ctx_sent in sent_split(entry["text"]):
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
                        sent_lemmas, sent_unlemm = nlp.proc_text(ctx_sent)

                        bi_quest_files[TEXT_FIELD_NAME].write(query_lemmas + '\n')
                        bi_quest_files[TEXT_UNLEMM_FIELD_NAME].write(query_unlemm + '\n')

                        bi_answ_files[TEXT_FIELD_NAME].write(sent_lemmas + '\n')
                        bi_answ_files[TEXT_UNLEMM_FIELD_NAME].write(sent_unlemm + '\n')

                        if bert_tokenizer is not None:
                            answ_bert_tok = get_retokenized(bert_tokenizer, ctx_sent_lc)
                            bi_quest_files[TEXT_BERT_TOKENIZED_NAME].write(query_bert_tok + '\n')
                            bi_answ_files[TEXT_BERT_TOKENIZED_NAME].write(answ_bert_tok + '\n')


        if use_precomputed_negatives:
            for entry in fields["negative_ctxs"]:
                psg_id = get_passage_id(entry)
                out_qrels.write(f'{query_idx} 0 {psg_id} 0\n')

inp_file.close()
out_queries.close()
out_qrels.close()
for _, f in bi_quest_files.items():
    f.close()
for _, f in bi_answ_files.items():
    f.close()

