#!/usr/bin/env python
# Convert MSMARCO passage collection
import sys
import json
import argparse
import multiprocessing
import pytorch_pretrained_bert

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    FileWrapper, read_stop_words, add_retokenized_field
from scripts.config import TEXT_BERT_TOKENIZED_NAME, MAX_DOC_SIZE, \
    TEXT_FIELD_NAME, DOCID_FIELD, BERT_BASE_MODEL, \
    TEXT_RAW_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, \
    IMAP_PROC_CHUNK_QTY, REPORT_QTY, SPACY_MODEL

parser = argparse.ArgumentParser(description='Convert MSMARCO-adhoc documents.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
parser.add_argument('--max_doc_size', metavar='max doc size bytes',
                    help='the threshold for the document size, if a document is larger it is truncated',
                    type=int, default=MAX_DOC_SIZE)
# Default is: Number of cores minus one for the spaning process
parser.add_argument('--proc_qty', metavar='# of processes', help='# of NLP processes to span',
                    type=int, default=multiprocessing.cpu_count() - 1)
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inp_file = FileWrapper(args.input)
out_file = FileWrapper(args.output, 'w')
max_doc_size = args.max_doc_size


bert_tokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

class PassParseWorker:
    def __call__(self, line):

        if not line:
            return None

        line = line[:max_doc_size]  # cut documents that are too long!
        fields = line.split('\t')
        if len(fields) != 2:
            return None

        pid, body = fields

        text, text_unlemm = nlp.proc_text(body)

        doc = {DOCID_FIELD: pid,
               TEXT_FIELD_NAME: text,
               TEXT_UNLEMM_FIELD_NAME: text_unlemm,
               TEXT_RAW_FIELD_NAME: body.lower()}
        add_retokenized_field(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)

        return json.dumps(doc) + '\n'


proc_qty = args.proc_qty
print(f'Spanning {proc_qty} processes')
pool = multiprocessing.Pool(processes=proc_qty)
ln = 0
for doc_str in pool.imap(PassParseWorker(), inp_file, IMAP_PROC_CHUNK_QTY):
    ln = ln + 1
    if doc_str is not None:
        out_file.write(doc_str)
    else:
        print('Ignoring misformatted line %d' % ln)

    if ln % REPORT_QTY == 0:
        print('Processed %d passages' % ln)

print('Processed %d passages' % ln)

inp_file.close()
out_file.close()
