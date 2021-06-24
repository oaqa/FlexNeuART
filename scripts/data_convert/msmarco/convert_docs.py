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
# Convert MSMARCO documents
#
import sys
import json
import argparse
import multiprocessing
import pytorch_pretrained_bert

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    FileWrapper, read_stop_words, add_retokenized_field, pretokenize_url
from scripts.config import TEXT_BERT_TOKENIZED_NAME, MAX_DOC_SIZE, \
    TEXT_FIELD_NAME, DOCID_FIELD, BERT_BASE_MODEL, \
    TITLE_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME, \
    TEXT_RAW_FIELD_NAME, \
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

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)

bert_tokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

class DocParseWorker:
    def __call__(self, line):

        if not line:
            return None
        line = line[:max_doc_size]  # cut documents that are too long!
        fields = line.split('\t')
        if len(fields) != 4:
            return None

        did, url, title, body = fields

        url_pretok = pretokenize_url(url)

        url_lemmas, url_unlemm = nlp.proc_text(url_pretok)
        title_lemmas, title_unlemm = nlp.proc_text(title)
        body_lemmas, body_unlemm = nlp.proc_text(body)

        text = title_lemmas + ' ' + body_lemmas
        text = text.strip()
        text_raw = (title.strip() + ' ' + body.strip())

        doc = {DOCID_FIELD: did,
               'url' : url_lemmas,
               'url_unlemm' : url_unlemm,
               TEXT_FIELD_NAME: text,
               TITLE_FIELD_NAME : title_lemmas,
               TITLE_UNLEMM_FIELD_NAME: title_unlemm,
               'body': body_unlemm,
               TEXT_RAW_FIELD_NAME: text_raw}
        add_retokenized_field(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)

        doc_str = json.dumps(doc) + '\n'
        return doc_str


proc_qty = args.proc_qty
print(f'Spanning {proc_qty} processes')
pool = multiprocessing.Pool(processes=proc_qty)
ln = 0
for doc_str in pool.imap(DocParseWorker(), inp_file, IMAP_PROC_CHUNK_QTY):
    ln = ln + 1
    if doc_str is not None:
        out_file.write(doc_str)
    else:
        # print('Misformatted line %d ignoring:' % ln)
        # print(line.replace('\t', '<field delimiter>'))
        print('Ignoring misformatted line %d' % ln)

    if ln % REPORT_QTY == 0:
        print('Processed %d docs' % ln)

print('Processed %d docs' % ln)

inp_file.close()
out_file.close()
