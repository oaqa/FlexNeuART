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
# Convert MSMARCO (v2) documents
#
import json
import argparse
import multiprocessing

from flexneuart.text_proc import SpacyTextParser
from flexneuart.data_convert.utils import MSMARCO_DOC_V2_FILE_PATTERN, \
                                                STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    multi_file_linegen, FileWrapper, read_stop_words, add_retokenized_field, pretokenize_url, get_bert_tokenizer
from scripts.config import TEXT_BERT_TOKENIZED_NAME, MAX_DOC_SIZE, \
    TEXT_FIELD_NAME, DOCID_FIELD, \
    TITLE_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME, \
    TEXT_RAW_FIELD_NAME, \
    IMAP_PROC_CHUNK_QTY, REPORT_QTY, SPACY_MODEL


parser = argparse.ArgumentParser(description='Convert MSMARCO (v2) documents')
parser.add_argument('--input', metavar='input directory',
                    help='input directory with un-tarred document file',
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

inp_source = multi_file_linegen(args.input, MSMARCO_DOC_V2_FILE_PATTERN)
out_file = FileWrapper(args.output, 'w')
max_doc_size = args.max_doc_size

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)

bert_tokenizer=None
if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = get_bert_tokenizer()

nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)

class DocParseWorker:
    def __call__(self, line):

        if not line:
            return None

        fields = json.loads(line)
        body = fields['body'][:max_doc_size] # cut documents that are too long!
        did = fields['docid']
        title = fields['title']
        url = fields['url']
        headings = fields['headings']

        url_pretok = pretokenize_url(url)

        url_lemmas, url_unlemm = nlp.proc_text(url_pretok)
        title_lemmas, title_unlemm = nlp.proc_text(title)
        body_lemmas, body_unlemm = nlp.proc_text(body)
        headings_lemmas, headings_unlemm = nlp.proc_text(headings)

        text = ' '.join([url_lemmas, headings_lemmas, title_lemmas, body_lemmas])
        text = text.strip()

        text_raw = ' '.join([url, headings, title, body])

        doc = {DOCID_FIELD: did,
               'url' : url_lemmas,
               'url_unlemm' : url_unlemm,
               'headings': headings_lemmas,
               'headings_unlemm': headings_unlemm,
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
for doc_str in pool.imap(DocParseWorker(), inp_source, IMAP_PROC_CHUNK_QTY):
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

# inp_source is not a file and doesn't need closing
out_file.close()
