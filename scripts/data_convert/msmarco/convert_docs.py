#!/usr/bin/env python
# Convert MSMARCO document collection
import sys
import json
import argparse
import multiprocessing
import pytorch_pretrained_bert

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    FileWrapper, readStopWords, addRetokenizedField, pretokenizeUrl
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

inpFile = FileWrapper(args.input)
outFile = FileWrapper(args.output, 'w')
maxDocSize = args.max_doc_size

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)

bertTokenizer=None
if BERT_TOK_OPT in arg_vars:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bertTokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True)

class DocParseWorker:
    def __call__(self, line):

        if not line:
            return None
        line = line[:maxDocSize]  # cut documents that are too long!
        fields = line.split('\t')
        if len(fields) != 4:
            return None

        did, url, title, body = fields

        url_pretok = pretokenizeUrl(url)

        url_lemmas, url_unlemm = nlp.procText(url_pretok)
        title_lemmas, title_unlemm = nlp.procText(title)
        body_lemmas, body_unlemm = nlp.procText(body)

        text = title_lemmas + ' ' + body_lemmas
        text = text.strip()
        text_raw = (title.strip() + ' ' + body.strip()).lower()
        doc = {DOCID_FIELD: did,
               'url' : url_lemmas,
               'url_unlemm' : url_unlemm,
               TEXT_FIELD_NAME: text,
               TITLE_FIELD_NAME : title_lemmas,
               TITLE_UNLEMM_FIELD_NAME: title_unlemm,
               'body': body_unlemm,
               TEXT_RAW_FIELD_NAME: text_raw}
        addRetokenizedField(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bertTokenizer)

        docStr = json.dumps(doc) + '\n'
        return docStr


proc_qty = args.proc_qty
print(f'Spanning {proc_qty} processes')
pool = multiprocessing.Pool(processes=proc_qty)
ln = 0
for docStr in pool.imap(DocParseWorker(), inpFile, IMAP_PROC_CHUNK_QTY):
    ln = ln + 1
    if docStr is not None:
        outFile.write(docStr)
    else:
        # print('Misformatted line %d ignoring:' % ln)
        # print(line.replace('\t', '<field delimiter>'))
        print('Ignoring misformatted line %d' % ln)

    if ln % REPORT_QTY == 0:
        print('Processed %d docs' % ln)

print('Processed %d docs' % ln)

inpFile.close()
outFile.close()
