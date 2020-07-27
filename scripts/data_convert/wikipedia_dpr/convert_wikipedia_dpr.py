#!/usr/bin/env python
import sys, os
import json
import argparse
import multiprocessing

#
# Convert a Wikipedia corpus used in the Facebook DPR project:
# https://github.com/facebookresearch/DPR/tree/master/data
#
# The input is a TAB-separated file with three columns: id, passage text, title
# This conversion script preserves original passages, but it also tokenizes them.
#

sys.path.append('.')

from scripts.data_convert.convert_common import readStopWords, FileWrapper, addRetokenizedField, \
                                                BERT_TOK_OPT, BERT_TOK_OPT_HELP
from scripts.data_convert.text_proc import SpacyTextParser
from scripts.config import STOPWORD_FILE, BERT_BASE_MODEL, SPACY_MODEL, \
                        DOCID_FIELD, TEXT_RAW_FIELD_NAME, \
                        REPORT_QTY, IMAP_PROC_CHUNK_QTY, \
                        TEXT_BERT_TOKENIZED_NAME,\
                        TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME
from pytorch_pretrained_bert import BertTokenizer

parser = argparse.ArgumentParser(description='Convert a Wikipedia corpus downloaded from github.com/facebookresearch/DPR.')
parser.add_argument('--input_file', metavar='input file', help='input directory',
                    type=str, required=True)
parser.add_argument('--out_file', metavar='output file',
                    help='output JSONL file',
                    type=str, required=True)
# Default is: Number of cores minus one for the spaning process
parser.add_argument('--proc_qty', metavar='# of processes', help='# of NLP processes to span',
                    type=int, default=multiprocessing.cpu_count() - 1)
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

args = parser.parse_args()
arg_vars = vars(args)
print(args)

bertTokenizer=None
if BERT_TOK_OPT in arg_vars:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bertTokenizer = BertTokenizer.from_pretrained(BERT_BASE_MODEL)

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)

# Lower cased
stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]


class PassParseWorker:
    def __init__(self, stopWords, spacyModel):
        # Lower cased
        self.textProcessor = SpacyTextParser(spacyModel, stopWords,
                                             keepOnlyAlphaNum=True, lowerCase=True,
                                             enablePOS=True)
    def __call__(self, line):

        if not line:
            return None

        line = line.strip()
        if not line:
            return None

        fields = line.split('\t')
        if ' '.join(fields) == 'id text title':
            return ''

        assert len(fields) == 3, f"Wrong format fline: {line}"
        passId, rawText, title = fields

        textLemmas, textUnlemm = self.textProcessor.procText(rawText)
        titleLemmas, titleUnlemm = self.textProcessor.procText(title)

        doc = {DOCID_FIELD: passId,
               TEXT_FIELD_NAME: titleLemmas + ' ' + textLemmas,
               TITLE_UNLEMM_FIELD_NAME: titleUnlemm,
               TEXT_UNLEMM_FIELD_NAME: textUnlemm,
               TEXT_RAW_FIELD_NAME: titleUnlemm + ' ' + rawText.lower()}

        addRetokenizedField(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bertTokenizer)
        return json.dumps(doc)


inpFile = FileWrapper(args.input_file)
outFile = FileWrapper(args.out_file, 'w')

proc_qty = args.proc_qty
print(f'Spanning {proc_qty} processes')
pool = multiprocessing.Pool(processes=proc_qty)
ln = 0
for docStr in pool.imap(PassParseWorker(stopWords, SPACY_MODEL), inpFile, IMAP_PROC_CHUNK_QTY):
    ln = ln + 1
    if docStr is not None:
        if docStr:
            outFile.write(docStr + '\n')
    else:
        print('Ignoring misformatted line %d' % ln)

    if ln % REPORT_QTY == 0:
        print('Processed %d passages' % ln)

print('Processed %d passages' % ln)

inpFile.close()
outFile.close()

