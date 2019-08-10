#!/usr/bin/env python3
import sys
import gzip
import json
import argparse
sys.path.append('.')
from scripts.data_convert.text_proc import *
from scripts.data_convert.convert_common import *

parser = argparse.ArgumentParser(description='Convert MSMARCO-adhoc documents or passage to BERT LM finetuning data file.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
parser.add_argument('--max_doc_size', metavar='max doc size bytes', help='the threshold for the document size, if a document is larger it is truncated',
                    type=int, default=MAX_DOC_SIZE)
parser.add_argument('--lower_case', metavar='lowercase', help='lowercase text',
                    type=bool, default=False)


args = parser.parse_args()
print(args)

inpFile = FileWrapper(args.input)
outFile = FileWrapper(args.output, 'w')
maxDocSize = args.max_doc_size

nlp = SpacyTextParser(SPACY_MODEL, [])

# Input file is a TSV file
ln=0
for line in inpFile:
  ln+=1
  if not line: 
    continue
  line = line[:maxDocSize] # cut documents that are too long!
  fields = line.split('\t')
  body = fields[-1] # passage or document, body is always the last field

  for oneSent in nlp(body).sents:
    oneSent = str(oneSent)
    if args.lower_case:
      oneSent = oneSent.lower()

  outFile.write('\n')
  if ln % REPORT_QTY == 0:
    print('Processed %d docs' % ln)

print('Processed %d docs' % ln)

inpFile.close()
outFile.close()
