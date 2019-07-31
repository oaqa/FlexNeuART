#!/usr/bin/env python3
import sys
import gzip
import json
import argparse
sys.path.append('.')
from scripts.data_convert.text_proc import *
from scripts.data_convert.convert_common import *

parser = argparse.ArgumentParser(description='Convert MSMARCO-adhoc queries.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)


args = parser.parse_args()
print(args)

inpFile = FileWrapper(args.input)
outFile = FileWrapper(args.output, 'w')

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)
nlp = SpacyTextParser("en_core_web_sm", stopWords, keepOnlyAlphaNum=True)

# Input file is a TSV file
ln=0
for line in inpFile:
  ln+=1
  line = line.strip()
  if not line: 
    continue
  fields = line.split('\t')
  if len(fields) != 2:
    print('Misformated line %d ignoring:' % ln)
    print(line.replace('\t', '<field delimiter>'))
    continue

  did, query = fields

  query_lemmas, query_unlemm = nlp.procText(query)
  query_lemmas = query_lemmas.lower().strip()
  query_unlemm = query_unlemm.lower().strip()

  doc = {'DOCNO' : did, 'text' : query_lemmas, 'text_unlemm' : query_unlemm}
  docStr = json.dumps(doc) + '\n'
  outFile.write(docStr)
  if ln % REPORT_QTY == 0:
    print('Processed %d queries' % ln)

print('Processed %d queries' % ln)

inpFile.close()
outFile.close()
