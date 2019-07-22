#!/usr/bin/env python3
import sys
import gzip
import json
sys.path.append('.')
from text_proc import *

if len(sys.argv) != 3:
  print('Usage: <input: can be in gz format> <output: can be in gz format>')
  sys.exit(1)

inpFileName = sys.argv[1]
outFileName = sys.argv[2]

# TODO need some common file to store constants and common functions like these open functions and stop word reading functions
inpFile = gzip.open(inpFileName) if inpFileName.endwith('.gz') else open(inpFileName)
outFile = gzip.open(outFilename, 'w') if outFileName.endwith('.gz') else open(outFileName, 'w')

stopWords = []
with open('data/stopwords.txt') as f:
  for w in f:
    w = w.strip()
    if w:
      stopWords.append(w)

nlp = SpacyTextParser("en_core_web_sm", stopWords)

# Input file is a TSV file
for line in inpFile:
  line = line.strip()
  if not line: 
    continue
  did, url, title, body = line.split('\t')

  title_lemmas, title_unlemm = nlp.procText(title)
  body_lemmas, body_unlemm = nlp.procText(body)

  text = title_lemmas + ' ' + body_lemmas
  text = text.strip()
  doc = {'DOCNO' : did, 'text' : text, 'title' : title_unlemm, 'body' : body_unlemm}
  docStr = json.dumps(doc)
  outFile.write(docStr + '\n')

inpFile.close()
outFile.close()
