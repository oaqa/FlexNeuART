#!/usr/bin/env python
import spacy
import sys
import time
import json
import codecs
import gzip
import io

if len(sys.argv) != 4:
  sys.stderr.write("Usage: <gzipped input JSON file> <gzipped output JSON file> <# of threads>\n") 
  sys.exit(1)

inpFileName=sys.argv[1]
outFileName=sys.argv[2]
threadQty=int(sys.argv[3])

def custom_pipeline(nlp):
  return (nlp.tagger,nlp.entity)

UTF8Reader = codecs.getreader('utf8')
UTF8Writer = codecs.getwriter('utf8')

#inpFile=io.open(inpFileName, 'r', encoding='utf-8', errors='replace')
#inpFile = UTF8Reader(gzip.open(inpFileName, 'r'))
# Really beats me why gzip.open shouldn't be passed to UTF8Reader, but it doesn't work this way
inpFile = gzip.open(inpFileName, 'r')
#outFile=io.open(outFileName, 'w', encoding='utf-8')
outFile = UTF8Writer(gzip.open(outFileName, 'w'))

nlp = spacy.load('en', create_pipeline=custom_pipeline)

# Each line has a single JSON piece
for line in inpFile: 
  data = json.loads(line)
  for p in data['passages']:
    passId=p['id']
    passText=p['text']
    res = { 'id' : passId }
    spacyInp = [passText]
    annotList = []
    for doc in nlp.pipe(spacyInp, batch_size=50, n_threads=threadQty):
      for ent in doc.ents:
        annotList.append({'type':'ner', 'label':ent.label_, 'start':ent.start_char, 'end':ent.end_char})
    res[ 'annotations' ] = annotList
    jres=json.dumps(res).replace('\n', ' ')
    outFile.write(jres+u'\n')

outFile.close()
