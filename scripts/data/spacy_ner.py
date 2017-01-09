#!/usr/bin/env python
import spacy
import sys
import io
import time
import json

if len(sys.argv) != 4:
  sys.stderr.write("Usage: <input JSON file> <output JSON file> <# of threads>\n") 
  sys.exit(1)

inpFileName=sys.argv[1]
outFileName=sys.argv[2]
threadQty=int(sys.argv[3])

def custom_pipeline(nlp):
  return (nlp.tagger,nlp.entity)

inpFile=io.open(inpFileName, 'r', encoding='utf-8', errors='replace')
outFile=io.open(outFileName, 'w', encoding='utf-8')

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
