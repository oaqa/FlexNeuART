#!/usr/bin/env python3
import sys, json
f=open(sys.argv[1])
doc=json.load(f)
f.close()
for e in doc['extractors']:
  if 'params' in e:
    prm = e['params']
    if 'fieldName' in prm: 
      prm['indexFieldName'] = prm['fieldName']
      del prm['fieldName']

print(doc)
with open(sys.argv[1], 'w') as fo:
  json.dump(doc, fo)
    
