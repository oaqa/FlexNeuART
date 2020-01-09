#!/usr/bin/env python
import os, sys, json
sys.path.append('scripts')
from gen_exper_desc.common_gen_desc import *

# Relative output location for descriptor files:

# <relative descriptor path> / bm25tune / <experiment-specific suffix>.json
# <relative descriptor path> / bm25tune.json

params =

outRootDir = sys.argv[1]
outDir = os.path.join(outRootDir, 'bm25tune')
relDescDir = sys.argv[2]

if not os.path.exists(outRootDir):
  print('Directory does not exist: %s' % outRootDir)
  sys.exit(1)

if not os.path.exists(outDir):
  print('Directory does not exist: %s' % outDir)
  sys.exit(1)

outDescDir = outRootDir

descDataJSON = []

for bi in range(7):
  for k1i in range(7):
    b = 0.3 + 0.15 * bi
    k1 = 0.4 + 0.2 * k1i
    bstr = '%g' % b
    k1str = '%g' % k1
    fid = 'bm25_k1=%s_b=%s' % (k1str, bstr)
    print(fid)
    jsonDesc = {
                "extractors" : [
                {
                  "type" : "TFIDFSimilarity",
                  "params" : {
                    "fieldName" : "text",
                    "similType" : "bm25",
                    "k1"        : k1str,
                    "b"         : bstr
                  }
                }
                ]
                }
    jsonFileName = fid + '.json'
    jsonExtrPath = os.path.join(outDir, jsonFileName)

    descDataJSON.append({EXPER_SUBDIR_PARAM : os.path.join(relDescDir, 'bm25tune', fid),
                         EXTR_TYPE_PARAM : jsonExtrPath})

    with open(jsonExtrPath, 'w') as f:
      json.dump(jsonDesc, f)


with open(os.path.join(outDescDir, 'bm25tune.json'), 'w') as of:
  json.dump(descDataJSON, of, indent=2)
      
