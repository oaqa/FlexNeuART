#!/usr/bin/env python
import os, sys, json

# Output location for descriptor files:
# Individual extractor files
# outRootDir / bm25tune / <experiment-specific suffix>.json
# outRootDir / bm25tune.desc

# Relative output location for experimental files:
# <experiment-specific suffix> 

outRootDir   = sys.argv[1]
outJsonDir = os.path.join(outRootDir, 'bm25tune')

if not os.path.exists(outRootDir):
  print('Directory does not exist: %s' % outRootDir)
  sys.exit(1)

if not os.path.exists(outJsonDir):
  print('Directory does not exist: %s' % outJsonDir)
  sys.exit(1)

outDescDir = outRootDir

with open(os.path.join(outDescDir, 'bm25tune.desc'), 'w') as of:
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
      jsonPath = os.path.join(outJsonDir, jsonFileName)
      of.write('%s dev1 %s\n' % (jsonPath, os.path.join('bm25tune', fid)))
        
      with open(jsonPath, 'w') as f:
        json.dump(jsonDesc, f)
      
