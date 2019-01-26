#!/usr/bin/env python
import os, sys, json, re

sys.path.append('scritps/exper')

from optimBM25 import bestBM25

# Output location for descriptor files:
# Individual extractor files
# outRootDir / colName / model1_addtune / <experiment-specific suffix>.json
# outRootDir / colName / model1_addtune.desc

# Relative output location for experimental files:
# model1_addtune / <experiment-specific suffix>

colName      = sys.argv[1]
outRootDir   = sys.argv[2]

if not colName in bestBM25:
  print('No BM25 settings in the file scripts.exper.optimBM25 for collection: %s' % colName)
  sys.exit(1)

confSubDir = 'model1_addtune'

outDescDir = os.path.join(outRootDir, colName)
outJsonDir = os.path.join(outRootDir, colName, confSubDir)

if not os.path.exists(outDescDir):
  print('Directory does not exist: %s' % outDescDir)
  sys.exit(1)

if not os.path.exists(outJsonDir):
  print('Directory does not exist: %s' % outJsonDir)
  sys.exit(1)

with open(os.path.join(outDescDir, '%s.desc' % confSubDir), 'w') as of:

  paramArr = []

  #for probSelfTran in [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5]:
  #  for lamb in [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5]:
  #    paramArr.append( (probSelfTran, lamb, 2.5e-3))

  # paramArr.append( (0.6, 0.05, 2.5e-3) )
  # paramArr.append( (0.7, 0.05, 2.5e-3) )
  # paramArr.append( (0.8, 0.05, 2.5e-3) )
  # paramArr.append( (0.9, 0.05, 2.5e-3) )
  #
  # paramArr.append((0.9, 0.01, 2.5e-3))
  # paramArr.append((0.9, 0.001, 2.5e-3))
  # paramArr.append((0.9, 0.0, 2.5e-3))

  paramArr.append((0.9, 0.0, 1e-3))
  paramArr.append((0.9, 0.0, 5e-4))
  paramArr.append((0.9, 0.0, 2.5e-4))
  paramArr.append((0.9, 0.0, 1e-4))

  for probSelfTran, lamb, minModel1Prob in paramArr:

      fid = 'bm25=text+model1=text_unlemm+lambda=%g+probSelfTran=%g+minTranProb=%g' % (lamb, probSelfTran, minModel1Prob)

      print(fid)
      extrList = [{
                    "type": "Model1Similarity",
                    "params": {
                      "fieldName": "text_unlemm",
                      "gizaIterQty": "5",
                      "probSelfTran": str(probSelfTran) + "f", # for float in Java
                      "lambda": lamb,
                      "minModel1Prob": minModel1Prob
                    }
                  }
                  ]

      extrList.append({
                      "type" : "TFIDFSimilarity",
                      "params" : {
                        "fieldName" : "text",
                        "similType" : "bm25",
                        "k1"        : bestBM25[colName]['k1'],
                        "b"         : bestBM25[colName]['b']
                      }
                    })

      jsonDesc = {"extractors" : extrList}
      jsonFileName = fid + '.json'
      jsonPath = os.path.join(outJsonDir, jsonFileName)
      of.write('%s dev2 %s\n' % (jsonPath, os.path.join(confSubDir, fid)))

      with open(jsonPath, 'w') as f:
        json.dump(jsonDesc, f)


