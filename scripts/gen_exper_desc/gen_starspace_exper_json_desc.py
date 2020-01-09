#!/usr/bin/env python
import os, sys, json, re

sys.path.append('scritps/exper')

from optimBM25 import bestBM25

# Input location:
# inpRootDir / colName / starspace / <file names>

# Output location for descriptor files:
# Individual extractor files
# outRootDir / colName / starspace / <experiment-specific suffix>.json
# outRootDir / colName / starspace.desc

# Relative output location for experimental files:
# starspace / <experiment-specific suffix> 

embedRootDir = sys.argv[1]
colName      = sys.argv[2]
outRootDir   = sys.argv[3]

if not colName in bestBM25:
  print('No BM25 settings in the file scripts.exper.optimBM25 for collection: %s' % colName)
  sys.exit(1)

outDescDir = os.path.join(outRootDir, colName)
outJsonDir = os.path.join(outRootDir, colName, 'starspace')

if not os.path.exists(outDescDir):
  print('Directory does not exist: %s' % outDescDir)
  sys.exit(1)

if not os.path.exists(outJsonDir):
  print('Directory does not exist: %s' % outJsonDir)
  sys.exit(1)

embedDir = os.path.join(embedRootDir, colName, 'starspace') 

for isFusion in [0, 1]:
  with open(os.path.join(outDescDir, 'starspace_fusion=%d.desc' % isFusion), 'w') as of:
    for distType in ['l2', 'cosine']:
      lst = []
      for fn in os.listdir(embedDir):
        fns = re.sub(r"[^0-9]", " ", fn)
        sortKey = []
        for  s in fns.split():
          sortKey.append(int(s))
        lst.append( (tuple(sortKey), fn) )

      lst.sort()

      for _, fn in lst:
        if fn.endswith('.query'):
          fid0 = fn[0:-len('.query')]
          if isFusion:
            fid = distType + '_fusion_' + fid0
          else:
            fid = distType + '_embonly_' + fid0
          print(fid)
          extrList = [{
                        "type" : "avgWordEmbed",
                        "params" : {
                          "fieldName" : "text_unlemm",
                          "queryEmbedFile" : "starspace/%s.query" % fid0,
                          "docEmbedFile"   : "starspace/%s.answer" % fid0,
                          "useIDFWeight"   : "True",
                          "useL2Norm"      : "True",
                          "distType"       : distType
                        }
                      }
                      ]

          if isFusion:
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
          of.write('%s dev2 %s\n' % (jsonPath, os.path.join('starspace', fid)))

          with open(jsonPath, 'w') as f:
            json.dump(jsonDesc, f)


