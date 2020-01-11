#!/usr/bin/env python
import os, sys, json, re

sys.path.append('scripts')

from data_convert.convert_common import *
from gen_exper_desc.common_gen_desc import *

class ExtrJsonGEN:

  def __call__(self):
    testOnly=False
    for fid, extrType in self.paramConf:
      yield fid, extrType, testOnly

  def __init__(self, k1, b, textFieldName=TEXT_FIELD_NAME, textUlemmFieldName=TEXT_UNLEMM_FIELD_NAME):
    self.k1 = k1
    self.b = b

    self.paramConf = []

    paramArr = []

    for probSelfTran in [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5]:
      for lamb in [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5]:
        paramArr.append( (probSelfTran, lamb))

    paramArr.append( (0.6, 0.05) )
    paramArr.append( (0.7, 0.05) )
    paramArr.append( (0.8, 0.05) )
    paramArr.append( (0.9, 0.05) )

    paramArr.append((0.9, 0.01))
    paramArr.append((0.9, 0.001))
    paramArr.append((0.9, 0.0))

    for probSelfTran, lamb in paramArr:

      fid = f'bm25={textFieldName}+model1={textUlemmFieldName}+lambda=%g+probSelfTran=%g' % (lamb, probSelfTran)

      extrList = [{
                    "type": "Model1Similarity",
                    "params": {
                      "indexFieldName": textUlemmFieldName,
                      "gizaIterQty"   : "5",
                      "probSelfTran"  : probSelfTran,
                      "lambda"        : lamb,
                      "minModel1Prob" : "2.5e-3f"
                    }
                  }
                  ]

      extrList.append({
                      "type" : "TFIDFSimilarity",
                      "params" : {
                        "indexFieldName"  : textFieldName,
                        "similType"       : "bm25",
                        "k1"              : self.k1,
                        "b"               : self.b
                      }
                    })

      self.paramConf.append( (fid, {"extractors": extrList}) )

    paramArr = []
    paramArr.append((0.9, 0.0, 1e-3))
    paramArr.append((0.9, 0.0, 5e-4))
    paramArr.append((0.9, 0.0, 2.5e-4))
    paramArr.append((0.9, 0.0, 1e-4))

    for probSelfTran, lamb, minModel1Prob in paramArr:
      fid = f'bm25={textFieldName}+model1={textUlemmFieldName}+lambda=%g+probSelfTran=%g+minTranProb=%g' % (
            lamb, probSelfTran, minModel1Prob)

      extrList = [{
        "type": "Model1Similarity",
        "params": {
          "indexFieldName": textUlemmFieldName,
          "gizaIterQty": "5",
          "probSelfTran": str(probSelfTran) + "f",  # for float in Java
          "lambda": lamb,
          "minModel1Prob": minModel1Prob
        }
      }
      ]

      extrList.append({
        "type": "TFIDFSimilarity",
        "params": {
          "indexFieldName": textUlemmFieldName,
          "similType": "bm25",
          "k1": self.k1,
          "b": self.b
        }
      })

      self.paramConf.append( (fid, {"extractors": extrList}) )


parser = ParserWithBM25Coeff('Model1 tuning param generator')
args = parser.args
genDescriptors(args, ExtrJsonGEN(k1=args.k1, b=args.b), 'model1tune.json', 'model1tune')




