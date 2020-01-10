#!/usr/bin/env python
import sys
sys.path.append('scripts')
from gen_exper_desc.common_gen_desc import *

def extrJsonGen():
  for bi in range(7):
    for k1i in range(7):
      b = 0.3 + 0.15 * bi
      k1 = 0.4 + 0.2 * k1i
      bstr = '%g' % b
      k1str = '%g' % k1
      fid = 'bm25_k1=%s_b=%s' % (k1str, bstr)

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

      yield fid, jsonDesc

parser = BaseParser('BM25 tuning param generator')

genDescriptors(parser.args, extrJsonGen, 'bm25tune.json', 'bm25tune')


