#!/usr/bin/env python
import sys
sys.path.append('scripts')
from gen_exper_desc.common_gen_desc import *
from data_convert.convert_common import *

class ExtrJsonGEN:
  def __init__(self, fieldName):
    self.fieldName = fieldName

  def __call__(self):
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
                        "indexFieldName" : self.fieldName,
                        "similType" : "bm25",
                        "k1"        : k1str,
                        "b"         : bstr
                      }
                    }
                    ]
                    }

        testOnly=False
        yield fid, jsonDesc, testOnly

parser = BaseParser('BM25 tuning param generator')

genDescriptors(parser.args, ExtrJsonGEN(TEXT_FIELD_NAME), 'bm25tune.json', 'bm25tune')


