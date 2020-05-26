#!/usr/bin/env python
import sys
import os
import shutil

sys.path.append('scripts')

from gen_exper_desc.common_gen_desc import *
from data_convert.convert_common import *

MODEL_SRC_PATH='scripts/exper/sample_exper_desc/one_feat.model'
MODEL_DST_REL_PATH='models'
MODEL_DST_NAME = 'one_feat.model'

class ParserBM25Coeff(BaseParser):
  def initAddArgs(self):
    self.parser.add_argument('--index_field_name',
                             metavar='BM25 index field name',
                             help='an index field for BM25 score', required=True)
    self.parser.add_argument('--query_field_name',
                             metavar='BM25 query field name',
                             help='an query field for BM25 score', default=None)

parser = ParserBM25Coeff('BM25 tuning param generator')
parser.parseArgs()

args = parser.getArgs()
args_var = vars(args)

indexFieldName = args.index_field_name
queryFieldName = args.query_field_name
outdir = args_var[OUT_DIR_PARAM]
outModelDir = os.path.join(outdir, MODEL_DST_REL_PATH)
if not os.path.exists(outModelDir):
  os.makedirs(outModelDir)
shutil.copyfile(MODEL_SRC_PATH, os.path.join(outModelDir, MODEL_DST_NAME))

modelRelName = os.path.join(args_var[REL_DESC_PATH_PARAM], MODEL_DST_REL_PATH, MODEL_DST_NAME)

fieldNameDesc = indexFieldName
if queryFieldName is not None:
    fileNameDesc = '%s_%s' % (queryFieldName, indexFieldName)

class ExtrBM25JsonGEN:
  def __init__(self, indexFieldName, queryFieldName):
    self.indexFieldName = indexFieldName
    self.queryFieldName = queryFieldName
    if self.queryFieldName is None:
        self.queryFieldName = self.indexFieldName


  def __call__(self):

    for bi in range(7):
      for k1i in range(7):
        b = 0.3 + 0.15 * bi
        k1 = 0.4 + 0.2 * k1i
        bstr = '%g' % b
        k1str = '%g' % k1
        fid = 'bm25tune_%s_k1=%s_b=%s' % (fieldNameDesc, k1str, bstr)

        jsonDesc = {
                    "extractors" : [
                    {
                      "type" : "TFIDFSimilarity",
                      "params" : {
                        "indexFieldName" : self.indexFieldName,
                        "queryFieldName": self.queryFieldName,
                        "similType" : "bm25",
                        "k1"        : k1str,
                        "b"         : bstr
                      }
                    }
                    ]
                    }

        # Test only is true, b/c there's nothing to train, but we need to provide the model
        testOnly=True
        yield fid, jsonDesc, testOnly, modelRelName 

prefix = f'bm25tune_{fileNameDesc}'
genRerankDescriptors(args, ExtrBM25JsonGEN(indexFieldName, queryFieldName),
                     f'{prefix}.json', prefix)


