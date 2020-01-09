import os
import sys
import argparse
import json

TEST_SET_PARAM="testSet"
EXPER_SUBDIR_PARAM="experSubdir"
EXTR_TYPE_PARAM="extrType"

class BaseParser:
  def initAddArgs(self):
    pass

  def __init__(self, progName):
    self.parser = argparse.ArgumentParser(description=progName)
    self.parser.add_argument('--outdir', metavar='output directory',
                        help='output directory',
                        type=str, required=True)
    self.parser.add_argument('--rel_desc_path', metavar='relative descriptor path',
                        help='relative descriptor path',
                        type=str, required=True)
    self.initAddArgs()

    self.args = self.parser.parse_args()
    print(self.args)


def genDescriptors(args, extrJsonGenFunc, jsonDescName, jsonSubDir):
  """
  A generic function to write a bunch of experimental descrptors.

  :param args:              arguments previously produce by the class inherited from BaseParser
  :param extrJsonGenFunc:   generator of extractor JSON and its file ID.
  :param jsonDescName:      the name of the top-level descriptor file that reference individual extractor JSONs.
  :param jsonSubDir:        a sub-directory to store extractor JSONs.

  """
  descDataJSON = []

  outJsonSubDir = os.path.join(args.outdir, jsonSubDir)
  if not os.path.exists(outJsonSubDir):
    os.makedirs(outJsonSubDir)

  for fileId, jsonDesc in extrJsonGenFunc():
    jsonFileName = fileId + '.json'

    descDataJSON.append({EXPER_SUBDIR_PARAM: os.path.join(jsonSubDir, fileId),
                         EXTR_TYPE_PARAM: os.path.join(args.rel_desc_path, jsonSubDir, jsonFileName)})

    with open(os.path.join(outJsonSubDir, jsonFileName), 'w') as of:
      json.dump(jsonDesc, of, indent=2)

  with open(os.path.join(args.outdir, jsonDescName), 'w') as of:
    json.dump(descDataJSON, of, indent=2)




