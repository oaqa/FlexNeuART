import os
import sys
import argparse
import json

# These parameter names must match parameter names in config.sh
EXTR_TYPE_PARAM="extrType"
EXPER_SUBDIR_PARAM="experSubdir"
TEST_ONLY_PARAM="testOnly"
MODEL_FINAL_PARAM="modelFinal"

FEAT_EXPER_SUBDIR="feat_exper"

REL_DESC_PATH_PARAM='rel_desc_path'
OUT_DIR_PARAM='outdir'

class BaseParser:
  def initAddArgs(self):
    pass

  def __init__(self, progName):
    self.parser = argparse.ArgumentParser(description=progName)
    self.parser.add_argument('--' + OUT_DIR_PARAM, metavar='output directory',
                        help='output directory',
                        type=str, required=True)
    self.parser.add_argument('--' + REL_DESC_PATH_PARAM, metavar='relative descriptor path',
                        help='relative descriptor path',
                        type=str, required=True)
    self.parser.add_argument('--exper_subdir', metavar='exper. results subdir.',
                        help='top-level sub-directory to store experimental results',
                        type=str, default=FEAT_EXPER_SUBDIR)
    self.initAddArgs()

  def getArgs(self):
    """
    :return: argument objects, to be used
    """
    return self.args

  def parseArgs(self):
    """This is deliberately implemented with a delayed optimization,
    so that a user can add new parameter definitions before arguments
    are parsed.
    """
    self.args = self.parser.parse_args()
    print(self.args)


def genRerankDescriptors(args, extrJsonGenFunc, jsonDescName, jsonSubDir):
  """
  A generic function to write a bunch of experimental descrptors (for the re-ranking only scenario).

  :param args:              arguments previously produce by the class inherited from BaseParser
  :param extrJsonGenFunc:   generator of extractor JSON and its file ID.
  :param jsonDescName:      the name of the top-level descriptor file that reference individual extractor JSONs.
  :param jsonSubDir:        a sub-directory to store extractor JSONs.

  """
  descDataJSON = []

  args_var = vars(args)

  outJsonSubDir = os.path.join(args_var[OUT_DIR_PARAM], jsonSubDir)
  if not os.path.exists(outJsonSubDir):
    os.makedirs(outJsonSubDir)

  for fileId, jsonDesc, testOnly, modelFinal in extrJsonGenFunc():
    jsonFileName = fileId + '.json'

    desc = {EXPER_SUBDIR_PARAM: os.path.join(args.exper_subdir, fileId),
            EXTR_TYPE_PARAM: os.path.join(args_var[REL_DESC_PATH_PARAM], jsonSubDir, jsonFileName),
            TEST_ONLY_PARAM: int(testOnly)}
    if modelFinal is not None:
      desc[MODEL_FINAL_PARAM] = modelFinal

    descDataJSON.append(desc)

    with open(os.path.join(outJsonSubDir, jsonFileName), 'w') as of:
      json.dump(jsonDesc, of, indent=2)

  with open(os.path.join(args.outdir, jsonDescName), 'w') as of:
    json.dump(descDataJSON, of, indent=2)




