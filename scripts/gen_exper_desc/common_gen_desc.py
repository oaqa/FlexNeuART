import sys
import argparse
from collections import namedtuple

TEST_SET_PARAM="testSet"
EXPER_SUBDIR_PARAM="experSubdir"
EXTR_TYPE_PARAM="extrType"

ExperDescGenParams=namedtuple('ExperDescGenParams', ['outputPath', 'relDescPath'])

def parseArgs(progName):
  """This function parsers arguments for experiment-generating scripts.
  It is used to obtain the output path as well as the relative descriptor path.
  These can be different, b/c, e.g., output path can be a full path. Relative path
  is a path relative to the collection root.

  :param  progName    name of the program
  :return a named tuple with parameters

  """

  parser = argparse.ArgumentParser(description='Convert MSMARCO-adhoc documents.')
  parser.add_argument('--outdir', metavar='output directory',
                      help='output directory',
                      type=str, required=True)
  parser.add_argument('--rel_desc_path', metavar='relative descriptor path',
                      help='relative descriptor path',
                      type=str, required=True)

  args = parser.parse_args()
  print(args)

  return ExperDescGenParams(outputPath=args.outdir, relDescPath=args.rel_desc_path)