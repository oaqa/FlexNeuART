#!/usr/bin/env python

import sys
import json
sys.path.append('scripts')

from data_convert.convert_common_qa import *

def readingFunctionSquad(input):

  root = json.load(open(input))

  for page in root["data"]:
    for para in page["paragraphs"]:
      for qinfo in para["qas"]:
        answerList =[e["text"] for e in qinfo["answers"]]
        yield qinfo["id"], qinfo["question"], answerList


convertAndSaveQueries(readingFunctionSquad)