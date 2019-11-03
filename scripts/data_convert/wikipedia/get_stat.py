#!/usr/bin/env python
import sys
import numpy as np
sys.path.append('scripts')
from data_convert.convert_common import *
from data_convert.text_proc import *
from data_convert.wikipedia import *
from pytorch_transformers.tokenization_bert import BertTokenizer

if len(sys.argv) != 2:
  print('Usage: <dump root dir>')
  sys.exit(1)

stopWords = [] # Currently, the list of stop-words is empty
textProc = SpacyTextParser(SPACY_MODEL, stopWords, sentSplit=True)

BERT_MODEL="bert-base-uncased"

sentQtys = []
tokQtys = []
bertPieceQtys = []

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

for fn in wikiExtractorFileIterator(sys.argv[1]):
  for wikiRec in SimpleXmlRecIterator(fn, 'doc'):
    pw = procWikipediaRecord(wikiRec)
    print(pw.id, pw.url, pw.title)
    wikiText = pw.content
    sentList = list(textProc(wikiText).sents)
    tQty = 0
    for s in sentList:
      tQty += len(s) # This includes non-word tokens, but it's ok

    sentQtys.append(len(sentList))
    tokQtys.append(tQty)
    bertToks = tokenizer.tokenize(wikiText)
    bertPieceQtys.append(len(bertToks))
  break


print('Sentence # distribution:  ', np.quantile(sentQtys, np.arange(1, 10)/10.0))
print('Token # distribution:     ', np.quantile(tokQtys, np.arange(1, 10)/10.0))
print('BERT token # distribution:', np.quantile(bertPieceQtys, np.arange(1, 10)/10.0))
