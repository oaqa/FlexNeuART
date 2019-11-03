#!/usr/bin/env python
import sys, os
import json
import argparse

#
# Convert a Wikipedia corpus previously processed by a wikiextractor
# https://github.com/attardi/wikiextractor
#

sys.path.append('scripts')
from data_convert.convert_common import *
from data_convert.text_proc import *
from data_convert.wikipedia import *
from pytorch_transformers.tokenization_bert import BertTokenizer

parser = argparse.ArgumentParser(description='Convert a Wikipedia corpus previously processed by a wikiextractor.')
parser.add_argument('--input_dir', metavar='input directory', help='input directory',
                    type=str, required=True)
parser.add_argument('--out_file', metavar='output file',
                    help='output JSON file',
                    type=str, required=True)
parser.add_argument('--bert_model', metavar='BERT model',
                    help='BERT model for tokenizer',
                    type=str, default='bert-base-uncased')
parser.add_argument('--bert_tok_qty', metavar='max # BERT toks.',
                    help='max # of BERT tokens in a piece.',
                    type=int, default=288)

args = parser.parse_args()
print(args)

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)
textProcessor = SpacyTextParser(SPACY_MODEL, stopWords, sentSplit=True, keepOnlyAlphaNum=True, lowerCase=True, enablePOS=False)

maxBertTokQty=args.bert_tok_qty
bertModel=args.bert_model

tokenizer = BertTokenizer.from_pretrained(bertModel, do_lower_case=True)

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]

def getSentText(wikiText, sentList, sentFirst, sentLast):
  return wikiText[sentList[sentFirst].start_char : sentList[sentLast].end_char + 1]

def writeOneDoc(outFile, pageId, passId, text):
  text_lemmas, text_unlemm = textProcessor.procText(text)

  doc = {DOCID_FIELD : '%s-%s' % (str(pageId), str(passId)),
         TEXT_FIELD_NAME : title_lemmas + ' '  + text_lemmas,
         TITLE_UNLEMM_FIELD_NAME : title_unlemm, 
         TEXT_UNLEMM_FIELD_NAME : text_unlemm,
         TEXT_RAW_FIELD_NAME : pw.title + ' ' + lastGoodSentText}
  docStr = json.dumps(doc) + '\n'
  outFile.write(docStr)

outFile = open(args.out_file, 'w')
seenIds = set()
for fn in wikiExtractorFileIterator(args.input_dir):
  for wikiRec in SimpleXmlRecIterator(fn, 'doc'):
    pw = procWikipediaRecord(wikiRec)
    print(pw.id, pw.url, pw.title)
    wikiText = pw.content
    origSentList = list(textProcessor(wikiText).sents)

    if pw.id in seenIds:
      print('Corrupt Wikipedia data, duplicate ID:', pw.id)
      sys.exit(1)

    title_lemmas, title_unlemm = textProcessor.procText(pw.title)

    currSent = 0

    passId = 0

    sentList = []
    for i in range(len(origSentList)):
      sentText = getSentText(wikiText, origSentList, i, i)
      if len(tokenizer.tokenize(sentText)) > maxBertTokQty:
        print('Found a sentence with more than %d BERT pieces will split into tokens before processing: %s' % (maxBertTokQty, str(sentList[currSent])))
        # Just treat each toekn
        sentList.extend(list(origSentList[i]))
      else:
        sentList.append(origSentList[i])

    while currSent < len(sentList):
      last = currSent 
      lastGoodSent = None
      lastGoodSentText = None
      while last < len(sentList):
        sentText = getSentText(wikiText, sentList, currSent, last)
        if len(tokenizer.tokenize(sentText)) <= maxBertTokQty:
          lastGoodSent = last
          lastGoodSentText = sentText
          last += 1
        else:
          break
      if lastGoodSent is None:
        print('Bug: should not be finding a sentence with more than %d BERT pieces at this point: %s' % (maxBertTokQty, str(sentList[currSent])))
        sys.exit(1)
      else:
        currSent = lastGoodSent + 1

        writeOneDoc(outFile, pw.id, passId, lastGoodSentText)

        passId += 1


outFile.close()

