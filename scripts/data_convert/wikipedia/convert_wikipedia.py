#!/usr/bin/env python
import sys, os
import json
import argparse
from multiprocessing import Pool

#
# Convert a Wikipedia corpus previously processed by a wikiextractor (https://github.com/attardi/wikiextractor)
# It tries to limit each chunk to have at most given number of BERT tokens.
# However, for efficiency reasons, this may be not a bullet-proof (and
# a bit approximate computation), because we split into sentences
# and the number of BERT tokens for each sentence separately.
# Indeed, results differ a bit when we try to re-tokenize each sequence
#

sys.path.append('.')

from scripts.data_convert.convert_common import readStopWords, SimpleXmlRecIterator, \
                                                procWikipediaRecord, wikiExtractorFileIterator
from scripts.data_convert.text_proc import SpacyTextParser
from scripts.config import STOPWORD_FILE, BERT_BASE_MODEL, SPACY_MODEL, \
                        DOCID_FIELD, TEXT_RAW_FIELD_NAME, TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME
from pytorch_pretrained_bert import BertTokenizer

parser = argparse.ArgumentParser(description='Convert a Wikipedia corpus previously processed by a wikiextractor.')
parser.add_argument('--input_dir', metavar='input directory', help='input directory',
                    type=str, required=True)
parser.add_argument('--out_file', metavar='output file',
                    help='output JSON file',
                    type=str, required=True)
parser.add_argument('--temp_file_pref', metavar='temp file prefix',
                    help='A prefix for intermediate temporary files that are merged in the end',
                    required=True)
parser.add_argument('--bert_tok_qty', metavar='max # BERT toks.',
                    help='max # of BERT tokens in a piece.',
                    type=int, default=288)
parser.add_argument('--proc_qty', metavar='# of processes',
                    help='# of parallel processes',
                    type=int, required=True)

args = parser.parse_args()
print(args)

# Lower cased
stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)
# Lower cased
textProcessor = SpacyTextParser(SPACY_MODEL, stopWords, sentSplit=True, keepOnlyAlphaNum=True, lowerCase=True,
                                enablePOS=False)

maxBertTokQty = args.bert_tok_qty

tokenizer = BertTokenizer.from_pretrained(BERT_BASE_MODEL, do_lower_case=True)

tempFilePref = args.temp_file_pref
procQty = args.proc_qty

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]


class FakeSentence:
    def __repr__(self):
        return '[%s,%d]' % (self.start_char, self.end_char)

    def __init__(self, s, e):
        self.start_char = s
        self.end_char = e


def getSentText(wikiText, sentList, sentFirst, sentLast):
    return wikiText[sentList[sentFirst].start_char: sentList[sentLast].end_char + 1]


def writeOneDoc(outFile, pageId, passId, titleLemmas, titleUnlemm, rawText):
    textLemmas, textUnlemm = textProcessor.procText(rawText)

    doc = {DOCID_FIELD: '%s-%s' % (str(pageId), str(passId)),
           TEXT_FIELD_NAME: titleLemmas + ' ' + textLemmas,
           TITLE_UNLEMM_FIELD_NAME: titleUnlemm,
           TEXT_UNLEMM_FIELD_NAME: textUnlemm,
           TEXT_RAW_FIELD_NAME: titleUnlemm + ' ' + rawText.lower()}
    docStr = json.dumps(doc) + '\n'
    outFile.write(docStr)


def procOneFile(e):
    wikiFileName, outFileName = e
    print(wikiFileName, outFileName)
    # Open for append, should be quite fast compared to all the other processing
    outFile = open(outFileName, 'a')

    for wikiRec in SimpleXmlRecIterator(wikiFileName, 'doc'):

        pw = procWikipediaRecord(wikiRec)
        print(pw.id, pw.url, pw.title)
        wikiText = pw.content
        origSentList = list(textProcessor(wikiText).sents)

        titleLemmas, titleUnlemm = textProcessor.procText(pw.title)

        currSent = 0

        passId = 0

        sentList = []
        for i in range(len(origSentList)):
            sentText = getSentText(wikiText, origSentList, i, i)
            toks = tokenizer.tokenize(sentText)
            if len(toks) > maxBertTokQty:
                longSent = origSentList[i]
                print('Tokenized version has %d tokens' % len(toks), toks)
                print(
                    'Found a sentence (%d:%d) with more than %d BERT pieces will split into tokens before processing: %s' %
                    (longSent.start_char, longSent.end_char, maxBertTokQty, str(longSent)))
                # Just treat each token
                sentTokList = []
                for tok in longSent:
                    # Spacy documentation claims that idx is an offset within a parent document,
                    # which apparently means an absolute offset, not offset within a sentence: https://spacy.io/api/token
                    sentTokList.append(FakeSentence(tok.idx, tok.idx + len(tok)))
                print('Long sentence split into tokens:', sentTokList)
                sentList.extend(sentTokList)
            else:
                sentList.append(origSentList[i])

        while currSent < len(sentList):
            last = currSent
            lastGoodSent = None
            tokQty = 0
            while last < len(sentList):
                sentText = getSentText(wikiText, sentList, last, last)
                # We assume that each sentence can be tokenized independently by the BERT tokenizer
                # I am not sure it's always true, but it's probably an good assumption with respect
                # to limiting each sequence to a given number of BERT tokens.
                # In a downstream task, whenever this assumption fails, we can always truncate
                # the sequence of tokens.
                tokQty += len(tokenizer.tokenize(sentText))
                if tokQty <= maxBertTokQty:
                    lastGoodSent = last
                    last += 1
                else:
                    break
            if lastGoodSent is None:
                print(
                    'Bug or weirdly long Wikipedia token: we should not be finding a text piece with more than %d BERT pieces at this point: %s' % (
                    maxBertTokQty, str(sentList[currSent])))
                currSent += 1
            else:
                # Must obtain sentence text, *BEFORE* currSent is updated!
                sentText = getSentText(wikiText, sentList, currSent, lastGoodSent)
                currSent = lastGoodSent + 1
                writeOneDoc(outFile, pw.id, passId,
                            titleLemmas=titleLemmas, titleUnlemm=titleUnlemm,
                            rawText=sentText)

                passId += 1

    outFile.close()


wikiFileList = []

currOutId = 0

tmpOutFileNameSet = set()

for wikiFileName in wikiExtractorFileIterator(args.input_dir):
    tmpOutFileName = f'{tempFilePref}.{currOutId}'

    if tmpOutFileName not in tmpOutFileNameSet:
        tmpOutFileNameSet.add(tmpOutFileName)
        # Create an empty output file
        with open(tmpOutFileName, 'w'):
            pass

    wikiFileList.append((wikiFileName, tmpOutFileName))
    currOutId = (currOutId + 1) % procQty

    # break

procPool = Pool(procQty)
procPool.map(procOneFile, wikiFileList)

print('Sub-processes finished, let us merge results now!')

seenIds = set()

with open(args.out_file, 'w') as outFile:
    for tmpOutFileName in tmpOutFileNameSet:
        print('Processing:', tmpOutFileName)
        for docStr in open(tmpOutFileName):
            doc = json.loads(docStr)
            docId = doc[DOCID_FIELD]
            if docId in seenIds:
                print('Bug or corrupt data, duplicate id: ', docId)
                sys.exit(1)
            seenIds.add(docId)
            outFile.write(docStr)

        os.unlink(tmpOutFileName)
