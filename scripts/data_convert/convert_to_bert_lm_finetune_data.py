#!/usr/bin/env python
# A generic script to generate data for finetuning BERT LM model from input JSONL file.
import sys
import argparse
import json

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import FileWrapper, replaceCharsNL
from scripts.config import SPACY_MODEL, TEXT_RAW_FIELD_NAME, REPORT_QTY

parser = argparse.ArgumentParser(description='Convert text BERT LM finetuning data file.')
parser.add_argument('--input', metavar='input JSON(L) file, can be compressed', help='input file',
                    type=str, required=True)
parser.add_argument('--output_pref', metavar='output file prefix', help='output file prefix',
                    type=str, required=True)
parser.add_argument('--max_set_size', metavar='max # of documents in a set',
                    default=1000_000,
                    help='the maximum number of set (in documents)',
                    type=int)
parser.add_argument('--lower_case', help='lowercase text',
                    action='store_true', default=False)

args = parser.parse_args()
print(args)

docQty = 0
setQty = 0
setId = 0

inpFile = FileWrapper(args.input)

nlp = SpacyTextParser(SPACY_MODEL, [], sentSplit=True)


def outFileName(pref, num):
    return pref + str(num) + '.txt'


print('Starting set 0')
outFile = FileWrapper(outFileName(args.output_pref, setId), 'w')

for line in inpFile:
    doc = json.loads(line)
    textRaw = doc[TEXT_RAW_FIELD_NAME]

    docSents = []

    for oneSent in nlp(textRaw).sents:
        oneSent = replaceCharsNL(str(oneSent)).strip()
        if args.lower_case:
            oneSent = oneSent.lower()
        if oneSent:
            docSents.append(oneSent)

    # Work hard to not write empty documents, b/c it'll upset the pregenerator
    if docSents:
        for oneSent in docSents:
            outFile.write(oneSent + '\n')
        outFile.write('\n')

    docQty += 1
    setQty += 1
    if docQty % REPORT_QTY == 0:
        print('Processed %d docs' % docQty)

    if setQty >= args.max_set_size:
        setQty = 0
        setId += 1
        print('Starting set %d' % setId)
        outFile.close()
        outFile = FileWrapper(outFileName(args.output_pref, setId), 'w')

print('Processed %d docs' % docQty)

inpFile.close()
outFile.close()
