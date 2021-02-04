import gzip, bz2
import collections
import re
import os
import sys
import json
import urllib
from bs4 import BeautifulSoup

from scripts.config import DEFAULT_ENCODING, STOPWORD_FILE, DOCID_FIELD

WikipediaRecordParsed = collections.namedtuple('WikipediaRecordParsed',
                                               'id url title content')

YahooAnswerRecParsed = collections.namedtuple('YahooAnswerRecParsed',
                                              'uri subject content bestAnswerId answerList')

MAX_NUM_QUERY_OPT = 'max_num_query'
MAX_NUM_QUERY_OPT_HELP = 'maximum # of queries to generate'
BERT_TOK_OPT = 'bert_tokenize'
BERT_TOK_OPT_HELP = 'Apply the BERT tokenizer and store result in a separate field'
OUT_BITEXT_PATH_OPT = 'out_bitext_path'
OUT_BITEXT_PATH_OPT_META = 'optional bitext path'
OUT_BITEXT_PATH_OPT_HELP = 'An optional output directory to store bitext'

# Replace \n and \r characters with spaces
def replaceCharsNL(s):
    return re.sub(r'[\n\r]', ' ', s)


class FileWrapper:

    def __enter__(self):
        return self

    def __init__(self, fileName, flags='r'):
        """Constructor, which opens a regular or gzipped-file

          :param  fileName a name of the file, it has a '.gz' or '.bz2' extension, we open a compressed stream.
          :param  flags    open flags such as 'r' or 'w'
        """
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        if fileName.endswith('.gz'):
            self._file = gzip.open(fileName, flags)
            self._isCompr = True
        elif fileName.endswith('.bz2'):
            self._file = bz2.open(fileName, flags)
            self._isCompr = True
        else:
            self._file = open(fileName, flags)
            self._isCompr = False

    def write(self, s):
        if self._isCompr:
            self._file.write(s.encode())
        else:
            self._file.write(s)

    def read(self, s):
        if self._isCompr:
            return self._file.read().decode()
        else:
            return self._file.read()

    def close(self):
        self._file.close()

    def __exit__(self, type, value, tb):
        self._file.close()

    def __iter__(self):
        for line in self._file:
            yield line.decode() if self._isCompr else line


def readStopWords(fileName=STOPWORD_FILE, lowerCase=True):
    """Reads a list of stopwords from a file. By default the words
       are read from a standard repo location and are lowercased.

      :param fileName a stopword file name
      :param lowerCase  a boolean flag indicating if lowercasing is needed.

      :return a list of stopwords
    """
    stopWords = []
    with open(fileName) as f:
        for w in f:
            w = w.strip()
            if w:
                if lowerCase:
                    w = w.lower()
                stopWords.append(w)

    return stopWords


def SimpleXmlRecIterator(fileName, recTagName):
    """A simple class to read XML records stored in a way similar to
      the Yahoo Answers collection. In this format, each record
      occupies a certain number of lines, but no record "share" the same
      line. The format may not be fully proper XML, but each individual
      record may be. It always starts with a given tag name ends with
      the same tag, e.g.,:

      <recordTagName ...>
      </recordTagName>

    :param fileName:  input file name (can be compressed).
    :param tagName:   a record tag name (for the tag that encloses the record)

    :return:   it yields a series of records
    """

    with FileWrapper(fileName) as f:

        recLines = []

        startEntry = '<' + recTagName
        endEntry = '</' + recTagName + '>'

        seenEnd = True
        seenStart = False

        ln = 0
        for line in f:
            ln += 1
            if not seenStart:
                if line.strip() == '':
                    continue  # Can skip empty lines
                if line.startswith(startEntry):
                    if not seenEnd:
                        raise Exception(f'Invalid format, no previous end tag, line {ln} file {fileName}')
                    assert (not recLines)
                    recLines.append(line)
                    seenEnd = False
                    seenStart = True
                else:
                    raise Exception(f'Invalid format, no previous start tag, line {ln} file {fileName}')
            else:
                recLines.append(line)
                noSpaceLine = line.replace(' ', '').strip()  # End tags may contain spaces
                if noSpaceLine.endswith(endEntry):
                    if not seenStart:
                        raise Exception(f'Invalid format, no previous start tag, line {ln} file {fileName}')
                    yield ''.join(recLines)
                    recLines = []
                    seenEnd = True
                    seenStart = False

        if recLines:
            raise Exception(f'Invalid trailing entries in the file {fileName} %d entries left' % (len(recLines)))


def removeTags(str):
    """Just remove anything that looks like a tag"""
    return re.sub(r'</?[a-z]+\s*/?>', '', str)


def pretokenizeUrl(url):
    """A hacky procedure to "pretokenize" URLs.

    :param  url:  an input URL
    :return a URL with prefixes (see below) removed and some characters replaced with ' '
    """
    remove_pref = ['http://', 'https://', 'www.']
    url = urllib.parse.unquote(url)
    changed = True
    while changed:
        changed = False
        for p in remove_pref:
            assert len(p) > 0
            if url.startswith(p):
                changed = True
                url = url[len(p):]
                break

    return re.sub(r'[.,:!\?/"+\-\'=_{}()|]', " ", url)


def wikiExtractorFileIterator(rootDir):
    """Iterate over all files produced by the wikiextractor and return file names.
    """
    dirList1Sorted = list(os.listdir(rootDir))
    dirList1Sorted.sort()
    for dn in dirList1Sorted:
        fullDirName = os.path.join(rootDir, dn)
        if os.path.isdir(fullDirName):
            dirList2Sorted = list(os.listdir(fullDirName))
            dirList2Sorted.sort()
            for fn in dirList2Sorted:
                if fn.startswith('wiki_'):
                    yield os.path.join(fullDirName, fn)


def procWikipediaRecord(recStr):
    """A procedure to parse a single Wikipedia page record
       from the wikiextractor output, which we assume it to have DEFAULT_ENCODING encoding.

    :param recStr:  One page content including encosing tags <doc> ... </doc>
    """
    doc = BeautifulSoup(recStr, 'lxml', from_encoding=DEFAULT_ENCODING)

    docRoot = doc.find('doc')
    if docRoot is None:
        raise Exception('Invalid format, missing <doc> tag')

    return WikipediaRecordParsed(id=docRoot['id'], url=docRoot['url'], title=docRoot['title'], content=docRoot.text)


def procYahooAnswersRecord(recStr):
    """A procedure to parse a single Yahoo-answers format entry.

    :param recStr: Answer content including enclosing tags <document>...</document>
    :return:  parsed data as YahooAnswerRecParsed entry
    """
    doc = BeautifulSoup(recStr, 'lxml')

    docRoot = doc.find('document')
    if docRoot is None:
        raise Exception('Invalid format, missing <document> tag')

    uri = docRoot.find('uri')
    if uri is None:
        raise Exception('Invalid format, missing <uri> tag')
    uri = uri.text

    subject = docRoot.find('subject')
    if subject is None:
        raise Exception('Invalid format, missing <subject> tag')
    subject = removeTags(subject.text)

    content = docRoot.find('content')
    content = '' if content is None else removeTags(content.text)  # can probably be missing

    bestAnswer = docRoot.find('bestanswer')
    bestAnswer = '' if bestAnswer is None else bestAnswer.text  # is missing occaisionally

    bestAnswerId = -1

    answList = []
    answers = docRoot.find('nbestanswers')
    if answers is not None:
        for answ in answers.find_all('answer_item'):
            answText = answ.text
            if answText == bestAnswer:
                bestAnswerId = len(answList)
            answList.append(removeTags(answText))

    return YahooAnswerRecParsed(uri=uri, subject=subject.strip(), content=content.strip(),
                                bestAnswerId=bestAnswerId, answerList=answList)


def jsonlGen(fileName):
    """A generator that produces parsed doc/query entries one by one.
      :param fileName: an input file name
    """

    with FileWrapper(fileName) as f:
        for i, line in enumerate(f):
            ln = i + 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except:
                raise Exception('Error parsing JSON in line: %d' % ln)

            if not DOCID_FIELD in data:
                raise Exception('Missing %s field in JSON in line: %d' % (DOCID_FIELD, ln))

            yield data


def readQueries(fileName):
    """Read queries from a JSONL file and checks the document ID is set.

    :param fileName: an input file name
    :return: an array where each entry is a parsed query JSON.
    """
    return list(jsonlGen(fileName))


def writeQueries(queryList, fileName):
    """Write queries to a JSONL file.

    :param queryList: an array of parsed JSON query entries
    :param fileName: an output file
    """
    with open(fileName, 'w') as f:
        for e in queryList:
            f.write(json.dumps(e))
            f.write('\n')


def unique(arr):
    return list(set(arr))


def getRetokenized(tokenizer, text):
    """Obtain a space separated re-tokenized text.
    :param tokenizer:  a tokenizer that has the function
                       tokenize that returns an array of tokens.
    :param text:       a text to re-tokenize.
    """
    return ' '.join(tokenizer.tokenize(text))


def addRetokenizedField(dataEntry,
                        srcField,
                        dstField,
                        tokenizer):
    """
    Create a re-tokenized field from an existing one.

    :param dataEntry:   a dictionary of entries (keys are field names, values are text items)
    :param srcField:    a source field
    :param dstField:    a target field
    :param tokenizer:    a tokenizer to use, if None, nothing is done
    """
    if tokenizer is not None:
        dst = ''
        if srcField in dataEntry:
            dst = getRetokenized(tokenizer, dataEntry[srcField])

        dataEntry[dstField] = dst


def readDocIdsFromForwardFileHeader(fwdFileName):
    """Read document IDs from the textual header
       of a forward index. Some basic integrity checkes are done.

       :param   fwdFileName: input file name
       :return  a set of document IDs.
    """
    f = open(fwdFileName)
    lines = [s.strip() for s in f]
    f.close()
    docQty, _ = lines[0].split()
    docQty = int(docQty)

    assert len(lines) > docQty + 2, f"File {fwdFileName} is too short: length isn't consistent with the header info"
    assert lines[1] == "", f"The second line in {fwdFileName} isn't empty as expected!"
    assert lines[-1] == "", f"The last line in {fwdFileName} isn't empty as expected!"
    k = 2
    while k < len(lines) and lines[k] != '':
        k = k + 1
    assert lines[k] == ''  # We check that the last line is empty, we must find the empty line!
    k = k + 1

    assert k + docQty + 1 == len(lines)
    res = lines[k:len(lines) - 1]
    assert len(res) == docQty

    return set(res)
