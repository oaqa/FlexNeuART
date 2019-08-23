import gzip
import collections
import re
from bs4 import BeautifulSoup

SPACY_MODEL = 'en_core_web_sm'
STOPWORD_FILE = 'data/stopwords.txt'
MAX_DOC_SIZE=16536 # 16 K should be more than enough!
REPORT_QTY=10000

class GizppedFileWrapperIterator:

  def __init__(self, fileHandler):
    self._file = fileHandler

class FileWrapper:

  def __enter__(self):
      return self

  def __init__(self, fileName, flags='r'):
    """Constructor, which opens a regular or gzipped-file

      :param  fileName a name of the file, it has a '.gz' extension, we open a gzip-stream.
      :param  flags    open flags such as 'r' or 'w'
    """
    if fileName.endswith('.gz'): 
      self._file = gzip.open(fileName, flags) 
      self._isGziped = True
    else: 
      self._file = open(fileName, flags)
      self._isGziped = False

  def write(self, s):
    if self._isGziped:
      self._file.write(s.encode())  
    else:
      self._file.write(s)

  def read(self, s):
    if self._isGziped:
      return self._file.read().decode()  
    else:
      return self._file.read()

  def close(self):
    self._file.close()

  def __exit__(self, type, value, tb):
    self._file.close()

  def __iter__(self):
    for line in self._file:
      yield line.decode() if self._isGziped else line


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
          continue # Can skip empty lines
        if line.startswith(startEntry) :
          if not seenEnd:
            raise Exception(f'Invalid format, no previous end tag, line {ln} file {fileName}')
          assert(not recLines)
          recLines.append(line)
          seenEnd = False
          seenStart = True
        else:
          raise Exception(f'Invalid format, no previous start tag, line {ln} file {fileName}')
      else:
        recLines.append(line)
        noSpaceLine = line.replace(' ', '').strip() # End tags may contain spaces
        if noSpaceLine.endswith(endEntry):
          if not seenStart:
            raise Exception(f'Invalid format, no previous start tag, line {ln} file {fileName}')
          yield ''.join(recLines)
          recLines = []
          seenEnd = True
          seenStart = False

    if recLines:
      raise Exception(f'Invalid trailine entries in the file {fileName} %d entries left' % (len(recLines)))

def removeTags(str):
  """Just remove anything that looks like a tag"""
  return re.sub(r'</?[a-z]+\s*/?>', '', str)

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
  content = '' if content is None else removeTags(content.text) # can probably be missing

  bestAnswer = docRoot.find('bestanswer')
  bestAnswer = '' if bestAnswer is None else bestAnswer.text # is missing occaisionally

  bestAnswerId = -1

  answList = []
  answers = docRoot.find('nbestanswers')
  if answers is not None:
    for answ in answers.find_all('answer_item'):
      answText = answ.text
      if answText == bestAnswer:
        bestAnswerId = len(answList)
      answList.append(removeTags(answText))

  YahooAnswerRecParsed = collections.namedtuple('YahooAnswerRecParsed', 'uri subject content bestAnswerId answerList')

  return YahooAnswerRecParsed(uri=uri, subject=subject.strip(), content=content.strip(),
                              bestAnswerId=bestAnswerId, answerList=answList)