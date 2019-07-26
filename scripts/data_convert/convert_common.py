#!/usr/bin/env python -u
import gzip

STOPWORD_FILE = 'data/stopwords.txt'
MAX_DOC_SIZE=16536 # 16 K should be more than enough!

def openFile(fileName, flags='r'):
  """Opens a regular or gzipped-file

    :param  fileName a name of the file, it has a '.gz' extension, we open a gzip-stream.
    :param  flags    open flags such as 'r' or 'w'
    :return a file handler that needs to be closed
  """
  return gzip.open(fileName, flags) if fileName.endswith('.gz') else open(fileName, flags)

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
