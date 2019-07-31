import gzip

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
