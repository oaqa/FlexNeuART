import collections
import numpy as np
import sklearn as skl
from tqdm import tqdm

FAKE_RUN_ID="fake_run"

QrelEntry = collections.namedtuple('QrelEntry',
                                   'queryId docId relGrade')

def genQrelStr(queryId, docId, relGrade):
  """Produces a string representing one QREL entry

  :param queryId:   question/query ID
  :param docId:     relevanet document/answer ID
  :param relGrade:  relevance grade

  :return: a string representing one QREL entry
  """
  return f'{queryId} 0 {docId} {relGrade}'


def qrelEntry2Str(qrelEntry):
  """Convert a parsed QREL entry to string.

  :param qrelEntry: input of the type QrelEntry
  :return:  string representation.
  """
  return genQrelStr(qrelEntry.queryId, qrelEntry.docId, qrelEntry.relGrade)


def parseQrelEntry(line):
  """Parse one QREL entry
  :param line  a single line with a QREL entry.
          Relevance graded is expected to be integer.

  :return a parsed QrelEntry entry.
  """

  line = line.strip()
  parts = line.split()
  if len(parts) != 4:
    raise Exception('QREL entry format error, expecting just 4 white-space separted field in the entry: ' + line)

  return QrelEntry(queryId=parts[0], docId=parts[2], relGrade=int(parts[3]))


def readQrels(fileName):
  """Read and parse QRELs.

  :param fileName: input file name
  :return: an array of parsed QREL entries
  """
  ln = 0
  res = []

  with open(fileName) as f:
    for line in tqdm(f, desc='loading qrels (by line)', leave=False):
      ln += 1
      line = line.strip()
      if not line:
        continue
      try:
        e = parseQrelEntry(line)
        res.append(e)
      except:
        raise Exception('Error parsing QRELs in line: %d' % ln)

  return res

def writeQrels(qrelList, fileName):
  """Write a list of QRELs to a file.

  :param qrelList:  a list of parsed QRELs
  :param fileName:  an output file name
  """
  with open(fileName, 'w') as f:
    for e in qrelList:
      f.write(qrelEntry2Str(e))
      f.write('\n')


def genRunEntryStr(queryId, docId, rank, score, runId):
  """A simple function to generate one run entry.

  :param queryId: query id
  :param docId:   document id
  :param rank:    entry rank
  :param score:   entry score
  :param runId:   run id

  """
  return f'{queryId} Q0 {docId} {rank} {score} {runId}'


def readQrelsDict(fileName):
  """Read QRELs in the form of a dictionary where keys are query IDs.

  :param fileName: QREL file name
  :return: a dictionary of dictionaries
  """
  result = {}
  for e in readQrels(fileName):
    result.setdefault(e.queryId, {})[e.docId] = int(e.relGrade)
  return result


def readRunDict(fileName):
  """Read a run file in the form of a dictionary where keys are query IDs.

  :param fileName: run file name
  :return:
  """
  result = {}
  for line in tqdm(fileName, desc='loading run (by line)', leave=False):
      qid, _, docid, rank, score, _ = line.split()
      result.setdefault(qid, {})[docid] = float(score)
  return result


def evalRun(runFileName, qrelFileName, metricFunc, topK, isBinary, debug=False):
  """Evaluate run stored in a file using QRELs stored in a file.

  :param runFileName:   a run file name
  :param qrelFileName:  a QREL file name
  :param metricFunc:    an sklearn metric function
  :param topK:          a cutoff (include only this number of entries
  :param isBinary:      true if relevance grades need to be binarized
  :return:  the average metric value
  """

  resArr = []

  print(f'Evaluating run {runFileName} with QREL file {qrelFileName} using metric function {metricFunc}')
  run = readRunDict(runFileName)
  qrels = readQrelsDict(qrelFileName)
  for qid, scoreDict in run.items():
    tmp = [(score, did) for did, score in scoreDict.items()]
    y_true = []
    y_pred = []
    for score, did in sorted(tmp, reverse=True):
      y_pred.append(score)
      rel = 0
      if qid in qrels and did in qrels[qid]:
        rel = qrels[qid]
      if isBinary:
        rel = int(rel > 0)
      y_true.append(rel)
    if topK is not None and topK > 0:
      y_true = y_true[0:topK]
      y_pred = y_pred[0:topK]
    val = metricFunc(y_true, y_pred)
    if debug:
      print('%s %g' % (qid, val))
    resArr.append(val)

  res = np.mean(resArr)
  if debug:
    print('mean %g' % res)

  return res
