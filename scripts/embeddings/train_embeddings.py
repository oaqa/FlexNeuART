import sys

def Usage(err):
  if not err is None:
    print err
  print "Usage: <top level output directory> <collection name: e,g., compr, stackoverflow> <field name> <output file prefix> <initial *BINARY* model or empty string> <vector dimensionality> <min word count> <# iterations>"
  sys.exit(1)


def quest_file_name(d, fieldName):
  return d + '/question_' + fieldName

def answ_file_name(d, fieldName):
  return d + '/answer_' + fieldName

if len(sys.argv) != 9:
  Usage(None)

topLevelDir = sys.argv[1]
colName     = sys.argv[2]
fieldName   = sys.argv[3]
outPrefix   = sys.argv[4]
initModel   = sys.argv[5]
dim         = int(sys.argv[6])
minCount    = int(sys.argv[7])
numIter     = int(sys.argv[8])

srcDir = topLevelDir + '/' + colName + '/tran'

if not os.path.isdir(srcDir):
  Usage("Cannot find source directory: '" + srcDir + "'")


from multiprocessing import cpu_count
from gensim.models.word2vec import Word2Vec, LineSentence

answerSource   = LineSentence(answ_file_name(srcDir, fieldName))
questionSource = LineSentence(quest_file_name(srcDir, fieldName))

print "Creating a model with iter=%d min_count=%d workers=%d" % (numIter, minCount, cpu_count())
model = Word2Vec(iter=numIter, min_count=minCount, workers=cpu_count())
if initModel != '':
  print "Loading binary model %s" % initModel
  model = Word2Vec.load_word2vec_format(initModel, binary=True)
  print "Initial model is loaded!" 

print "Building vocabulary from answers"
model.build_vocab(answerSource, update=True)
print "Training on answers"
model.train(answerSource)
print "Building vocabulary from questions"
model.build_vocab(questionSource, update=True)
print "Training on questions"
model.train(questionSource)
print "Training is finished!"

print "Saving the model"
model.save(outPrefix + ".model")
model.save_word2vec_format(outPrefix + ".txt")

