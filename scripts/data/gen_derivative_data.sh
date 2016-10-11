#!/bin/bash

#
# "This script will generated derivative data of several types:"
# (for StackOverflow not all the data is generated)
# "1. Filtered translation tables"
# "2. Sparse word embeddings based on filtered translation tables"
# "3. Filtered word embeddings (from several sources)"
# "4. Retrofitted versions of filtered word embeddings"
#

STACK_OVERFLOW="stackoverflow"

function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}

function check_pipe {
  f="${PIPESTATUS[*]}"
  name=$1
  if [ "$f" != "0 0" ] ; then
    echo "******************************************"
    echo "* Failed (pipe): $name, exit statuses: $f "
    echo "******************************************"
    exit 1
  fi
}

# Task 1, let's filter translation tables
GIZA_ITER_QTY=5
MIN_PROB_TRAN=0.0001
MAX_WORD_TRAN_QTY=1000000

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, $STACK_OVERFLOW"
  exit 1
fi

TRAN_DIR="tran/$collect"

if [ ! -d "$TRAN_DIR" ] ; then
  echo "Directory $TRAN_DIR doesn't exist"
  exit 1
fi

FIELD_LIST="text text_unlemm"

for field in $FIELD_LIST ; do
  TRAN_DIR_TEXT="$TRAN_DIR/$field"
  TRAN_DIR_TEXT_ORIG="$TRAN_DIR/$field.orig"

  if [ ! -d "$TRAN_DIR_TEXT_ORIG" ] ; then
    echo "Directory $TRAN_DIR_TEXT_ORIG doesn't exist"
    exit 1
  fi

  if [ ! -d "$TRAN_DIR_TEXT" ] ; then
    mkdir $TRAN_DIR_TEXT
    check "mkdir $TRAN_DIR_TEXT"
  fi

  scripts/giza/filter_tran_table_and_voc.sh tran/$collect $field $GIZA_ITER_QTY memfwdindex/$collect $MIN_PROB_TRAN $MAX_WORD_TRAN_QTY
  check "filter_tran_table_and_voc"
done

# Task 2: let's create sparse word embeddings based on filtered translation tables
if [ "$collect" != "$STACK_OVERFLOW" ] ; then
  MIN_PROB_HIGH_ORDER=0.001
  MAX_DIGIT=5
  MAX_MODEL_ORDER=4
  MAX_WORD_TRAN_EMBED_QTY=100000

  scripts/embeddings/run_gen_tran_embed.sh -giza_iter_qty $GIZA_ITER_QTY -giza_root_dir tran/$collect -m $MAX_MODEL_ORDER -memindex_dir memfwdindex/$collect/ -p $MIN_PROB_HIGH_ORDER -o WordEmbeddings/$collect/tran_embed -max_digit $MAX_DIGIT -max_word_qty $MAX_WORD_TRAN_EMBED_QTY
  check "run_gen_tran_embed.sh"
  #cd WordEmbeddings/$collect 
  #check "cd WordEmbeddings/$collect"
  #rm -f tran_embed*.bz2
  #bzip2 tran_embed*
  #check "bzip2"
  #cd -
fi

# Task 3: create filtered word embeddings (from several sources)"
COMPLETE_EMBED_DIR=WordEmbeddings/Complete
OUT_EMBED_DIR=WordEmbeddings/$collect
MAX_WORD_EMBED_QTY=100000

cd src/main/c
check "cd src/main/c"
make 
check "make"
cd -

FILT_CMD="scripts/embeddings/filter_embed.py memfwdindex/$collect/text_unlemm $MAX_WORD_EMBED_QTY"

#if [ "$collect" != "$STACK_OVERFLOW" ] ; then
#  # We don't compress word embeddings or else C++ code won't be able to read it!
#  for f in glove.twitter.27B.200d glove.840B.300d paragram_300_sl999  paragram_300_ws353 ; do
#    bzcat $COMPLETE_EMBED_DIR/$f.txt.bz2 | $FILT_CMD > $OUT_EMBED_DIR/$f.txt
#    check_pipe "bzcat $COMPLETE_EMBED_DIR/$fb.bz2 ... "
#  done
#fi

src/main/c/convert_word2vec $COMPLETE_EMBED_DIR/GoogleNews-vectors-negative300.bin | $FILT_CMD > $OUT_EMBED_DIR/word2vec.txt
check_pipe "src/main/c/convert_word2vec $COMPLETE_EMBED_DIR/GoogleNews-vectors-negative300.bin ... "

# Task 4 retrofit
#if [ "$collect" != "$STACK_OVERFLOW" ] ; then
#  EMBED_LIST="glove.twitter.27B.200d glove.840B.300d paragram_300_sl999  paragram_300_ws353 word2vec"
#else
  EMBED_LIST="word2vec"
#fi

MIN_RETROFIT_PROB=0.001
scripts/embeddings/do_retrofit.sh $collect  "$EMBED_LIST"  $MIN_RETROFIT_PROB
check "scripts/embeddings/do_retrofit.sh $collect  \"$EMBED_LIST\"  $MIN_RETROFIT_PROB"

