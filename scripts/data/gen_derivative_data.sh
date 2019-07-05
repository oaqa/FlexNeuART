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
MIN_PROB_TRAN=0.001
#MAX_WORD_TRAN_QTY=1000000

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, $STACK_OVERFLOW"
  exit 1
fi

tran_collect=$collect
if [ "$collect" = "manner" ] ; then
  tran_collect="ComprMinusManner"
fi

TRAN_DIR="tran/$tran_collect"

if [ ! -d "$TRAN_DIR" ] ; then
  echo "Directory $TRAN_DIR doesn't exist"
  exit 1
fi

FIELD_LIST="text_unlemm text"

for field in $FIELD_LIST ; do
  TRAN_DIR_TEXT="$TRAN_DIR/$field"
  TRAN_DIR_TEXT_ORIG="$TRAN_DIR/$field.orig"

  if [ ! -d "$TRAN_DIR_TEXT_ORIG" ] ; then
    echo "Warning: Directory $TRAN_DIR_TEXT_ORIG doesn't exist"
    exit 1
  fi

  if [ ! -d "$TRAN_DIR_TEXT" ] ; then
    mkdir $TRAN_DIR_TEXT
    check "mkdir $TRAN_DIR_TEXT"
  fi

  scripts/giza/simple_filter_tran_table.sh $TRAN_DIR $field $GIZA_ITER_QTY $MIN_PROB_TRAN 
  check "simple_filter_tran_table.sh"
done

# Create filtered word embeddings (from several sources)"
COMPLETE_EMBED_DIR=WordEmbeddings/Complete
OUT_EMBED_DIR=WordEmbeddings/$collect
MAX_WORD_EMBED_QTY=100000

if [ ! -d "$OUT_EMBED_DIR" ] ; then
  mkdir -p "$OUT_EMBED_DIR"
  check "mkdir -p $OUT_EMBED_DIR"
fi

cd src/main/c
check "cd src/main/c"
make 
check "make"
cd -

FILT_CMD="scripts/embeddings/filter_embed.py memfwdindex/$collect/text_unlemm $MAX_WORD_EMBED_QTY"

src/main/c/convert_word2vec $COMPLETE_EMBED_DIR/GoogleNews-vectors-negative300.bin | $FILT_CMD > $OUT_EMBED_DIR/word2vec.txt
check_pipe "src/main/c/convert_word2vec $COMPLETE_EMBED_DIR/GoogleNews-vectors-negative300.bin ... "

cat $OUT_EMBED_DIR/word2vec_tran_text_unlemm_dim=300_unfilt.txt | $FILT_CMD > $OUT_EMBED_DIR/word2vec_tran_text_unlemm_dim=300_filt.txt
check_pipe "cat $OUT_EMBED_DIR/word2vec_tran_text_unlemm_dim=300_unfilt.txt | $FILT_CMD > $OUT_EMBED_DIR/word2vec_tran_text_unlemm_dim=300_filt.txt"

# Task 4 retrofit
EMBED_LIST="word2vec word2vec_tran_text_unlemm_dim=300_filt"

MIN_RETROFIT_PROB=0.001
scripts/embeddings/do_retrofit.sh $collect  "$EMBED_LIST"  $MIN_RETROFIT_PROB
check "scripts/embeddings/do_retrofit.sh $collect  \"$EMBED_LIST\"  $MIN_RETROFIT_PROB"

