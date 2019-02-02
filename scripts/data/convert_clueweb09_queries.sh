#!/bin/bash -e
source scripts/common.sh

COL_DIR=$1

if [ "$COL_DIR" = "" ] ; then
  echo "Specify the collection directory (1st arg)"
  exit 1
fi
if [ ! -d "$COL_DIR" ] ; then
  echo "Not a directory $COL_DIR (1st arg)"
  exit 1
fi

CLUEWEB09_COMMWORD_FILE=$2
if [ "$CLUEWEB09_COMMWORD_FILE" = "" ] ; then
  echo "Specify the CLUEWEB09 commmon file with common words, e.g., ~/TextCollect/ClueWeb09FreqWords/d20 (2d arg)"
  exit 1
fi
if [ ! -f "$CLUEWEB09_COMMWORD_FILE" ] ; then
  echo "Not a file $CLUEWEB09_COMMWORD_FILE (2d arg)"
  exit 1
fi

checkVarNonEmpty "$COL_DIR" "collection directory"
rm -rf "$COL_DIR/output"

mkdir -p "$COL_DIR/output/train"
mkdir -p "$COL_DIR/output/test"

outDir="$COL_DIR/output/train"
# These are all very idiosyncratic scripts assuming files are in specific locations

scripts/data/extract_nonoverlap_queries_with_qrels_for1MQ.py  "$COL_DIR"

# in a top-level directory.

for  subDir in output/train output/test ; do
  scripts/data/run_convert_clueweb09_queries.sh  -common_word_file "$CLUEWEB09_COMMWORD_FILE" -in_file "$COL_DIR/$subDir/queries.txt" -solr_file "$COL_DIR/$subDir/SolrAnswerFile.txt" -stop_word_file data/stopwords.txt 
done
