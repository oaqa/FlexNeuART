#!/bin/bash
. scripts/common.sh
URL_LIST=("http://trec.nist.gov/data/terabyte/06/06.topics.701-850.txt" \
          "http://trec.nist.gov/data/terabyte/04/04.qrels.12-Nov-04" "http://trec.nist.gov/data/terabyte/05/05.adhoc_qrels" "http://trec.nist.gov/data/terabyte/06/qrels.tb06.top50")
for url in ${URL_LIST[*]} ; do
  wget $url
  check "wget $url"
done
col=gov2
INPUT_DIR="input/$col"
OUTPUT_DIR="output/$col"
DIR=("$INPUT_DIR" "$OUTPUT_DIR/train" "$OUTPUT_DIR/test")
for d in ${DIR[*]} ; do
  if [ ! -d  $d ] ; then
    mkdir -p $d
    check "mkdir -p $d"
  fi
done
# This will also change relevance grade so that 1 becomes 3 and and 2 becomes 4
# Because 4 is the maximum grade in gdeval.pl
QREL_FILE="qrels_all_graded.txt"
cat "04.qrels.12-Nov-04" "05.adhoc_qrels" "qrels.tb06.top50" | awk '{r=$4;if (r>0) r+=2;print $1" "$2" "$3" "r;}'> $OUTPUT_DIR/train/$QREL_FILE
check_pipe "cat ..."
cp $OUTPUT_DIR/train/$QREL_FILE $OUTPUT_DIR/test/$QREL_FILE

rm "04.qrels.12-Nov-04" "05.adhoc_qrels" "qrels.tb06.top50" 
check "rm 04.qrels ... "

mv 06.topics.701-850.txt $INPUT_DIR/topics-701-850.txt
check "mv topics ..."
