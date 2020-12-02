#!/bin/bash -e
# The main script to convert MSMARCO passage collection 
# It is called after download_msmarco_pass.sh
. scripts/data_convert/common_conv.sh

checkVarNonEmpty "ANSWER_FILE"
checkVarNonEmpty "QUESTION_FILE"
checkVarNonEmpty "inputDataDir"


BERT_TOK_OPT=" --bert_tokenize"

for part in pass train dev eval test2019 test2020 ; do
  mkdir -p $inputDataDir/$part
done

for year in 2019 2020 ; do
  python -u scripts/data_convert/msmarco/convert_queries.py \
    $BERT_TOK_OPT \
    --input  "$src/msmarco-test${year}-queries.tsv" \
    --output "$inputDataDir/test${year}/$QUESTION_FILE"
done

python -u scripts/data_convert/msmarco/convert_pass.py \
    $BERT_TOK_OPT \
    --input "$src/collection.tsv.gz"  \
    --output "$inputDataDir/pass/${ANSWER_FILE}.gz"

for part in train dev eval ; do
  # eval has no qrels for some reason
  if [ "$part" != "eval" ] ; then
    cp "$src/qrels.$part.tsv" "$inputDataDir/$part/$QREL_FILE"
  fi

  python -u scripts/data_convert/msmarco/convert_queries.py \
    $BERT_TOK_OPT \
    --input  "$src/queries.$part.tsv" \
    --output "$inputDataDir/$part/$QUESTION_FILE"
done


cp "$src/2019qrels-pass.txt" "$inputDataDir/test2019/$QREL_FILE"