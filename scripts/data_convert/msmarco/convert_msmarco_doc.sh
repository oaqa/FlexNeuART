#!/bin/bash -e
# The main script to convert MSMARCO document collection
. scripts/data_convert/common_conv.sh

checkVarNonEmpty "ANSWER_FILE_JSONL"
checkVarNonEmpty "QUESTION_FILE_JSONL"
checkVarNonEmpty "inputDataDir"
checkVarNonEmpty "QREL_FILE"

BERT_TOK_OPT=" --bert_tokenize"

for part in docs train dev test2019 test2020 ; do
  mkdir -p "$inputDataDir/$part"
done

for year in 2019 2020 ; do
  python -u scripts/data_convert/msmarco/convert_queries.py \
    $BERT_TOK_OPT \
    --input  "$src/msmarco-test${year}-queries.tsv" \
    --output "$inputDataDir/test${year}/$QUESTION_FILE_JSONL"
done

python -u scripts/data_convert/msmarco/convert_docs.py \
    $BERT_TOK_OPT \
    --input "$src/msmarco-docs.tsv.gz"  \
    --output "$inputDataDir/docs/${ANSWER_FILE_JSONL}.gz"

for part in train dev ; do
  zcat $src/msmarco-doc${part}-qrels.tsv.gz > "$inputDataDir/$part/$QREL_FILE"
  scripts/data_convert/msmarco/convert_queries.py \
    $BERT_TOK_OPT \
    --input  "$src/msmarco-doc${part}-queries.tsv.gz" \
    --output "$inputDataDir/$part/$QUESTION_FILE_JSONL"
done

cp $src/2019qrels-docs.txt "$inputDataDir/test2019/$QREL_FILE"

