#!/bin/bash -e
# The main script to convert MSMARCO document collection
. scripts/data_convert/common_conv.sh

checkVarNonEmpty "ANSWER_FILE"
checkVarNonEmpty "QUESTION_FILE"
checkVarNonEmpty "inputDataDir"
checkVarNonEmpty "QREL_FILE"

BERT_TOK_OPT=" --bert_tokenize"

for part in docs train dev test2019 ; do
  mkdir -p "$inputDataDir/$part"
done

python -u scripts/data_convert/msmarco_adhoc/convert_docs.py \
    $BERT_TOK_OPT \
    --input "$src/msmarco-docs.tsv.gz"  \
    --output "$inputDataDir/docs/${ANSWER_FILE}.gz"

for part in train dev ; do
  zcat $src/msmarco-doc${part}-qrels.tsv.gz > "$inputDataDir/$part/$QREL_FILE"
  scripts/data_convert/msmarco_adhoc/convert_queries.py \
    $BERT_TOK_OPT \
    --input  "$src/msmarco-doc${part}-queries.tsv.gz" \
    --output "$inputDataDir/$part/$QUESTION_FILE"
done

scripts/data_convert/msmarco_adhoc/convert_queries.py \
    $BERT_TOK_OPT \
    --input  "$src/msmarco-test2019-queries.tsv.gz" \
    --output "$inputDataDir/test2019/$QUESTION_FILE"


