#!/bin/bash -e
# The main script to convert MSMARCO (v2) document collection
source ./data_convert/common_conv.sh

checkVarNonEmpty "ANSWER_FILE_JSONL"
checkVarNonEmpty "QUESTION_FILE_JSONL"
checkVarNonEmpty "inputDataDir"
checkVarNonEmpty "QREL_FILE"

BERT_TOK_OPT=" --bert_tokenize"

for part in pass train dev dev2 ; do
  mkdir -p "$inputDataDir/$part"
done

inputDoc2PassDir="$COLLECT_ROOT/$collect/input_data_doc2pass/docs"
echo "Additional input sub-directory with doc2pass mapping data: $inputDoc2PassDir"
mkdir -p "$inputDoc2PassDir"

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

for part in train dev dev2 ; do
  cp "$src/passv2_${part}_qrels.tsv" "$inputDataDir/$part/$QREL_FILE"
  ./data_convert/msmarco/convert_queries.py \
    $BERT_TOK_OPT \
    --input  "$src/passv2_${part}_queries.tsv" \
    --output "$inputDataDir/$part/$QUESTION_FILE_JSONL"
done


python -u ./data_convert/msmarco_v2/convert_pass.py \
    $BERT_TOK_OPT \
    --input "$src/msmarco_v2_passage"  \
    --output_main "$inputDataDir/pass/${ANSWER_FILE_JSONL}.gz"  \
    --output_doc2pass "$inputDoc2PassDir/${ANSWER_FILE_JSONL}.gz"


