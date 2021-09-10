#!/bin/bash -e
# The main script to convert Cranfield collection
# It is called after download_msmarco_pass.sh
source ./data_convert/common_conv.sh

checkVarNonEmpty "ANSWER_FILE_JSONL"
checkVarNonEmpty "QUESTION_FILE_JSONL"
checkVarNonEmpty "inputDataDir"
checkVarNonEmpty "QREL_FILE"

BERT_TOK_OPT=" --bert_tokenize "

for part in docs queries ; do
  mkdir -p $inputDataDir/$part
done

python -u ./data_convert/cranfield/convert_docs.py \
 $BERT_TOK_OPT \
--input "$src/cran.all.1400" \
--output "$inputDataDir/docs/${ANSWER_FILE_JSONL}.gz"

python -u ./data_convert/cranfield/convert_queries.py \
 $BERT_TOK_OPT \
--input "$src/cran.qry" \
--output "$inputDataDir/queries/${QUESTION_FILE_JSONL}"

# binary QRELs (just in case we would want to use them later)
cat "$src/cranqrel" | awk '{qrel=$3 > 0; print $1" 0 "$2" "qrel}' > "$inputDataDir/queries/qrels.txt.binary"
# graded QRELs, but no negatives!
cat "$src/cranqrel" | awk '{qrel=$3; if (qrel < 0) qrel=0; print $1" 0 "$2" "qrel}' > "$inputDataDir/queries/$QREL_FILE"
