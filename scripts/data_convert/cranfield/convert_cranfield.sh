#!/bin/bash -e
# The main script to convert Cranfield collection
# It is called after download_msmarco_pass.sh
. scripts/data_convert/common_conv.sh

checkVarNonEmpty "ANSWER_FILE"
checkVarNonEmpty "QUESTION_FILE"
checkVarNonEmpty "inputDataDir"
checkVarNonEmpty "QREL_FILE"

BERT_TOK_OPT=" --bert_tokenize "

for part in docs queries ; do
  mkdir -p $inputDataDir/$part
done

python -u scripts/data_convert/cranfield/convert_docs.py \
 $BERT_TOK_OPT \
--input "$src/cran.all.1400" \
--output "$inputDataDir/docs/${ANSWER_FILE}.gz"

python -u scripts/data_convert/cranfield/convert_queries.py \
 $BERT_TOK_OPT \
--input "$src/cran.qry" \
--output "$inputDataDir/queries/${QUESTION_FILE}"

# binary QRELs
#cat "$src/cranqrel" | awk '{qrel=$3 > 0; print $1" 0 "$2" "qrel}' > "$inputDataDir/queries/$QREL_FILE"
# graded QRELs, but no negatives!
cat "$src/cranqrel" | awk '{qrel=$3; if (qrel < 0) qrel=0; print $1" 0 "$2" "qrel}' > "$inputDataDir/queries/$QREL_FILE"