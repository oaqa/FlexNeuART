#!/bin/bash -e
# The main script to convert MSMARCO document collection
source scripts/common_proc.sh
src=$1
if [ "$src" = "" ] ; then
  echo "Specify the source directory (1st arg)"
  exit 1
fi
dst=$2
if [ "$dst" = "" ] ; then
  echo "Specify the target directory (2d arg)"
  exit 1
fi

for part in docs train dev test2019 ; do
  mkdir -p $dst/input_data/$part
done

for part in train dev ; do
  zcat $src/msmarco-doc${part}-qrels.tsv.gz > $dst/input_data/$part/qrels.txt
  scripts/data_convert/msmarco_adhoc/convert_queries.py --input  $src/msmarco-doc${part}-queries.tsv.gz --output $dst/input_data/$part/QuestionFields.jsonl
done

scripts/data_convert/msmarco_adhoc/convert_queries.py --input  $src/msmarco-test2019-queries.tsv.gz --output $dst/input_data/test2019/QuestionFields.jsonl

python -u scripts/data_convert/msmarco_adhoc/convert_docs.py --input $src/msmarco-docs.tsv.gz  --output $dst/input_data/docs/AnswerFields.jsonl.gz 

