#!/bin/bash -e
# The main script to convert MSMARCO passage collection 
# It should be invoked after a bit of data massaging,
# which is down by download_msmarco_pass.sh script
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

mkdir -p $dst/input_data/test2019
python -u scripts/data_convert/msmarco_adhoc/convert_queries.py --input  $src/msmarco-test2019-queries.tsv --output $dst/input_data/test2019/QuestionFields.jsonl

for part in devTop1000 pass train dev eval ; do
  mkdir -p $dst/input_data/$part
done

for part in train dev ; do
  cp $src/qrels.$part.tsv $dst/input_data/$part/qrels.txt
done
cp $src/qrels.dev.tsv $dst/input_data/devTop1000/qrels.txt

for part in train dev eval devTop1000 ; do
  python -u scripts/data_convert/msmarco_adhoc/convert_queries.py --input  $src/queries.$part.tsv --output $dst/input_data/$part/QuestionFields.jsonl
done

python -u scripts/data_convert/msmarco_adhoc/convert_pass.py --input $src/collection.tsv.gz  --output $dst/input_data/pass/AnswerFields.jsonl.gz 

