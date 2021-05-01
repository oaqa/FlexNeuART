#!/bin/bash -e
source scripts/config.sh
source scripts/common_proc.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "QUESTION_FILE_JSONL"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

src=$1
if [ "$src" = "" ] ; then
  echo "Specify the source directory (1st arg)"
  exit 1
fi
collPref=$2
if [ "$collPref" = "" ] ; then
  echo "Specify the prefix name of the target collection (2d arg)"
  exit 1
fi

for colType in nq trivia squad ; do
  collect="${collPref}_${colType}"

  for part in train dev ; do
    inpFile="$src/${colType}_${part}.json.gz"
    inputDataSubDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR/$part"

    echo "Source file: $src"
    echo "Target data directory: $inputDataSubDir"

    if [ ! -d "$inputDataSubDir" ] ; then
      mkdir -p "$inputDataSubDir"
    fi

    bitextOpt=""
    if [ "$part" = "train" ] ; then
      bitextOpt="--out_bitext_path "$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$BITEXT_SUBDIR""
    fi

    python -u scripts/data_convert/wikipedia_dpr/convert_queries.py \
        --input "$inpFile" \
        --output_queries "$inputDataSubDir/$QUESTION_FILE_JSONL" \
        --output_qrels "$inputDataSubDir/$QREL_FILE" \
        --bert_tokenize \
        $bitextOpt

  done
done
