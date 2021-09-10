#!/bin/bash -e
# This script:
# 1. splits/copies downloaded Facebook DPR queries. The result is stored in the raw-input sub-directory
#    the queries are split into three parts: bitext, train_fusion, and development queries.
#    It is possible to specify partition sizes.
# 2. then it converts all the queries
#
source ./config.sh
source ./common_proc.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "INPUT_RAW_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "QUESTION_FILE_JSONL"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

boolOpts=("h" "help" "print help")

partNames="bitext,train_fusion,dev"
partSizes=",2500,2500"

seed=0

paramOpts=(
"seed"           "seed"        "random seed, default ($seed)"
"partition_sizes" "partSizes"   "sizes for partitions $partNames, empty means all remaining, default: $partSizes"
)

parseArguments $@

usageMain="<$SAMPLE_COLLECT_ARG> <download directory> <collection type>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

downloadDir=${posArgs[1]}
if [ "$downloadDir" = "" ] ; then
  genUsage "$usageMain" "Specify the download directory (2d arg)"
  exit 1
fi

colType=${posArgs[2]}
if [ "$colType" = "" ] ; then
  genUsage "$usageMain" "Specify a collection type: nq, trivia, or squad (3d arg)"
  exit 1
fi

inputRawDir="$COLLECT_ROOT/$collect/$INPUT_RAW_SUBDIR"

if [ ! -d "$inputRawDir" ] ; then
  mkdir -p "$inputRawDir"
fi


cp "$downloadDir/${colType}_dev.json.gz" "$inputRawDir/${colType}_dev_official.json.gz"

./data_convert/wikipedia_dpr/split_dpr_raw_queries.py \
  --seed $seed \
  --src_file "$downloadDir/${colType}_train.json.gz" \
  --partitions_names "$partNames" \
  --partitions_sizes "$partSizes"  \
  --dst_file_pref "$inputRawDir/${colType}"

# Finally convert the queries
for part in bitext train_fusion dev dev_official ; do
  outDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR/$part"
  if [ ! -d "$outDataDir" ] ; then
    mkdir -p "$outDataDir"
  fi

  if [ "$part" = "bitext" ] ; then
    bitextPathOpt="--out_bitext_path $COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$BITEXT_SUBDIR"
  else
    bitextPathOpt=""
  fi
  ./data_convert/wikipedia_dpr/convert_queries.py \
    --bert_tokenize \
    --input "$inputRawDir/${colType}_${part}.json.gz" \
    --part_type $part \
    --output_queries "$outDataDir/$QUESTION_FILE_JSONL" \
    --output_qrels "$outDataDir/$QREL_FILE" \
    $bitextPathOpt
done

