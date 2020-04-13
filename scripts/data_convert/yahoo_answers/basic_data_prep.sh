#!/bin/bash -e

. scripts/common_proc.sh
. scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"

boolOpts=(\
"h" "help" "print help"
)

parseArguments $@

usageMain="<collection name> <mgiza location>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify <collection name> (1st arg)"
  exit 1
fi

mgizaDir=${posArgs[1]}
if [ "$mgizaDir" = "" ] ; then
  genUsage "$usageMain" "Specify <mgiza location> (2d arg)"
  exit 1
fi
if [ ! -d "$mgizaDir" ] ; then
  echo "Not a directory: $mgizaDir"
  exit 1
fi

rm -rf $COLLECT_ROOT/$collect/$LUCENE_CACHE_SUBDIR/

scripts/index/create_lucene_index.sh $collect

scripts/index/create_fwd_index.sh \
  $COLLECT_ROOT mapdb \
  'text:parsedBOW text_unlemm:parsedText text_bert_tok:parsedText text_raw:raw'



export minTranProb=0.001
export topWordQty=100000


for fieldName in text_unlemm text_bert_tok ; do

  scripts/cedr/build_vocab.py \
    --field $fieldName \
    --input $COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR/$BITEXT_SUBDIR/${ANSWER_FILE}.gz  \
            $COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR/$DEFAULT_TRAIN_SUBDIR/${ANSWER_FILE}.gz  \
    --output $COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/vocab/$fieldName.voc

  scripts/giza/create_tran.sh \
    $collect \
    fieldName \
    "$mgizaDir"

  scripts/giza/filter_tran_table_and_voc.sh \
    $collect \
    $fieldName \
    $minTranProb \
    $topWordQty


done