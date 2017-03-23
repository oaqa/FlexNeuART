#!/bin/bash
. scripts/common.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr, stackoverflow, squad (1st arg)"
  exit 1
fi

PARTS_Q=("train" "dev1" "dev2" "test")
PARTS_A=(${PARTS_Q[*]})

# For some collections we use fewer questions for training & testing than it was originally supposed 
# by collection split
TEST_QMAX=""
TRAIN_QMAX=""

if [ "$collect" = "manner" ] ; then
  NOP=
elif [ "$collect" = "compr" -o "$collect" = "stackoverflow" ] ; then
  PARTS_A+=("tran")
  TRAIN_QMAX="50000"
  TEST_QMAX="10000"
elif [ "$collect" = "squad" ] ; then
  PARTS_A+=("tran" "wiki")
  PARTS_Q+=("tran" )
else
  echo "Unknown collection: $collect"
  exit 1
fi 

echo "Question parts: ${PARTS_Q[*]}"
echo "Answer parts: ${PARTS_A[*]}"

total_qty=0

# PARTS_A is supposed to contain all the parts in PARTS_Q

for part in ${PARTS_A[*]}  ; do
  echo -n "$part "
done
echo " total"

for part in ${PARTS_Q[*]}  ; do
  wcr=(`wc output/$collect/$part/SolrQuestionFile.txt`)
  qty_max=""
  qty=${wcr[0]}
  if [ "$part" = "test" ] ; then
    qty_max="$TEST_QMAX"
  fi
  if [ "$part" = "train" ] ; then
    qty_max="$TRAIN_QMAX"
  fi
  if [ "$qty_max" != "" ] ; then
    if [ "$qty" -gt "$qty_max" ] ; then
      qty=$qty_max
    fi
  fi
  echo -n "$qty "
  total_qty=$(($total+$qty))
done
echo " $total_qty"


wcq=(`wc output/manner/*/question_text|grep total`)
wca=(`wc output/manner/*/answer_text|grep total`)
echo "Average # of words in questions:"
echo "${wcq[1]}/${wcq[0]}" | bc -l
echo "Average # of words in answers:"
echo "${wca[1]}/${wca[0]}" | bc -l

