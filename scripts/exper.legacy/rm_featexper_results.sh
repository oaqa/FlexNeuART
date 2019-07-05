#!/bin/bash
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

FEATURE_DESC_FILE="$2"
if [ "$FEATURE_DESC_FILE" = "" ] ; then
  echo "Specify a feature description file (2d arg)"
  exit 1
fi
if [ ! -f "$FEATURE_DESC_FILE" ] ; then
  echo "Not a file (2d arg)"
  exit 1
fi
FILT_N="$3"

function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}

EXPER_DIR="results/feature_exper/"


echo -e "extractor_type\tembed_list\ttop_k\trecall\tquery_num\tnum_correct\tP@1\tMRR"

n=`wc -l "$FEATURE_DESC_FILE"|awk '{print $1}'`
n=$(($n+1))
for ((i=1;i<$n;++i))
  do
    line=`head -$i "$FEATURE_DESC_FILE"|tail -1`
    if [ "$line" !=  "" ]
    then
      EXTR_TYPE=`echo $line|awk '{print $1}'`
      EMBED_LIST=`echo $line|awk '{print $2}'`
      TEST_SET=`echo $line|awk '{print $3}'`
      # Each experiment should run in its separate directory
      suffix="$EXTR_TYPE/$EMBED_LIST"
      EXPER_DIR_UNIQUE="$EXPER_DIR/$collect/$TEST_SET/$suffix"
      if [ ! -d "$EXPER_DIR_UNIQUE" ] ; then
        echo "Directory doesn't exist: $EXPER_DIR_UNIQUE"
        exit 1
      fi
      echo "Are you sure that you want to delete the directory (type yes to confirm)"
      echo "$EXPER_DIR_UNIQUE"
      read -a answ
      if [ "$answ" = "yes" ] ; then
        echo "Answer is yes: deleting"
        rm -rf "$EXPER_DIR_UNIQUE"
        check "rm -rf "$EXPER_DIR_UNIQUE""
      else
        echo "Answer is not yes: skipping"
      fi
    fi
  done
