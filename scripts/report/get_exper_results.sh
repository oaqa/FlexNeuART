#!/bin/bash
. scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FEAT_EXPER_SUBDIR"
checkVarNonEmpty "EXPER_DESC_SUBDIR"

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr, squad (1st arg)"
  exit 1
fi
experDescLoc="$COLLECT_ROOT/$collect/$EXPER_DESC_SUBDIR"

EXTRACTORS_DESC=$2
if [ "$EXTRACTORS_DESC" = "" ] ; then
  "Specify a file with extractor description relative to dir. '$experDescLoc' (2d arg)"
  exit 1
fi
experDescPath=$experDescLoc/$EXTRACTORS_DESC
if [ ! -f "$experDescPath" ] ; then
  echo "Not a file '$experDescPath' (wrong 2d arg)"
  exit 1
fi

FILT_N="$3"
if [ "$FILT_N" = "" ] ; then
  FILT_N="*"
fi

experDir="$COLLECT_ROOT/$collect/$FEAT_EXPER_SUBDIR"

echo -e "extractor_type\ttop_k\tquery_qty\tNDCG@20\tERR@20\tP@20\tMAP\tMRR\tRecall"

n=`wc -l "$experDescPath"|awk '{print $1}'`
n=$(($n+1))
for ((ivar=1;ivar<$n;++ivar))
  do
    line=`head -$ivar "$experDescPath"|tail -1`
    line=$(removeComment "$line")
    if [ "$line" !=  "" ]
    then
      # Each experiment should run in its separate directory
      extrType=`echo $line|awk '{print $1}'`
      testSet=`echo $line|awk '{print $2}'`
      experSubdir=`echo $line|awk '{print $3}'`

      experDirUnique=$(getExperDirUnique "$experDir" "$testSet" "$experSubdir")
      if [ ! -d "$experDirUnique" ] ; then
        echo "Directory doesn't exist: $experDirUnique"
        exit 1
      fi
      pd=$PWD
      cd $experDirUnique/rep
      check "cd $experDirUnique/rep"

      # Let's read timings
      query_time="N/A"
      stat_file="stat_file"
      if [ -f "$stat_file" ] ; then
        fn=`head -1 $stat_file|cut -f 1`
        if [ "$fn" != "QueryTime" ] ; then
          "Wrong format of the file (expecting that the first field is QueryTime, but got: '$fn'): $stat_file"
          exit 1
        fi
        query_time=`head -2 $stat_file|tail -1|cut -f 1`
        if [ "$query_time" = "" ] ; then
          "Cannot retrieve QueryTime from line 2 in the file: $stat_file"
          exit 1
        fi
      fi

      for f in `ls -tr out_${FILT_N}.rep` ; do 
        top_k=`echo $f|sed 's/out_//'|sed 's/.rep//'`
        query_qty=`grepFileForVal "$f" "# of queries"`
        ndcg20=`grepFileForVal "$f" "NDCG@20"`
        err20=`grepFileForVal "$f" "ERR@20"`
        p20=`grepFileForVal "$f" "P@20"`
        map=`grepFileForVal "$f" "MAP"`
        mrr=`grepFileForVal "$f" "Reciprocal rank"`
        recall=`grepFileForVal "$f" "Recall"`
        echo -e "$extrType\t$top_k\t$query_qty\t$ndcg20\t$err20\t$p20\t$map\t$mrr\t$recall"
      done
      cd $pd
      check "cd $pd"
    fi
  done
