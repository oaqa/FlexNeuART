#!/bin/bash
. scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "REP_SUBDIR"
checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "TEST_PART_PARAM"
checkVarNonEmpty "TEST_PART_PARAM"
checkVarNonEmpty "STAT_FILE"

boolOpts=(\
"h" "help" "print help" \
"debug_print" "debug" "print every executed command"
)

paramOpts=(\
"test_part" "defaultTestPart" "default test set, e.g., dev1" \
)

parseArgs $@

usageMain="<collection> <feature desc. file relative to collection root> <filter for cand. qty>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 0
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

collectSubdir="$COLLECT_ROOT/$collect"

featDescFile=${posArgs[1]}
if [ "$featDescFile" = "" ] ; then
  genUsage "$usageMain" "Specify a feature description file *RELATIVE* to $collectSubdir (2d arg)"
  exit 1
fi

checkVarNonEmpty "featDescFile"
experDescPath=$collectSubdir/$featDescFile
if [ ! -f "$experDescPath" ] ; then
  echo "Not a file '$experDescPath'"
  exit 1
fi

fltN="${posArgs[2]}"
if [ "$fltN" = "" ] ; then
  fltN="*"
fi

if [ "$debug" = "1" ] ; then
  set -x
fi


tmpConf=`mktemp`

echo -e "extractor_type\ttop_k\tquery_qty\tNDCG@20\tERR@20\tP@20\tMAP\tMRR\tRecall"

ivar=0
for ((ivar=1;;++ivar)) ; do

  stat=`scripts/exper/parse_exper_conf.py "$experDescPath" "$((ivar-1))" "$tmpConf"`

  if [ "stat" = "#ERR" ] ; then
    echo "Failed to get entry $ivar from experiment config $experDescPath"
    exit 1
  elif [ "$stat" = "#END" ] ; then # out of range
    break
  else

    testPart=`grepFileForVal "$tmpConf" $TEST_PART_PARAM`
    experSubdir=`grepFileForVal "$tmpConf" $EXPER_SUBDIR_PARAM`
    if [ "$testPart" = "" ] ; then
      testPart=$defaultTestPart
    fi
    if [ "$testPart" = "" ] ; then
      echo "Specify $TEST_PART_PARAM in config # $ivar or set the script parameter -test_part"
      exit 1
    fi
    if [ "$experSubdir" = "" ] ; then
      echo "Missing $EXPER_SUBDIR_PARAM config # $ivar"
      exit 1
    fi

    experDirBase=`getExperDirBase "$collectSubdir" "$testPart" "$experSubdir"`
    if [ ! -d "$experDirBase" ] ; then
      echo "Directory doesn't exist: $experDirBase"
      exit 1
    fi
    pd=$PWD
    cd $experDirBase/$REP_SUBDIR

    # Let's read timings
    query_time="N/A"
    stat_file="$STAT_FILE"
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

    for f in `ls -tr out_${fltN}.rep` ; do
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


rm "$tmpConf"