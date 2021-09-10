#!/bin/bash
. ./common_proc.sh
. ./config.sh

checkVarNonEmpty "REP_SUBDIR"
checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "TEST_PART_PARAM"
checkVarNonEmpty "TEST_PART_PARAM"
checkVarNonEmpty "STAT_FILE"

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

currDir=$PWD

boolOpts=(\
"h" "help" "print help" \
"debug_print" "debug" "print every executed command" \
)

paramOpts=(\
"test_part" "defaultTestPart" "default test set, e.g., dev1" \
"flt_cand_qty" "fltN" "a filter to include runs only with this # of candidates" \
"print_best_metr" "printBestMetr" "print only best result according to the specified metric" \
)

parseArguments $@

usageMain="<collection> <feature desc. file relative to collection root> <output file>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
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

outFile="${posArgs[2]}"
if [ "$outFile" = "" ] ; then
    genUsage "$usageMain" "Missing output file!"
    exit 1
fi
# Must make it absolute path
outFile="$PWD/$outFile"

if [ "$fltN" = "" ] ; then
  if [ "$printBestMetr" != "" ] ; then
    genUsage "$usageMain" "Finding for best-parameter config. requires you specify the filter for # of candidates!"
    exit 1
  fi
  fltN="*"
  echo "Including runs for all candidate record numbers"
else
  echo "Including only runs that generated $fltN candidate records"
fi

if [ "$debug" = "1" ] ; then
  set -x
fi

tmpConf=`mktemp`

metrListGrep=(\
"ndcg@10" \
"ndcg@20" \
"ndcg@100" \
"err@20" "p20" "map" "mrr" "recall")

metrListPrint=(\
"NDCG@10" \
"NDCG@20" \
"NDCG@100" \
"ERR@20" "P@20" "MAP" "R-RANK" "RECALL")

header="test_part\texper_subdir\ttop_k\tquery_qty"

for ((i=0;i<${#metrListPrint[*]};i++)) ; do
  header+="\t${metrListPrint[$i]}"
done

echo -e "$header" > "$outFile"

ivar=0

bestVal="1e-100"
bestSubdir=""

for ((ivar=1;;++ivar)) ; do

  stat=`./exper/parse_exper_conf.py "$experDescPath" "$((ivar-1))" "$tmpConf"`

  if [ "stat" = "#ERR" ] ; then
    echo "Failed to get entry $ivar from experiment config $experDescPath"
    exit 1
  elif [ "$stat" = "#END" ] ; then # out of range
    break
  else

    testPart=`$currDir/grep_file_for_val.py "$tmpConf" $TEST_PART_PARAM`
    experSubdir=`$currDir/grep_file_for_val.py "$tmpConf" $EXPER_SUBDIR_PARAM`
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
      query_qty=`$currDir/grep_file_for_val.py "$f" "# of queries"`

      row="$testPart\t$experSubdir\t$top_k\t$query_qty"
      for ((i=0;i<${#metrListGrep[*]};i++)) ; do
        metrGrepName=${metrListGrep[$i]}
        metrPrintName=${metrListPrint[$i]}
        val=`$currDir/grep_file_for_val.py "$f" "$metrGrepName" "1"`
        if [ "$metrGrepName" = "$printBestMetr" -o "$metrPrintName" = "$printBestMetr" ] ; then
          cmp=`isGreater "$val" "$bestVal"`
          if [ "$bestSubdir" = "" -o "$cmp" = "1" ] ; then
            bestSubdir=$experSubdir
            bestVal=$val
          fi
        fi
        row+="\t$val"
      done

      echo -e "$row" >> "$outFile"
    done
    cd $pd
    check "cd $pd"
  fi
done

if [ "$printBestMetr" != "" ] ; then
  echo "Best results for metric $printBestMetr:"
  echo "Value: $bestVal"
  echo "Result sub-dir: $bestSubdir"
fi


rm "$tmpConf"
