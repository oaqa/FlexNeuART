#!/bin/bash
. scripts/common_proc.sh
. scripts/config.sh


checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FEAT_EXPER_SUBDIR"
checkVarNonEmpty "EXPER_DESC_SUBDIR"
checkVarNonEmpty "DEFAULT_NUM_RAND_RESTART"
checkVarNonEmpty "DEFAULT_NUM_TREES"
checkVarNonEmpty "DEFAULT_METRIC_TYPE"
checkVarNonEmpty "CAND_PROV_LUCENE"
checkVarNonEmpty "EXPER_SUBDIR"

numRandRestart=$DEFAULT_NUM_RAND_RESTART
numTrees=$DEFAULT_NUM_TREES
metricType=$DEFAULT_METRIC_TYPE
""
useLMARTParam=""

checkVarNonEmpty "DEFAULT_TRAIN_CAND_QTY"
checkVarNonEmpty "DEFAULT_TEST_CAND_QTY_LIST"

trainCandQty=$DEFAULT_TRAIN_CAND_QTY
testCandQtyList=$DEFAULT_TEST_CAND_QTY_LIST


globalParams=""

useSeparateShell=1
parallelExperQty=1
numCpuCores=""

threadQty=""

function usage {
  msg=$1
  echo $msg
  cat <<EOF
Usage: <collection> <feature desc. file in subdir $EXPER_DESC_SUBDIR> [additional options]
Additional options:
  -max_num_query_test   max. # of test queries
  -num_cpu_cores        # of available CPU cores
  -thread_qty           # of threads
  -parallel_exper_qty   # of experiments to run in parallel (default $parallelExperQty)
  -delete_trec_runs     delete TREC run files
  -no_separate_shell    use this for debug purposes only
  -reuse_feat           reuse previously generated features
  -use_lmart            use Lambda-MART instead of coordinate ascent
  -num_trees            # of trees in Lambda-MART (default $numTrees)
  -num_rand_restart     # of random restart for coordinate ascent (default $numRandRestart)
  -train_cand_qty       # of candidates for training (default $trainCandQty)
  -test_cand_qty_list   a comma-separate list of # candidates for testing (default $testCandQtyList)
  -metric_type          evaluation metric (default $metricType)
  -skip_eval            skip/disable evaluation, just produce TREC runs
  -test_model_results   additionally test model performance on the training set
  -max_num_query_train  max. # of training queries
EOF
}

while [ $# -ne 0 ] ; do
  OPT_VALUE=""
  OPT=""
  echo $1|grep "^-" >/dev/null
  if [ $? = 0 ] ; then
    OPT_NAME="$1"
    OPT_VALUE="$2"
    if [ "$OPT_NAME" = "-use_lmart" ] ; then
      useLMARTParam="-use_lmart"
      # option without an argument shift by 1
      shift 1
    elif [ "$OPT_NAME" = "-reuse_feat" ] ; then
      noRegenFeatParam="$OPT_NAME"
      # option without an argument shift by 1
      shift 1
    elif [ "$OPT_NAME" = "-test_model_results" ] ; then
      noRegenFeatParam="$OPT_NAME"
      # option without an argument shift by 1
      shift 1
    elif [ "$OPT_NAME" = "-delete_trec_runs" ] ; then
      deleteTrecRunsParam=$OPT_NAME
      # option without an argument shift by 1
      shift 1
    elif [ "$OPT_NAME" = "-skip_eval" ] ; then
      skipEvalParam=$OPT_NAME
      # option without an argument shift by 1
      shift 1
    elif [ "$OPT_NAME" = "-no_separate_shell" ] ; then
      useSeparateShell=0
      # option without an argument shift by 1
      shift 1
    elif [ "$OPT_NAME" = "-h" -o "$OPT_NAME" = "-help" ] ; then
      usage
      exit 1
    else
      OPT_VALUE="$2"
      OPT="$1 $2"
      if [ "$OPT_VALUE" = "" ] ; then
        echo "Option $OPT_NAME requires an argument." >&2
        exit 1
      fi
      shift 2 # option with an argument: shift by two
      case $OPT_NAME in
        -thread_qty)
          threadQty=$OPT_VALUE
          ;;
        -num_rand_restart)
          numRandRestart=$OPT_VALUE
          ;;
        -num_cpu_cores)
          numCpuCores=$OPT_VALUE
          ;;
        -train_cand_qty)
          trainCandQty=$OPT_VALUE
          ;;
        -test_cand_qty_list)
          testCandQtyList=$OPT_VALUE
          ;;
        -num_trees)
          numTrees=$OPT_VALUE
          ;;
        -parallel_exper_qty)
          parallelExperQty=$OPT_VALUE
          ;;
        -metric_type)
          metricType=$OPT_VALUE
          ;;
        -max_num_query_train)
          maxQueryQtyParams=" $maxQueryQtyParams $OPT"
          ;;
        -max_num_query_test)
          maxQueryQtyParams=" $maxQueryQtyParams $OPT"
          ;;
        *)
          echo "Invalid option: $OPT_NAME" >&2
          exit 1
          ;;
      esac
    fi


  else
    POS_ARGS=(${POS_ARGS[*]} $1)
    shift 1
  fi
done


collect=${POS_ARGS[0]}
if [ "$collect" = "" ] ; then
  usage "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

featDescFile=${POS_ARGS[1]}
if [ "$featDescFile" = "" ] ; then
  usage "Specify a feature description file (2d arg)"
  exit 1
fi

if [ "$numCpuCores" = "" ] ; then
  numCpuCores=`getNumCpuCores`
fi
if [ "$numCpuCores" = "" ] ; then
  usage "Cannot guess # of CPU cores, please, provide # of CPUs cores"
  exit 1
fi

if [ "$threadQty" = "" ] ; then
  threadQty=$(($numCpuCores/$parallelExperQty))
  if [ "$threadQty" = "0" ] ; then
    threadQty=1
  fi
fi

echo "The number of CPU cores:      $numCpuCores"
echo "The number of || experiments: $parallelExperQty"
echo "The number of threads:        $threadQty"

collectRoot="$COLLECT_ROOT/$collect"
experDescLoc="$collectRoot/$EXPER_DESC_SUBDIR"

checkVarNonEmpty "featDescFile"
experDescPath=$experDescLoc/$featDescFile
if [ ! -f "$experDescPath" ] ; then
  echo "Not a file '$experDescPath'"
  exit 1
fi

nTotal=0
nRunning=0

echo "Experiment descriptor file:                                 $experDescPath"
echo "Number of parallel experiments:                             $parallelExperQty"
echo "Number of threads in feature extractors/query applications: $threadQty"

tmpConf=`mktemp`

childPIDs=()
nrun=0
nfail=0
for ((ivar=1;;++ivar))
  do
    scripts/exper/parse_exper_conf.py "$experDescPath" $ivar "$tmpConf"
    $cont=`cat $tmpConf`


    if [ "$cont" =  "" ]
    then
      echo "Failed to get entry $ivar from experiment config $experDescPath"
      exit 1
    elif [ "$cont" != "#OOR" ] ; then # not out of range

      parsedConf=`echo $line|scripts/exper/parse_oneline_conf.py`

      confParams=""


      # Each experiment should run in its own sub-directory
      getExperDirBase=$(getExperDirBase "$COLLECTION_ROOT/$EXPER_SUBDIR" "$testSet" "$experSubdir")

# Don't quote $globalParams or any other "*Param*
    cmd=`cat <<EOF
          scripts/exper/run_one_feature_exper.sh \
              "$collect" \
              "$getExperDirBase" \
              "$testSet" \
              $globalParams $confParams &> $getExperDir/exper.log
EOF
`
      if [ "$useSeparateShell" = "1" ] ; then
        bash -c "$cmd" &

        pid=$!
        childPIDs+=($pid)
        echo "Started a process $pid, working dir: $getExperDir"
        nRunning=$(($nRunning+1))
        nrun=$(($nrun+1))
      else
        echo "Starting a process, working dir: $getExperDir"
        bash -c "$cmd"
      fi

    fi
    if [ "$nRunning" -ge $parallelExperQty ] ; then
      waitChildren ${childPIDs[*]}
      childPIDs=()
      nRunning=0
    fi
  done
waitChildren ${childPIDs[*]}
echo "$SEP_DEBUG_LINE"
echo "$nrun experiments executed"
echo "$nfail experiments failed"
if [ "$nfail" -gt "0" ] ; then
  echo "Check the log in working directories!!!"
fi
echo "$SEP_DEBUG_LINE"
rm "$tmpConf"

