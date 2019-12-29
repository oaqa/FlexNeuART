#!/bin/bash
. scripts/common_proc.sh
. scripts/config.sh


checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FEAT_EXPER_SUBDIR"
checkVarNonEmpty "EXPER_DESC_SUBDIR"
checkVarNonEmpty "DEFAULT_NUM_RAND_RESTART"
checkVarNonEmpty "DEFAULT_NUM_TREES"
checkVarNonEmpty "DEFAULT_METRIC_TYPE"
checkVarNonEmpty "NO_FEAT_EXTRACTOR"

numRandRestart=$DEFAULT_NUM_RAND_RESTART
numTrees=$DEFAULT_NUM_TREES
metricType=$DEFAULT_METRIC_TYPE

maxQueryQtyTrain=""
maxQueryQtyTest=""
useLMARTParam=""

checkVarNonEmpty "DEFAULT_TRAIN_CAND_QTY"
checkVarNonEmpty "DEFAULT_TEST_CAND_QTY_LIST"

trainCandQty=$DEFAULT_TRAIN_CAND_QTY
testCandQtyList=$DEFAULT_TEST_CAND_QTY_LIST

noRegenFeatParam=""

deleteTrecRunsParam="" # Shouldn't delete these runs by default
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
  -parallel_exper_qty   # of experiments to run in parallel (default $parallelExperQty)
  -delete_trec_runs     delete TREC run files
  -no_separate_shell    use this for debug purposes only
  -no_regen_feat        do not regenerate features
  -thread_qty           # of threads
  -use_lmart            use Lambda-MART instead of coordinate ascent
  -num_trees            # of trees in Lambda-MART (default $numTrees)
  -num_rand_restart     # of random restart for coordinate ascent (default $numRandRestart)
  -train_cand_qty       # of candidates for training (default $trainCandQty)
  -test_cand_qty_list   a comma-separate list of # candidates for testing (default $testCandQtyList)
  -metric_type          evaluation metric (default $metricType)
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
      # option without an argument
      shift 1
    elif [ "$OPT_NAME" = "-no_regen_feat" ] ; then
      noRegenFeatParam="$OPT_NAME"
      shift 1
    elif [ "$OPT_NAME" = "-no_separate_shell" ] ; then
      useSeparateShell=0
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
      shift 2
      case $OPT_NAME in
        -thread_qty)
          threadQty=$OPT_VALUE
          ;;
        -num_rand_restart)
          numRandRestart=$OPT_VALUE
          ;;
        -delete_trec_runs)
          deleteTrecRunsParam=$OPT
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
          maxQueryQtyTrain=$OPT_VALUE
          ;;
        -max_num_query_test)
          maxQueryQtyTest=$OPT_VALUE
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

experDescLoc="$COLLECT_ROOT/$collect/$EXPER_DESC_SUBDIR"

checkVarNonEmpty "featDescFile"
experDescPath=$experDescLoc/$featDescFile
if [ ! -f "$experDescPath" ] ; then
  echo "Not a file '$experDescPath'"
  exit 1
fi

experDir="$COLLECT_ROOT/$collect/$FEAT_EXPER_SUBDIR"
if [ ! -d "$experDir" ] ; then
  mkdir -p $experDir
  if [ "$?" != "0" ] ; then
    echo "Cannot create '$experDir'"
    exit 1
  fi
fi

nTotal=0
nRunning=0

echo "Experiment descriptor file:                                 $experDescPath"
echo "Number of parallel experiments:                             $parallelExperQty"
echo "Number of threads in feature extractors/query applications: $threadQty"

maxQueryQtyParams=""
if [ "$maxQueryQtyTrain" != "" ] ; then
  maxQueryQtyParams="$maxQueryQtyParams -max_num_query_train $maxQueryQtyTrain "
fi
if [ "$maxQueryQtyTest" != "" ] ; then
  maxQueryQtyParams="$maxQueryQtyParams -max_num_query_test $maxQueryQtyTest "
fi

# Don't quote $maxQueryQtyParams and other *Param*

n=`wc -l "$experDescPath"|awk '{print $1}'`
n=$(($n+1))
childPIDs=()
nrun=0
nfail=0
for ((ivar=1;ivar<$n;++ivar))
  do
    line=$(head -$ivar "$experDescPath"|tail -1)
    line=$(removeComment "$line")
    if [ "$line" !=  "" ]
    then
      extrConfigFile=`echo $line|awk '{print $1}'`
      if [ "$extrConfigFile" = "" ] ; then
        echo "Missing feature-extractor config file (1st field) in line $line, file $experDescPath"
        exit 1
      fi
      if [ "$extrConfigFile" = "$NO_FEAT_EXTRACTOR" ] ; then
        extrConfigPath=$NO_FEAT_EXTRACTOR
      else
        extrConfigPath="$experDescLoc/$extrConfigFile"
        if [ ! -f "$extrConfigPath" -a ! f ] ; then
          echo "Missing feature-extractor configuration file: $extrConfigPath"
          exit 1
        fi
      fi

      testSet=`echo $line|awk '{print $2}'`
      if [ "$testSet" = "" ] ; then
        echo "Missing test set (e.g., dev1) (2d field) in line $line, file $experDescPath"
        exit 1
      fi
      experSubdir=`echo $line|awk '{print $3}'`
      if [ "$testSet" = "" ] ; then
        echo "Missing experimental sub-dir (3d field) in line $line, file $experDescPath"
        exit 1
      fi
      # Each experiment should run in its separate sub-directory
      experDirUnique=$(getExperDirUnique "$experDir" "$testSet" "$experSubdir")
      if [ ! -d "$experDirUnique" ] ; then
        mkdir -p "$experDirUnique"
        if [ "$?" != "0" ] ; then
          echo "Failed to create $experDirUnique"
          exit 1
        fi
      fi


cmd=`cat <<EOF
    scripts/exper/run_one_feature_exper.sh \
      $collect "$experDirUnique" \
      "$testSet" \
      $trainCandQty \
      $testCandQtyList \
      -extr_type "$extrConfigPath" \
      $maxQueryQtyParams \
      -thread_qty $threadQty \
      -num_rand_restart $numRandRestart \
      -num_trees $numTrees \
      -metric_type $metricType \
      $deleteTrecRunsParam \
      $useLMARTParam $noRegenFeatParam &> $experDirUnique/exper.log
EOF
`
      if [ "$useSeparateShell" = "1" ] ; then
        bash -c "$cmd" &

        pid=$!
        childPIDs+=($pid)
        echo "Started a process $pid, working dir: $experDirUnique"
        nRunning=$(($nRunning+1))
        nrun=$(($nrun+1))
      else
        echo "Starting a process, working dir: $experDirUnique"
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
echo "============================================"
echo "$nrun experiments executed"
echo "$nfail experiments failed"
if [ "$nfail" -gt "0" ] ; then
  echo "Check the log in working directories!!!"
fi
echo "============================================"

