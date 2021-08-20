#!/bin/bash
. scripts/common_proc.sh
. scripts/config.sh

currDir=$PWD

# Be careful when you rename parameter names, certain things
# are used in Python scripts

# Notes on the number candidates generated
# 1. if the user doesn't specify the # of candidates explicitly (candQty in JSON), it's set to the maximum # of answers to produce
# 2. if there's an intermediate re-ranker it re-ranks all retrieved candidates. However, if the number of candidates
#    is smaller than the maximum number of entries to produce (defined by test_cand_qty_list), then the list of
#    candidates is truncated *AFTER* being resorted using an intermediate re-ranker.

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "DEFAULT_NUM_RAND_RESTART"
checkVarNonEmpty "DEFAULT_TRAIN_SUBDIR"
checkVarNonEmpty "DEFAULT_NUM_TREES"
checkVarNonEmpty "DEFAULT_METRIC_TYPE"
checkVarNonEmpty "CAND_PROV_LUCENE"
checkVarNonEmpty "DEV1_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

checkVarNonEmpty "TEST_PART_PARAM"
checkVarNonEmpty "TRAIN_PART_PARAM"
checkVarNonEmpty "EXPER_SUBDIR_PARAM"
checkVarNonEmpty "TEST_ONLY_PARAM"
checkVarNonEmpty "TRAIN_ONLY_PARAM"

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

numRandRestart=$DEFAULT_NUM_RAND_RESTART
numTrees=$DEFAULT_NUM_TREES
metricType=$DEFAULT_METRIC_TYPE

useLMARTParam=""

checkVarNonEmpty "DEFAULT_TRAIN_CAND_QTY"
checkVarNonEmpty "DEFAULT_TEST_CAND_QTY_LIST"

trainCandQty=$DEFAULT_TRAIN_CAND_QTY
testCandQtyList=$DEFAULT_TEST_CAND_QTY_LIST

debugPrint="0"
globalParams=""
defaultModelFinal=""

useSeparateShell=1
parallelExperQty=1
numCpuCores=""

threadQty=""

defaultTestPart=""
defaultTrainPart="$DEFAULT_TRAIN_SUBDIR"
addExperSubdir=""
clean="0"

function usage {
  msg=$1
  echo $msg
  cat <<EOF
Usage: <collection> <feature desc. file relative to collection root> [additional options]
Additional options:
  -max_num_query_test     max. # of test queries
  -test_part              default test set, e.g., $DEV1_SUBDIR (can be redefined in the experimental config)
  -clean                  remove an existing experimental directory
  -model_final            final-stage model (relative to the collection root)
  -train_part             default train set, e.g., $DEFAULT_TRAIN_SUBDIR (can be redefined in the experimental config)
  -train_cand_qty         # of candidates for training (default $trainCandQty)
  -max_final_rerank_qty   max. # of records to re-rank using the final re-ranker
  -test_cand_qty_list     a comma-separate list of # candidates for testing (default $testCandQtyList)
  -metric_type            evaluation metric (default $metricType)
  -add_exper_subdir       additional experimental sub-directory
  -skip_eval              skip/disable evaluation, just produce TREC runs
  -max_num_query_train    max. # of training queries
  -num_cpu_cores          # of available CPU cores
  -thread_qty             # of threads
  -parallel_exper_qty     # of experiments to run in parallel (default $parallelExperQty)
  -reuse_feat             reuse previously generated features
  -delete_trec_runs       delete TREC run files
  -model1_subdir          Model1 sub-directory (relative to $DERIVED_DATA_SUBDIR)
  -no_separate_shell      use this for debug purposes only
  -debug_print            print every executed command
EOF
}

globalParams=""

while [ $# -ne 0 ] ; do
  optValue=""
  opt=""
  if [[ "$1" = -* ]] ; then
    optName="$1"
    optValue="$2"
    if [ "$optName" = "-reuse_feat" ] ; then
      globalParams+=" $optName"
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-debug_print" ] ; then
      globalParams+=" $optName"
      set -x
      debuPrint=1
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-delete_trec_runs" ] ; then
      globalParams+=" $optName"
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-skip_eval" ] ; then
      globalParams+=" $optName"
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-clean" ] ; then
      clean="1"
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-no_separate_shell" ] ; then
      useSeparateShell=0
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-h" -o "$optName" = "-help" ] ; then
      usage
      exit 1
    else
      optValue="$2"
      opt="$1 $2"
      if [ "$optValue" = "" ] ; then
        echo "Option $optName requires an argument." >&2
        exit 1
      fi
      shift 2 # option with an argument: shift by two
      case $optName in
        -thread_qty)
          threadQty=$optValue
          globalParams+=" $opt"
          ;;
        -num_cpu_cores)
          numCpuCores=$optValue
          globalParams+=" $opt"
          ;;
        -add_exper_subdir)
          addExperSubDir=$optValue
          ;;
        -model1_subdir)
          globalParams+=" $optName \"$optValue\""
          ;;
        -train_cand_qty)
          globalParams+=" $opt"
          ;;
        -test_cand_qty_list)
          globalParams+=" $opt"
          ;;
        -parallel_exper_qty)
          parallelExperQty="$optValue"
          ;;
        -metric_type)
          globalParams+=" $opt"
          ;;
        -max_num_query_train)
          globalParams+=" $opt"
          ;;
        -max_num_query_test)
          globalParams+=" $opt"
          ;;
        -test_part)
          defaultTestPart=$optValue
          ;;
        -train_part)
          defaultTrainPart=$optValue
          ;;
        -model_final)
          defaultModelFinal=$optValue
          ;;
        -max_final_rerank_qty)
          globalParams+=" $opt"
          ;;
        *)
          echo "Invalid option: $optName" >&2
          exit 1
          ;;
      esac
    fi
  else
    posArgs=(${posArgs[*]} $1)
    shift 1
  fi
done


collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  usage "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi


collectSubdir="$COLLECT_ROOT/$collect"

featDescFile=${posArgs[1]}
if [ "$featDescFile" = "" ] ; then
  usage "Specify a feature description file *RELATIVE* to $collectSubdir (2d arg)"
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


checkVarNonEmpty "featDescFile"
experDescPath=$collectSubdir/$featDescFile
if [ ! -f "$experDescPath" ] ; then
  echo "Not a file '$experDescPath'"
  exit 1
fi

nTotal=0
nRunning=0

echo "$SEP_DEBUG_LINE"

echo "Experiment descriptor file:                                 $experDescPath"
echo "Default test set:                                           $defaultTestPart"
echo "Number of parallel experiments:                             $parallelExperQty"
echo "Number of threads in feature extractors/query applications: $threadQty"

echo "$SEP_DEBUG_LINE"

tmpConf=`mktemp`

# Mapping between JSON field names and corresponding script parameters
jsonParamMap=(
  # candidate provider options
  cand_prov           candProv          # candidate provider type
  cand_prov_add_conf  candProvAddConf   # optional additional config
  cand_prov_uri       candProvURI       # candidate provider file location or IP address
  cand_prov_qty       candProvQty       # number of candidates (can be overriden by -test_cand_qty_list)
  # TREC run id
  run_id runId

  # Feaure extractor configuration files
  extr_type_final     extrTypeFinal    # an optional final re-ranker config
  extr_type_interm    extrTypeInterm   # an optional intermediate re-ranker config

  # Model files
  model_interm  modelInterm   # optional model for the intermediate re-ranker
  model_final   modelFinal    # optional model for the final re-ranker

  # Learning-to-rank (LETOR) parameters
  use_lmart         useLMART # if true we use LambdaMART
  num_rand_restart  numRandRestart   # number of random restarts in coordinate ascent
  num_trees         numTrees   # number of trees for LambdaMart
)

childPIDs=()
nrun=0
nfail=0
for ((ivar=1;;++ivar)) ; do

  stat=`scripts/exper/parse_exper_conf.py "$experDescPath" "$((ivar-1))" "$tmpConf"`

  if [ "$?" != "0" ] ; then
    echo "Failed to parse configuration $ivar from the file $experDescPath"
    exit 1
  fi

  if [ "stat" = "#ERR" ] ; then
    echo "Failed to get entry $ivar from experiment config $experDescPath"
    exit 1
  elif [ "$stat" = "#END" ] ; then # out of range
    break
  else

    echo "Parsed experiment parameters:"
    cat "$tmpConf"
    echo "========================================"

    testPart=`$currDir/scripts/grep_file_for_val.py "$tmpConf" $TEST_PART_PARAM`
    trainPart=`$currDir/scripts/grep_file_for_val.py "$tmpConf" $TRAIN_PART_PARAM`

    if [ "$trainPart" = "" ] ; then
      trainPart="$defaultTrainPart"
    fi
    # here trainPart must be defined
    checkVarNonEmpty "trainPart"

    experSubdir=`$currDir/scripts/grep_file_for_val.py "$tmpConf" $EXPER_SUBDIR_PARAM`
    testOnly=`$currDir/scripts/grep_file_for_val.py "$tmpConf" $TEST_ONLY_PARAM`
    trainOnly=`$currDir/scripts/grep_file_for_val.py "$tmpConf" $TRAIN_ONLY_PARAM`

    if [ "$testOnly" = "1" -a "$trainOnly" = "1" ] ; then
      echo "Incompatible options, you cannot simulataneously set $TEST_PART_PARAM and $TRAIN_ONLY_PARAM to 1!"
      exit 1
    fi

    if [ "$trainOnly" != "1" ] ; then
      if [ "$testPart" = "" ] ; then
        testPart=$defaultTestPart
      fi
      if [ "$testPart" = "" ] ; then
        echo "Specify $TEST_PART_PARAM in config # $ivar, set the script parameter -test_part, or use option $TRAIN_ONLY_PARAM to disable testing"
        exit 1
      fi
    fi

    if [ "$experSubdir" = "" ] ; then
      echo "Missing $EXPER_SUBDIR_PARAM config # $ivar"
      exit 1
    fi

    if [ "$trainOnly" = "1" ] ; then
      experPart="$trainPart"
    else
      experPart="$testPart"
    fi

    # Each experiment should run in its own sub-directory
    experDirBase=`getExperDirBase "$collectSubdir" "$experPart" "$experSubdir"`
    if [ "$?" != "0" ] ; then
      echo "Failed to obtain an experimental directory!"
      exit 1
    fi

    if [ "$addExperSubDir" != "" ] ; then
      experDirBase="$experDirBase/$addExperSubDir"
    fi

    if [ -d "$experDirBase" ] ; then
      # The helper experimental script will clean it up
      if [ "$clean" = "1" ] ; then
        echo "Experimental directory already exists (removing contents): $experDirBase"
        # Be very careful with this sort of deletions,
        # double-check it's not empty again, otherwise we might try to delete
        # files at the root file-system directory
        if [ "$experDirBase" != "" ] ; then
          echo "Cleaning the experimental directory: $experDirBase"
          rm -rf $experDirBase/*
        else
          echo "Bug: empty experDirBase here!"
          exit 1
        fi
      else
        echo "Experimental directory already exists (ignoring): $experDirBase"
        continue
      fi
    else
      mkdir -p "$experDirBase"
    fi

    singleConfParams="-thread_qty $threadQty"

    if [ "$testOnly" = "1" ] ; then
      singleConfParams+=" -test_only"
    fi
    if [ "$trainOnly" = "1" ] ; then
      singleConfParams+=" -train_only"
    fi

    for ((i=0;i<${#jsonParamMap[*]};i+=2)) ; do
      paramName=${jsonParamMap[$i]}
      jsonParamName=${jsonParamMap[$(($i+1))]}
      paramVal=`$currDir/scripts/grep_file_for_val.py "$tmpConf" "$jsonParamName"`
      # useLMART requires special treatment
      if [ "$paramName" = "use_lmart" -a "$paramVal" != "1" ] ; then
        paramVal=""
      fi
      # Overriding the value of the final model
      if [ "$paramName" = "model_final" -a "$paramVal" = "" ] ; then
        paramVal="$defaultModelFinal"
      fi

      if [ "$paramVal" != "" ] ; then
        singleConfParams+=" -${paramName} \"$paramVal\""
      fi
    done


    experDirBaseRelative=`getExperDirBase "" "$experPart" "$experSubdir"`
    if [ "$?" != "0" ] ; then
      echo "Failed to obtain a relative experimental directory!"
      exit 1
    fi

# Don't quote $globalParams or any other "*Param*
  cmd=`cat <<EOF
        scripts/exper/run_one_experiment.sh \
            "$collect" \
            "$experDirBaseRelative" \
            "$testPart" \
            -train_part $trainPart \
            $globalParams $singleConfParams
EOF
`
    logFileName="$experDirBase/exper.log"
    if [ "$useSeparateShell" = "1" ] ; then
      bash -c "$cmd"  &> "$logFileName" &

      pid=$!
      childPIDs+=($pid)
      echo "Started a process $pid, working dir: $experDirBase"
      #echo "Command run: $cmd"
      echo "Process log file: $logFileName"
    else
      echo "Starting a process, working dir: $experDirBase"
      #echo "Command run: $cmd"
      echo "Process log file: $logFileName"
      bash -c "$cmd"  2>&1 | tee "$logFileName"
      checkPipe
    fi
    nRunning=$(($nRunning+1))
    nrun=$(($nrun+1))

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
if [ "$nfail" -gt "0" ] ; then
  exit 1
fi

