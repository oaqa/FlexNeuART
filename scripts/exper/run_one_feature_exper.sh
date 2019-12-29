#!/bin/bash
source scripts/common_proc.sh
source scripts/config.sh

# These flags are mostly for debug purposes
regenFeat="1"
recompModel="1"
rerunLucene="1"
testModelResults="0" # Use this for debug purposes only
# end of debug flags

POS_ARGS=()

threadQty=1
maxQueryQtyTrain=""
maxQueryQtyTest=""
maxQueryQtyTrainParam=""
maxQueryQtyTestParam=""

deleteTrecRuns=0 # Shouldn't delete these runs by default

useLMART="0"

checkVarNonEmpty "DEFAULT_NUM_RAND_RESTART"
checkVarNonEmpty "DEFAULT_NUM_TREES"
checkVarNonEmpty "DEFAULT_METRIC_TYPE"
checkVarNonEmtpy "NO_FEAT_EXTRACTOR"

numRandRestart=$DEFAULT_NUM_RAND_RESTART
numTrees=$DEFAULT_NUM_TREES
metricType=$DEFAULT_METRIC_TYPE

extrType=""
trainCandQty=""

while [ $# -ne 0 ] ; do
  OPT_VALUE=""
  OPT=""
  echo $1|grep "^-" >/dev/null
  if [ $? = 0 ] ; then
    OPT_NAME="$1"
    if [ "$OPT_NAME" = "-use_lmart" ] ; then
      useLMART="1"
      # option without an argument
      shift 1
    elif [ "$OPT_NAME" = "-no_regen_feat" ] ; then
      regenFeat="0"
      shift 1
    elif [ "$OPT_NAME" = "-delete_trec_runs" ] ; then
      deleteTrecRuns="1"
      shift 1
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
        -extr_type)
          extrType=$OPT_VALUE
          ;;
        -num_trees)
          numTrees=$OPT_VALUE
          ;;
        -metric_type)
          metricType=$OPT_VALUE
          ;;
        -max_num_query_train)
          maxQueryQtyTrain=$OPT_VALUE
          maxQueryQtyTrainParam=" -max_num_query $maxQueryQtyTrain"
          ;;
        -max_num_query_test)
          maxQueryQtyTest=$OPT_VALUE
          maxQueryQtyTestParam=" -max_num_query $maxQueryQtyTest"
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
  echo "Specify a sub-collection, e.g., squad (1st arg)"
  exit 1
fi

experDirBase=${POS_ARGS[1]}
if [ "$experDirBase" = "" ] ; then
  echo "Specify a working directory (2d arg)!"
  exit 1
fi

testPart=${POS_ARGS[2]}

if [ "$testPart" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (2d arg)"
  exit 1 
fi

trainCandQty=${POS_ARGS[3]}
if [ "$trainCandQty" = "" ] ; then
  echo "Specify a # of candidate records for training (3d arg)!"
  exit 1
fi

testCandQtyList=${POS_ARGS[4]}
if [ "$testCandQtyList" = "" ] ; then
  echo "Specify a comma-separated list of candidate record # for testing (retrieved for each query) (4th arg)!"
  exit 1
fi

testCandQtyListSpaceSep=`echo $testCandQtyList|sed 's/,/ /g'`


# Do it only after argument parsing
set -eo pipefail

experDir="$experDirBase/exper"
trecRunDir="$experDirBase/trec_runs"
reportDir="$experDirBase/rep"

mkdir -p "$experDir"
check "mkdir -p $experDir"
mkdir -p "$trecRunDir"
check "mkdir -p $trecRunDir"
mkdir -p "$reportDir"
check "mkdir -p $reportDir"

echo "Deleting old reports from the directory: ${reportDir}"
rm -f ${reportDir}/*
check "rm -f ${reportDir}/*"

checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "GIZA_ITER_QTY"
checkVarNonEmpty "TRAIN_SUBDIR"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"
checkVarNonEmpty "LUCENE_CACHE_SUBDIR"
checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FAKE_RUN_ID"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
embedDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR/"
gizaRootDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$GIZA_SUBDIR"

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[3]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi


cacheDir="$COLLECT_ROOT/$collect/$LUCENE_CACHE_SUBDIR/"

if [ ! -d "$cacheDir" ] ; then
  mkdir -p "$cacheDir"
  check "mkdir -p $cacheDir"
fi

for part in "$TRAIN_SUBDIR" "$testPart" ; do
  if [ ! -d "$cacheDir/$part" ] ; then
   mkdir -p "$cacheDir/$part"
   check "mkdir -p $cacheDir/$part"
  fi
done

if [ "$maxQueryQtyTrain" = "" ] ; then
  cacheFileTrain="$cacheDir/$TRAIN_SUBDIR/all_queries_$testCandQtyList"
else
  cacheFileTrain="$cacheDir/$TRAIN_SUBDIR/max_query_qty=${maxQueryQtyTrain}_$testCandQtyList"
fi

if [ "$maxQueryQtyTest" = "" ] ; then
  cacheFileTest="$cacheDir/$testPart/all_queries_$testCandQtyList"
else
  cacheFileTest="$cacheDir/$testPart/max_query_qty=${maxQueryQtyTrain}_$testCandQtyList"
fi

URI="$COLLECT_ROOT/$collect/$LUCENE_INDEX_SUBDIR"

echo "==============================================="
echo "Using $testPart for testing!"
echo "Experiment directory:           $experDir"
echo "QREL file:                      $QREL_FILE"
echo "Directory with TREC-style runs: $trecRunDir"
echo "Report directory:               $reportDir"
echo "Query cache file (for training):$cacheFileTrain"
echo "Query cache file (for testing): $cacheFileTest"
echo "==============================================="
echo "Maximum number of test queries parameter: $maxQueryQtyTrainParam"
echo "Number of candidate records for training: $trainCandQty"
echo "Maximum number of train candidates:       $maxQueryQtyTestParam"
echo "A list for the number of test candidates: $testCandQtyList"
echo "==============================================="
echo "Data directory:          $inputDataDir"
echo "Data file name:          $queryFileName"
echo "Forward index directory: $fwdIndexDir"
echo "Lucene index directory:  $URI"
echo "Embedding directory:     $embedDir"
echo "GIZA root directory:     $gizaRootDir"
echo "==============================================="
echo "Use LAMBDA MART:            $useLMART"
echo "Learning metric:            $metricType"
if [ "$useLMART" = "1" ] ; then
  echo "Number of trees:            $numTrees"
else
  echo "Number of random restarts:  $numRandRestart"
fi
echo "==============================================="


outPrefTrain="out_${collect}_${TRAIN_SUBDIR}"
outPrefTest="out_${collect}_${testPart}"
fullOutPrefTrain="$experDir/$outPrefTrain"
fullOutPrefTest="$experDir/$outPrefTest"

queryLogFile=${trecRunDir}/query.log

resourceDirParams=" -fwd_index_dir \"$fwdIndexDir\" -embed_dir \"$embedDir\" -giza_root_dir \"$gizaRootDir\" -giza_iter_qty $GIZA_ITER_QTY "

# Don't quote $modelParams,$resourceDirParams, $maxQueryQtyTrainParam, $maxQueryQtyTestParam,
# and other *Param*

modelParams=""

if [ "$extrType" != "" -a "$extrType" != "$NO_FEAT_EXTRACTOR" ] ; then
  if [ "$trainCandQty" = "" ] ; then
    echo "Specify the numbers of candidate records retrieved for the training subset for each query (-train_cand_qty)!"
    exit 1
  fi

  if [ "$regenFeat" = "1" ] ; then
    # gen_features.sh provides a list of resource directories on its own
    scripts/query/gen_features.sh "$collect" \
                                  "$QREL_FILE" \
                                  "$TRAIN_SUBDIR" \
                                  lucene "$URI" "$trainCandQty" \
                                  "$extrType" "$experDir" \
                                  $maxQueryQtyTrainParam  \
                                  -out_pref "$outPrefTrain" \
                                  -thread_qty "$threadQty" \
                                  -query_cache_file "$cacheFileTrain" 2>&1
  fi

  modelFile="${fullOutPrefTrain}_${trainCandQty}.model"
  modelParams="$resourceDirParams -extr_type_final \"$extrType\" -model_final \"$modelFile\""

  if [ "$recompModel" = "1" ] ; then
    model_log_file="$experDir/model.log"
    echo > $model_log_file

    checkVarNonEmpty "metricType"

    # Here we rely on the naming convention of the feature-generation app, which generates
    # output for every value of the number of candidate records (for training).
    # We simply specify only one value in this case, namely, $trainCandQty
    if [ "$useLMART" = "1" ] ; then
      checkVarNonEmpty "numTrees"
      scripts/letor/ranklib_train_lmart.sh "${fullOutPrefTrain}_${trainCandQty}.feat" \
                                            "$modelFile" \
                                            "$numTrees" "$metricType" 2>&1 | tee -a "$model_log_file"
    else
      checkVarNonEmpty "numRandRestart"
      scripts/letor/ranklib_train_coordasc.sh "${fullOutPrefTrain}_${trainCandQty}.feat" "$modelFile" \
                                            "$numRandRestart" "$metricType" 2>&1 | tee -a "$model_log_file"
    fi



    if [ "$testModelResults" = "1" ] ; then
      # This part is for debug purposes only
      checkVarNonEmpty "trainCandQty"
      scripts/query/run_query.sh  -u "$URI" -cand_prov lucene -q "$inputDataDir/$TRAIN_SUBDIR/$queryFileName" \
                                  -n "$trainCandQty" \
                                  -run_id "$FAKE_RUN_ID" \
                                  -o "$trecRunDir/run_check_train_metrics" \
                                  -thread_qty "$threadQty"  \
                                  $modelParams \
                                  $maxQueryQtyTrainParam \
                                  -query_cache_file "$cacheFileTrain" 2>&1

      scripts/exper/eval_output.py "$inputDataDir/$TRAIN_SUBDIR/$QREL_FILE" \
                                    "${trecRunDir}/run_check_train_metrics_${trainCandQty}"

      echo "Model tested, now exiting!"
      exit 0
    fi
  fi
fi


if [ "$rerunLucene" = 1 ] ; then

  # Don't quote $resourceDirParams
  scripts/query/run_query.sh  -u "$URI" -cand_prov lucene -q "$inputDataDir/$testPart/$queryFileName"  \
                              -n "$testCandQtyList" \
                              -run_id "$FAKE_RUN_ID" \
                              -o "$trecRunDir/run" -thread_qty "$threadQty" \
                              $maxQueryQtyTestParam \
                              $modelParams \
                              -query_cache_file "$cacheFileTest" 2>&1|tee "$queryLogFile"
fi


qrels="$inputDataDir/$testPart/$QREL_FILE"

rm -f "${reportDir}/out_*"

for oneN in $testCandQtyListSpaceSep ; do
  echo "======================================"
  echo "N=$oneN"
  echo "======================================"
  reportPref="${reportDir}/out_${oneN}"

  scripts/exper/eval_output.py "$qrels"  "${trecRunDir}/run_${oneN}" "$reportPref" "$oneN"
  check "eval_output.py"
done

if [ "$deleteTrecRuns" = "1" ] ; then
  echo "Deleting trec runs from the directory: ${trecRunDir}"
  rm ${trecRunDir}/*
  # There should be at least one run, so, if rm fails, it fails because files can't be deleted
else
  echo "Bzipping trec runs in the directory: ${trecRunDir}"
  rm -f ${trecRunDir}/*.bz2
  bzip2 ${trecRunDir}/*
  # There should be at least one run, so, if rm fails, it fails because files can't be deleted
fi
echo "Bzipping trec_eval output in the directory: ${reportDir}"
bzip2 ${reportDir}/*.trec_eval
echo "Bzipping gdeval output in the directory: ${reportDir}"
bzip2 ${reportDir}/*.gdeval
