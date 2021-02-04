#!/bin/bash
source scripts/common_proc.sh
source scripts/config.sh
# This script works in two modes:
# 1. Train and test the model (if the final model is not specified)
# 2. Test an already specified trained model (-model_final)
# If the extractor is not specified, training is not possible,
# so we will simply run a test without any model and feature extractor


regenFeat="1"
recompModel="1" # for debug only
testModelResults="0"

posArgs=()

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
checkVarNonEmpty "DEFAULT_TRAIN_SUBDIR"
checkVarNonEmpty "DEFAULT_INTERM_CAND_QTY"
checkVarNonEmpty "DEFAULT_TRAIN_CAND_QTY"
checkVarNonEmpty "DEFAULT_TEST_CAND_QTY_LIST"
checkVarNonEmpty "SEP_DEBUG_LINE"

checkVarNonEmpty "FAKE_RUN_ID"

runId=$FAKE_RUN_ID

numRandRestart=$DEFAULT_NUM_RAND_RESTART
numTrees=$DEFAULT_NUM_TREES
metricType=$DEFAULT_METRIC_TYPE

extrType=""

skipEval=0

checkVarNonEmpty "CAND_PROV_LUCENE"
checkVarNonEmpty "CAND_PROV_NMSLIB"

candProvType="$CAND_PROV_LUCENE"

candProvURI=""
candQty="$DEFAULT_INTERM_CAND_QTY"
candProvAddConf=""
candProvAddConfParam=""

maxFinalRerankQtyParam=""

extrTypeIntermParam=""
modelIntermParam=""

modelFinalParams=""
modelFinal=""

trainPart="$DEFAULT_TRAIN_SUBDIR"
trainCandQty="$DEFAULT_TRAIN_CAND_QTY"
intermCandQty="$DEFAULT_INTERM_CAND_QTY"
testCandQtyList="$DEFAULT_TEST_CAND_QTY_LIST"


skipEval=0
testOnly=0
trainOnly=0

while [ $# -ne 0 ] ; do
  optValue=""
  opt=""
  if [[ "$1" = -* ]] ; then
    optName="$1"
    if [ "$optName" = "-use_lmart" ] ; then
      useLMART="1"
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-debug_print" ] ; then
      set -x
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-reuse_feat" ] ; then
      regenFeat="0"
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-skip_eval" ] ; then
      skipEval=1
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-test_model_results" ] ; then
      testModelResults="1"
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-delete_trec_runs" ] ; then
      deleteTrecRuns="1"
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-test_only" ] ; then
      testOnly=1
      # option without an argument shift by 1
      shift 1
    elif [ "$optName" = "-train_only" ] ; then
      trainOnly=1
      # option without an argument shift by 1
      shift 1
    else
      optValue="$2"
      opt="$1 $2"
      if [ "$optValue" = "" ] ; then
        echo "Option $optName requires an argument." >&2
        exit 1
      fi
      shift 2 # option with an argument shift by two
      case $optName in
        -thread_qty)
          threadQty=$optValue
          ;;
        -cand_prov_add_conf)
          candProvAddConf=$optValue
          candProvAddConfParam=$opt
          ;;
        -cand_prov_uri)
          candProvURI=$optValue
          ;;
        -num_rand_restart)
          numRandRestart=$optValue
          ;;
        -run_id)
          runId=$optValue
          ;;
        -train_part)
          trainPart=$optValue
          ;;
        -extr_type)
          extrType=$optValue
          ;;
        -model_interm)
          modelIntermParam=$opt
          ;;
        -model_final)
          modelFinal=$optValue
          ;;
        -train_cand_qty)
          trainCandQty=$optValue
          ;;
        -cand_prov)
          candProvType=$optValue
          ;;
        -cand_qty)
          candQty=$optValue
          ;;
        -test_cand_qty_list)
          testCandQtyList=$optValue
          ;;
        -extr_type_interm)
          extrTypeIntermParam=$opt
        ;;
        -num_trees)
          numTrees=$optValue
          ;;
        -metric_type)
          metricType=$optValue
          ;;
        -max_num_query_train)
          maxQueryQtyTrain=$optValue
          maxQueryQtyTrainParam=" -max_num_query $maxQueryQtyTrain"
          ;;
        -max_num_query_test)
          maxQueryQtyTest=$optValue
          maxQueryQtyTestParam=" -max_num_query $maxQueryQtyTest"
          ;;
        -max_final_rerank_qty)
          maxFinalRerankQtyParam=$opt
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
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

experDirBase=${posArgs[1]}
if [ "$experDirBase" = "" ] ; then
  echo "Specify a working directory (2d arg)!"
  exit 1
fi

if [ "$trainOnly" != "1" ] ; then
  testPart=${posArgs[2]}
  if [ "$testPart" = "" ] ; then
    echo "Specify a test part, e.g., dev1, or use -train_only (3d arg)"
    exit 1
  fi
fi

testCandQtyListSpaceSep=`echo $testCandQtyList|sed 's/,/ /g'`

if [ "$modelFinal" != "" -a "$extrType" = "" ] ; then
  echo "You specified the final model, but not the extractor type!"
  exit 1
fi

if [ "$testOnly" = "0" ] ; then
  # Should be set to a default value in the beginning
  checkVarNonEmpty "trainPart"
  echo "Running training in addition to test b/c we have an extractor, but not the final model!"
   if [ "$extrType" = "" ] ; then
    echo "In the training mode, you need to specify the feature extractor -extr_type!"
    exit 1
  fi
else
  echo "Running in test only model"
  if [ "$modelFinal" == "" -a "$extrType" != "" ] ; then
    echo "You specified the extractor type, but not the final model!"
    exit 1
  fi
fi

# Do it only after argument parsing
set -eo pipefail

checkVarNonEmpty "LETOR_SUBDIR"
checkVarNonEmpty "TRECRUNS_SUBDIR"
checkVarNonEmpty "REP_SUBDIR"

letorDir="$experDirBase/$LETOR_SUBDIR"
trecRunDir="$experDirBase/$TRECRUNS_SUBDIR"
reportDir="$experDirBase/$REP_SUBDIR"

checkVarNonEmpty "experDirBase"
if [ -d "$experDirBase" ] ; then
  # Be very careful with this sort of deletions,
  # double-check it's not empty again
  if [ "$experDirBase" != "" ] ; then
    rm -rf "$experDirBase/*"
  else
    echo "Bug: empty experDirBase here!"
    exit 1
  fi
fi

mkdir -p "$letorDir"
mkdir -p "$trecRunDir"
mkdir -p "$reportDir"


checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"



checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "GIZA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
embedDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR/"
gizaRootDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$GIZA_SUBDIR"

commonResourceParams="\
-fwd_index_dir \"$fwdIndexDir\" \
-embed_dir \"$embedDir\" \
-giza_root_dir \"$gizaRootDir\" "

checkVarNonEmpty "inputDataDir" # set by set_common_resource_vars.sh
checkVarNonEmpty "commonResourceParams"  # set by set_common_resource_vars.sh

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[3]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

# Caching is only marginally useful.
# However, when enabled it can accidentally screw up things quite a bit 
queryCacheParamTrain=""
queryCacheParamTest=""

if [ "$candProvType" = "$CAND_PROV_LUCENE" -a "$candProvURI" = "" ] ; then
  candProvURI="$LUCENE_INDEX_SUBDIR"
else
  if [ "$candProvURI" = "" ] ; then
    echo "You must specify -cand_prov_uri for the provider $candProvType"
    exit 1
  fi

fi

# All provider URIs except for NMSLIB are relative to the collection location
if [ "$candProv" != "$CAND_PROV_NMSLIB" ] ; then
  candProvURI="$COLLECT_ROOT/$collect/$candProvURI"
fi


# Don't quote $modelFinalParams,
#             $candProvAddConfParam,
#             $commonParams,
#             $maxQueryQtyTrainParam,
#             $maxQueryQtyTestParam,
# as we llas other *Param*

commonAddParams="-cand_qty $candQty $candProvAddConfParam \
 -thread_qty "$threadQty" \
$extrTypeIntermParam $modelIntermParam \
$commonResourceParams" 

outPrefTrain="out_${collect}_${trainPart}"
outPrefTest="out_${collect}_${testPart}"
fullOutPrefTrain="$letorDir/$outPrefTrain"
fullOutPrefTest="$letorDir/$outPrefTest"

queryLogFile=${trecRunDir}/query.log


if [ "$testOnly" = "0" ] ; then
  if [ "$modelFinal" != "" ] ; then
    echo "Bug: here the modelFinal variable should be empty!"
  fi
  modelFinal="${fullOutPrefTrain}_${trainCandQty}.model"
fi
if [ "$modelFinal" != "" ] ; then
  checkVarNonEmpty "extrType"
  modelFinalParams="-extr_type_final \"$extrType\" -model_final \"$modelFinal\""
fi

echo "$SEP_DEBUG_LINE"
echo "Parameter and settings review"
echo "$SEP_DEBUG_LINE"

echo "Data directory:          $inputDataDir"
if [ "$testOnly" = "0" ] ; then
  echo "Training part:           $trainPart"
fi
echo "Test part:               $testPart"
echo "Data file name:          $queryFileName"
echo "Forward index directory: $fwdIndexDir"
echo "Embedding directory:     $embedDir"
echo "GIZA root directory:     $gizaRootDir"

echo "$SEP_DEBUG_LINE"

echo "Experiment directory:                    $letorDir"
echo "RUN id:                                  $runId"
echo "QREL file:                               $QREL_FILE"
echo "Directory with TREC-style runs:          $trecRunDir"
echo "Report directory:                        $reportDir"
echo "Query cache file params (for training):  $queryCacheParamTrain"
echo "Query cache file params (for testing):   $queryCacheParamTest"

echo "$SEP_DEBUG_LINE"

echo "Candidate provider type:                  $candProvType"
echo "Candidate provider URI:                   $candProvURI"
echo "Candidate provider # of candidates param  $candQtyParam"
echo "Candidate provider addition configuration $candProvAddConf"

echo "$SEP_DEBUG_LINE"

echo "Maximum number of test queries parameter: $maxQueryQtyTrainParam"
echo "Number of candidate records for training: $trainCandQty"
echo "Maximum number of train candidates:       $maxQueryQtyTestParam"
echo "A list for the number of test candidates: $testCandQtyList"

echo "$SEP_DEBUG_LINE"

echo "Common parameters shared at all steps:    $commonParams"
echo "Intermediate extractor parameters:        $extrTypeIntermParam"
echo "Intermediate model parameters:            $modelIntermParam"
echo "Final model parameters:                   $modelFinalParams"

echo "Use LAMBDA MART:            $useLMART"
echo "Learning metric:            $metricType"
if [ "$useLMART" = "1" ] ; then
  echo "Number of trees:            $numTrees"
else
  echo "Number of random restarts:  $numRandRestart"
fi
echo "$SEP_DEBUG_LINE"

if [ "$testOnly" = "0" ] ; then
  if [ "$trainCandQty" = "" ] ; then
    echo "Specify the numbers of candidate records retrieved for the training subset for each query (-train_cand_qty)!"
    exit 1
  fi

  if [ "$regenFeat" = "1" ] ; then
    checkVarNonEmpty "extrType"

    # This APP can require a lot of memory
    setJavaMem 5 9
    target/appassembler/bin/GenFeaturesAppMultThread -u "$candProvURI" -cand_prov "$candProvType" \
                                    -run_id "$runId" \
                                    -q "$inputDataDir/$trainPart/$queryFileName" \
                                    -qrel_file "$inputDataDir/$trainPart/$QREL_FILE" \
                                    -n "$trainCandQty" \
                                    -f "$fullOutPrefTrain" \
                                    -extr_type_final \"$extrType\" \
                                     $commonAddParams \
                                     $maxQueryQtyTrainParam  \
                                     $queryCacheParamTrain 2>&1 | tee "${fullOutPrefTrain}_${trainCandQty}.log"

  fi

  if [ "$recompModel" = "1" ] ; then
    modelLogFile="$letorDir/model.log"
    echo > $modelLogFile

    checkVarNonEmpty "metricType"
    checkVarNonEmpty "modelFinal"

    # Here we rely on the naming convention of the feature-generation app, which generates
    # output for every value of the number of candidate records (for training).
    # We simply specify only one value in this case, namely, $trainCandQty
    if [ "$useLMART" = "1" ] ; then
      checkVarNonEmpty "numTrees"
      scripts/letor/ranklib_train_lmart.sh "${fullOutPrefTrain}_${trainCandQty}.feat" \
                                            "$modelFinal" \
                                            "$numTrees" "$metricType" 2>&1 | tee -a "$modelLogFile"
    else
      checkVarNonEmpty "numRandRestart"
      scripts/letor/ranklib_train_coordasc.sh "${fullOutPrefTrain}_${trainCandQty}.feat" "$modelFinal" \
                                            "$numRandRestart" "$metricType" 2>&1 | tee -a "$modelLogFile"
    fi


    if [ "$testModelResults" = "1" ] ; then
      # This part is for debug purposes only
      checkVarNonEmpty "trainCandQty"
      target/appassembler/bin/QueryAppMultThread  -u "$candProvURI" -cand_prov "$candProvType" \
                                  -q "$inputDataDir/$trainPart/$queryFileName" \
                                  -n "$trainCandQty" \
                                  -run_id "$runId" \
                                  -o "$trecRunDir/run_check_train_metrics" \
                                  $commonAddParams \
                                  $maxFinalRerankQtyParam \
                                  $modelFinalParams \
                                  $maxQueryQtyTrainParam \
                                  $cacheFileParamTrain 2>&1

      scripts/exper/eval_output.py "$inputDataDir/$trainPart/$QREL_FILE" \
                                    "${trecRunDir}/run_check_train_metrics_${trainCandQty}"

      echo "Model tested, now exiting!"
      exit 0
    fi
  fi
fi

if [ "$trainOnly" = "1" ] ; then
  exit 0
fi

statFile="$reportDir/$STAT_FILE"
$resourceDirParams
target/appassembler/bin/QueryAppMultThread \
                            -u "$candProvURI" -cand_prov "$candProvType" \
                            -q "$inputDataDir/$testPart/$queryFileName"  \
                            -n "$testCandQtyList" \
                            -run_id "$runId" \
                            -o "$trecRunDir/run"  -save_stat_file "$statFile" \
                            $commonAddParams \
                            $maxFinalRerankQtyParam \
                            $maxQueryQtyTestParam \
                            $modelFinalParams \
                            $queryCacheParamTest 2>&1|tee "$queryLogFile"


qrels="$inputDataDir/$testPart/$QREL_FILE"

rm -f "${reportDir}/out_*"

if [ "$skipEval" != "1" ] ; then
  for oneN in $testCandQtyListSpaceSep ; do
    echo "$SEP_DEBUG_LINE"
    echo "N=$oneN"
    echo "$SEP_DEBUG_LINE"
    reportPref="${reportDir}/out_${oneN}"

    scripts/exper/eval_output.py "$qrels"  "${trecRunDir}/run_${oneN}" "$reportPref" "$oneN"
  done

  echo "Bzipping trec_eval output in the directory: ${reportDir}"
  bzip2 ${reportDir}/*.trec_eval
  echo "Bzipping gdeval output in the directory: ${reportDir}"
  bzip2 ${reportDir}/*.gdeval
fi

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

