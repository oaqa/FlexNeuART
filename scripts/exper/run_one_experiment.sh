#!/bin/bash
source ./common_proc.sh
source ./config.sh

# This script works in two modes:
# 1. Train and test the model (if the final model is not specified)
# 2. Test an already specified trained model (-model_final)
# If the extractor is not specified, training is not possible.

regenFeat="1"
recompModel="1" # for debug only

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
checkVarNonEmpty "DEFAULT_CAND_PROV_QTY"
checkVarNonEmpty "DEFAULT_TRAIN_CAND_QTY"
checkVarNonEmpty "DEFAULT_TEST_CAND_QTY_LIST"
checkVarNonEmpty "SEP_DEBUG_LINE"
checkVarNonEmpty "QUESTION_FILE_PREFIX"

checkVarNonEmpty "FAKE_RUN_ID"

runId=$FAKE_RUN_ID

numRandRestart=$DEFAULT_NUM_RAND_RESTART
numTrees=$DEFAULT_NUM_TREES
metricType=$DEFAULT_METRIC_TYPE

extrTypeFinal=""

skipEval=0

checkVarNonEmpty "CAND_PROV_LUCENE"
checkVarNonEmpty "CAND_PROV_NMSLIB"

candProvType="$CAND_PROV_LUCENE"

candProvURI=""
candProvQty="$DEFAULT_CAND_PROV_QTY"
candProvAddConf=""
candProvAddConfParam=""

maxFinalRerankQtyParam=""

extrTypeIntermParam=""
modelIntermParam=""

modelFinalParams=""
modelFinal=""

trainPart="$DEFAULT_TRAIN_SUBDIR"
trainCandQty="$DEFAULT_TRAIN_CAND_QTY"
testCandQtyList="$DEFAULT_TEST_CAND_QTY_LIST"

checkVarNonEmpty "MODEL1_SUBDIR"
model1SubDir="$MODEL1_SUBDIR"

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
        -model1_subdir)
          model1SubDir=$optValue
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
        -extr_type_final)
          extrTypeFinal=$optValue
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
        -cand_prov_qty)
          candProvQty=$optValue
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

experDirBaseRelative=${posArgs[1]}
if [ "$experDirBaseRelative" = "" ] ; then
  echo "Specify a working directory relative to the collection directory (2d arg)!"
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

if [ "$modelFinal" != "" -a "$extrTypeFinal" = "" ] ; then
  echo "You specified the final model, but not the extractor type!"
  exit 1
fi

if [ "$testOnly" = "0" ] ; then
  # Should be set to a default value in the beginning
  checkVarNonEmpty "trainPart"
  echo "Running training in addition to testing"
   if [ "$extrTypeFinal" = "" ] ; then
    echo "In the training mode, you need to specify the feature extractor -extr_type!"
    exit 1
  fi
else
  echo "Running in test only model"
  if [ "$modelFinal" == "" -a "$extrTypeFinal" != "" ] ; then
    echo "You specified the extractor type, but not the final model!"
    exit 1
  fi
fi

# Do it only after argument parsing
set -eo pipefail

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

collectDir="$COLLECT_ROOT/$collect"
inputDataDir="$collectDir/$INPUT_DATA_SUBDIR"
fwdIndexDir="$FWD_INDEX_SUBDIR/"
embedDir="$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR/"
model1Dir="$DERIVED_DATA_SUBDIR/$model1SubDir"

checkVarNonEmpty "LETOR_SUBDIR"
checkVarNonEmpty "TRECRUNS_SUBDIR"
checkVarNonEmpty "REP_SUBDIR"

checkVarNonEmpty "experDirBaseRelative"
experDirBase="$collectDir/$experDirBaseRelative"

letorDirRelative="$experDirBaseRelative/$LETOR_SUBDIR"
letorDir="$experDirBase/$LETOR_SUBDIR"
trecRunDir="$experDirBase/$TRECRUNS_SUBDIR"
reportDir="$experDirBase/$REP_SUBDIR"

checkVarNonEmpty "experDirBase"

mkdir -p "$letorDir"
mkdir -p "$trecRunDir"
mkdir -p "$reportDir"


checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"




commonResourceParams="\
-collect_dir $collectDir \
-fwd_index_dir $fwdIndexDir \
-embed_dir $embedDir \
-model1_dir $model1Dir "

checkVarNonEmpty "inputDataDir" # set by set_common_resource_vars.sh
checkVarNonEmpty "commonResourceParams"  # set by set_common_resource_vars.sh

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


# Don't quote $modelFinalParams,
#             $candProvAddConfParam,
#             commonAddParams,
#             $maxQueryQtyTrainParam,
#             $maxQueryQtyTestParam,
# as we llas other *Param*

commonAddParams="\
  -cand_prov_qty $candProvQty $candProvAddConfParam \
  -thread_qty "$threadQty" \
  $extrTypeIntermParam
  $modelIntermParam \
  $commonResourceParams"

outPrefTrain="out_${collect}_${trainPart}"
outPrefTest="out_${collect}_${testPart}"
fullOutPrefTrainRelative="$letorDirRelative/$outPrefTrain"
fullOutPrefTrain="$letorDir/$outPrefTrain"
fullOutPrefTest="$letorDir/$outPrefTest"

queryLogFile=${trecRunDir}/query.log

# When training we use an absolute location to save a model.
# However, when the test app loads the model it expects all the resources and configs
# to be relative to the collection directory.
if [ "$testOnly" = "0" ] ; then
  if [ "$modelFinal" != "" ] ; then
    echo "Aborting training, because the final model is specified! Remove the final model specification or run in test-only mode!"
    exit 1
  fi
  modelFinal="${fullOutPrefTrain}_${trainCandQty}.model"
  modelFinalRelative="${fullOutPrefTrainRelative}_${trainCandQty}.model"
else
  # This model file name comes from the extractor JSON and it is supposed to be collection-relative
  modelFinalRelative="$modelFinal"
fi
if [ "$modelFinal" != "" ] ; then
  checkVarNonEmpty "extrTypeFinal"
  modelFinalParams="-extr_type_final \"$extrTypeFinal\" -model_final \"$modelFinalRelative\""
fi

echo "$SEP_DEBUG_LINE"
echo "Parameter and settings review"
echo "$SEP_DEBUG_LINE"

echo "Data directory:          $inputDataDir"
if [ "$testOnly" = "0" ] ; then
  echo "Training part:           $trainPart"
fi
echo "Test part:               $testPart"
echo "Forward index directory: $fwdIndexDir"
echo "Embedding directory:     $embedDir"
echo "Model1 directory:        $model1Dir"

echo "$SEP_DEBUG_LINE"

echo "Experiment directory:                    $letorDir"
echo "RUN id:                                  $runId"
echo "QREL file:                               $QREL_FILE"
echo "Directory with TREC-style runs:          $trecRunDir"
echo "Report directory:                        $reportDir"

echo "$SEP_DEBUG_LINE"

echo "Candidate provider type:                  $candProvType"
echo "Candidate provider URI:                   $candProvURI"
echo "Candidate provider # of candidates        $candQty"
echo "Candidate provider add. configuration     $candProvAddConf"

echo "$SEP_DEBUG_LINE"

echo "Maximum number of train queries parameter: $maxQueryQtyTrainParam"
echo "Number of candidate records for training:  $trainCandQty"
echo "A list for the number of test candidates:  $testCandQtyList"
echo "Maximum number of test queries parameter:  $maxQueryQtyTestParam"

echo "$SEP_DEBUG_LINE"

echo "Common parameters shared at all steps:    $commonAddParams"
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
#
# The querying apps have complicate settings for the number of entries returned and reranked
#
# 1. The maximum num. of entries returned by the provider is defined as the maximum of parameter -cand_prov_qty
#    and the maximum number of entries requested via the parameter -n
# 2. The number of requested entries is different between training and test runs. For training, -n is
#    forced to be equal to the value passed via parameter -train_cand_qty
# 3. If an intermediate re-ranker is specified, it re-ranks *ALL* entries returned by the provider.
# 4. However, the number of entries re-ranked by the final re-ranker can be limited using the parameter -max_final_rerank_qty
#    In this case, the scores of the entries with ranks higher than the value -max_final_rerank_qty, are updated
#    in such a way that
#     a. they are ranked lower than the top entries
#     b. and their original relative order is preserved
#

if [ "$testOnly" = "0" ] ; then
  if [ "$trainCandQty" = "" ] ; then
    echo "Specify the numbers of candidate records retrieved for the training subset for each query (-train_cand_qty)!"
    exit 1
  fi

  if [ "$regenFeat" = "1" ] ; then
    checkVarNonEmpty "extrTypeFinal"

    # For the final training, we re-rank only top-K candidates.
    if [ "$maxFinalRerankQtyParam" != "" ] ; then
      echo "WARNING: during training we set -max_final_rerank_qty to $trainCandQty"
    fi

    NO_MAX=1
    setJavaMem 1 16 $NO_MAX
    GenFeaturesAppMultThread -u "$candProvURI" -cand_prov "$candProvType" \
                                    -run_id "$runId" \
                                    -query_file_pref "$inputDataDir/$trainPart/$QUESTION_FILE_PREFIX" \
                                    -qrel_file "$inputDataDir/$trainPart/$QREL_FILE" \
                                    -n "$trainCandQty" \
                                    -max_final_rerank_qty "$trainCandQty" \
                                    -f "$fullOutPrefTrain" \
                                    -extr_type_final \"$extrTypeFinal\" \
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
      ./letor/ranklib_train_lmart.sh "${fullOutPrefTrain}_${trainCandQty}.feat" \
                                            "$modelFinal" \
                                            "$numTrees" "$metricType" 2>&1 | tee -a "$modelLogFile"
    else
      checkVarNonEmpty "numRandRestart"
      ./letor/ranklib_train_coordasc.sh "${fullOutPrefTrain}_${trainCandQty}.feat" "$modelFinal" \
                                            "$numRandRestart" "$metricType" 2>&1 | tee -a "$modelLogFile"
    fi

  fi
fi

if [ "$trainOnly" = "1" ] ; then
  exit 0
fi

statFile="$reportDir/$STAT_FILE"
$resourceDirParams
NO_MAX=1
setJavaMem 1 16 $NO_MAX
QueryAppMultThread \
    -u "$candProvURI" -cand_prov "$candProvType" \
    -query_file_pref "$inputDataDir/$testPart/$QUESTION_FILE_PREFIX" \
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

    ./exper/eval_output.py "$qrels"  "${trecRunDir}/run_${oneN}" "$reportPref" "$oneN"
  done

  echo "Bzipping trec_eval output in the directory: ${reportDir}"
  bzip2 ${reportDir}/*.trec_eval
# Don't normally do it any more
#  echo "Bzipping gdeval output in the directory: ${reportDir}"
#  bzip2 ${reportDir}/*.gdeval
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

