#!/bin/bash
. scripts/common_proc.sh
. scripts/config.sh

POS_ARGS=()

numCoresCPU=`getNumCpuCores`
check "getting the number of CPU cores, do you have /proc/cpu/info?"
threadQty=$numCoresCPU

nmslibURI=""

extrTypeIntermParam=""
modelIntermParam=""
candQtyParam=""
# Shouldn't delete these runs by default!
delete_trec_runs="0"
nmslibAddParams=""
nmslibExtrType=""
modelFile=""
maxNumQueryParam=""

compScores="1"
runId=$FAKE_RUN_ID

while [ $# -ne 0 ] ; do
  echo $1|grep "^-" >/dev/null 
  if [ $? = 0 ] ; then
    OPT_NAME="$1"
    if [ "$OPT_NAME" = "-delete_trec_runs" -o "$OPT_NAME" = "-knn_interleave" ] ; then
      OPT_VALUE=""
      OPT=""
      # option without an argument
      shift 1
    elif [ "$OPT_NAME" = "-skip_eval" ] ; then
      OPT_VALUE=""
      OPT=""
      # option without an argument
      shift 1
    else
      OPT_VALUE="$2"
      OPT="$1 $2"
      if [ "$OPT_VALUE" = "" ] ; then  
        echo "Option $OPT_NAME requires an argument." >&2
        exit 1
      fi
      shift 2
    fi
    case $OPT_NAME in
      -thread_qty)
        threadQty=$OPT_VALUE 
        ;;
      -model_file)
        modelFile=$OPT_VALUE 
        ;;
      -max_num_query)
        maxNumQueryParam=$OPT
        ;;
      -run_id)
        runId=$OPT_VALUE
        ;;
      -nmslib_addr)
        nmslibURI=$OPT_VALUE
        ;;
      -knn_interleave)
        nmslibAddParams="$nmslibAddParams $OPT_NAME"
        ;;
      -skip_eval)
        compScores="0"
        ;;
      -extr_type_nmslib)
        nmslibExtrType="$OPT_VALUE"
        ;;
      -model_interm)
        modelIntermParam=$OPT
        ;;
      -extr_type_interm)
        extrTypeIntermParam=$OPT
       ;; 
      -cand_qty)
        candQtyParam=$OPT
       ;; 
      -delete_trec_runs)
        delete_trec_runs="1"
       ;; 
      *)
        echo "Invalid option: $OPT_NAME" >&2
        exit 1
        ;;
    esac
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

testPart=${POS_ARGS[1]}
if [ "$testPart" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (2d arg)"
  exit 1 
fi

candProvType=${POS_ARGS[2]}
if [ "$candProvType" = "" ] ; then
  echo "Specify a candidate type: lucene, nmslib (3d arg)"
  exit 1 
fi

experSubDir=${POS_ARGS[3]}
if [ "$experSubDir" = "" ] ; then
  echo "Specify a sub-directory to store final results (3d positional arg)!"
  exit 1
fi

extrType="${POS_ARGS[4]}"

if [ "$extrType" = "" ] ; then
  echo "Specify a feature extractor path or none (4th positional arg)"
  exit 1
fi

if [ "$extrType" != "none" ] ; then
  if [ "$modelFile" = "" ] ; then
    echo "Specify a model file (-model_file)!"
    exit 1
  fi
fi

nTestStr="${POS_ARGS[5]}"

if [ "$nTestStr" = "" ] ; then
  echo "Specify a comma-separated list of candidate records # retrieved for testing for each query (5th positional arg)!"
  exit 1
fi

nTestList=`echo $nTestStr|sed 's/,/ /g'`

checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "GIZA_SUBDIR"
checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "FINAL_EXPER_SUBDIR"
checkVarNonEmpty "COLLECT_ROOT"

checkVarNonEmpty "testPart"
checkVarNonEmpty "experSubDir"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
embedDir="$COLLECT_ROOT/$collect/$EMBED_SUBDIR/"
gizaRootDir="$COLLECT_ROOT/$collect/$GIZA_SUBDIR"

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[2]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

experDirBase="$COLLECT_ROOT/$collect/$FINAL_EXPER_SUBDIR"

experDirUnique=$(getExperDirUnique "$experDirBase" "$testPart" "$experSubDir")

if [ ! -d "$experDirUnique" ] ; then
  mkdir -p "$experDirUnique"
  if [ "$?" != "0" ] ; then
    echo "Cannot create '$experDirUnique'"
    exit 1
  fi
fi


if [ "$candProvType" = "lucene" ] ; then
  URI="$COLLECT_ROOT/$collect/$LUCENE_INDEX_SUBDIR"
elif [ "$candProvType" = "nmslib" ] ; then
  if [ "$nmslibURI" = "" ] ; then
    echo "Missing parameter -nmslib_addr"
    exit 1
  fi
  URI=$nmslibURI
  if [ "$nmslibExtrType" = "" ] ; then
    echo "Missing parameter -extr_type_nmslib"
    exit 1
  fi
  nmslibAddParams=" -extr_type_nmslib \"$nmslibExtrType\" $nmslibAddParams"
else
  echo "Invalid provider type: $candProvType"
fi

experDir="$COLLECT_ROOT/$collect/$FINAL_EXPER_SUBDIR"
if [ ! -d "$experDir" ] ; then
  mkdir -p $experDir
  if [ "$?" != "0" ] ; then
    echo "Cannot create '$experDir'"
    exit 1
  fi
else
  # Clean-up"
  rm -rf "$experDir/*"
fi

checkVarNonEmpty "experDir"

experDirUnique=$(getExperDirUnique "$experDir" "$testPart" "$experSubDir")

checkVarNonEmpty "experDirUnique"

trecRunDir="$experDirUnique/trec_runs"
reportDir="$experDirUnique/rep"

mkdir -p $trecRunDir
mkdir -p "$reportDir"


echo "Deleting old reports from the directory: ${reportDir}"
rm -f ${reportDir}/*
check "rm -f ${reportDir}/*"

# No extractor type here, the user will have to specify the sub-directory that is extractor-type-specific
OUT_PREF_TEST="out_${collect}_${testPart}"
FULL_OUT_PREF_TEST="$experDirUnique/$OUT_PREF_TEST"

query_log_file=${reportDir}/query.log
check "query_log_file=${reportDir}/query.log"

echo "============================================"
echo "Using $testPart for testing!"
echo "Candidate provider type:        $candProvType"
echo "Candidate provider URI:         $URI"
echo "Run id:                         $runId"
echo "Data directory:                 $inputDataDir"
echo "Data file name:                 $queryFileName"

echo "Experiment directory:           $experDirUnique"
echo "Report directory:               $reportDir"

if [ "$maxNumQueryParam" != "" ] ; then
  echo "Max number of queries param.:   $maxNumQueryParam"
fi
echo "Directory with TREC-style runs: $trecRunDir"
echo "Model file:                     $modelFile"
if [ "$extrTypeIntermParam" != "" ] ; then
  echo "Intermediate extractor param.:  $extrTypeIntermParam"
fi
if [ "$modelIntermParam" != "" ] ; then
  echo "Intermediate model param.:      $modelIntermParam"
fi
if [ "$nmslibAddParams != """ ] ; then
  echo "NMSLIB add params:              $nmslibAddParams"
fi
echo "============================================"

statFile="$reportDir/stat_file"

if [ "$extrType" = "none" ] ; then
  extrFinalParam=""
else
  extrFinalParam=" -model_final $modelFile -extr_type_final $extrType"
fi

resourceDirParams=" -fwd_index_dir \"$fwdIndexDir\" -embed_dir \"$embedDir\" -giza_root_dir \"$gizaRootDir\" -giza_iter_qty $GIZA_ITER_QTY "

trecRunPrefix="$trecRunDir/run"

# Despite all the efforts these quotes won't preserve spaces :-( due to quirks of passing arguments
cmd="scripts/query/run_query.sh  -u \"$URI\"  -cand_prov $candProvType -thread_qty $threadQty \
  -run_id $runId \
  $nmslibAddParams \
  $resourceDirParams \
  $extrFinalParam \
  -q \"$inputDataDir/$testPart/$queryFileName\"  -n $nTestStr \
  -o \"$trecRunPrefix\"  -save_stat_file \"$statFile\" \
  $candQtyParam $maxNumQueryParam $extrTypeIntermParam $modelIntermParam"

execAndCheck "$cmd 2>&1|tee $query_log_file"

if [ "$compScores" = "1" ] ; then 

  rm -f "${reportDir}/out_*"

  qrels="$inputDataDir/$testPart/$QREL_FILE"

  for oneN in $nTestList ; do
    echo "======================================"
    echo "N=$oneN"
    echo "======================================"
    reportPref="${reportDir}/out_${oneN}"
  
    scripts/exper/eval_output.py "$qrels" "${trecRunPrefix}_${oneN}" "$reportPref" "$oneN"
    check "eval_output.py"
  done
fi
  
if [ "$delete_trec_runs" = "1" ] ; then
  echo "Deleting trec runs from the directory: ${trecRunDir}"
  rm ${trecRunDir}/*
  # There should be at least one run, so, if rm fails, it fails because files can't be deleted
  check "rm ${trecRunDir}/*" 
else
  echo "Bzipping trec runs in the directory: ${trecRunDir}"
  rm -f ${trecRunDir}/*.bz2
  bzip2 ${trecRunDir}/*
  # There should be at least one run, so, if rm fails, it fails because files can't be deleted
  check "bzip2 ${trecRunDir}/*" 
fi

if [ "$compScores" = "1" ] ; then 
  echo "Bzipping trec_eval output in the directory: ${reportDir}"
  rm -f ${reportDir}/*.trec_eval.bz2
  bzip2 ${reportDir}/*.trec_eval
  check "bzip2 ${reportDir}/*.trec_eval"
fi
