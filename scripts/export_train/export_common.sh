#@IgnoreInspection BashAddShebang


checkVarNonEmpty "DEV1_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "CAND_PROV_LUCENE"
checkVarNonEmpty "CAND_PROV_NMSLIB"

# This default is set by a calling script
checkVarNonEmpty "outSubdir"

threadQty=1
hardNegQty=0
sampleMedNegQty=5
sampleEasyNegQty=0
candTrainQty=100
candTrain4PosQty=1000
candTestQty=20
randSeed=0
maxDocWhitespaceQty=-1 # means no truncation
queryExportFieldName=""
candProv=$CAND_PROV_LUCENE
providerURI=""
candProvAddConf=""
handleCaseParam=""

boolOpts=(
"h" "help" "print help"
"keep_case" "keepCase" "do not lowercase"
)

paramOpts=(
"thread_qty"             "threadQty"           "# of threads"
"cand_prov"              "candProv"            "Candidate record provider type"
"cand_prov_uri"          "providerURI"         "Provider URI: an index location, a query server address, etc"
"cand_prov_add_conf"     "candProvAddConf"     "JSON with additional candidate provider parameters"
"out_subdir"             "outSubdir"           "output sub-directory (default $outSubdir)"
"query_export_field"     "queryExportFieldName" "an optional name of the query field name whose content we export (if different from the index field name)"
"hard_neg_qty"           "hardNegQty"          "A max. # of *HARD* negative examples (all K top-score candidates) per query (default $hardNegQty)"
"sample_med_neg_qty"     "sampleMedNegQty"     "A max. # of *MEDIUM* negative samples (negative candidate and QREL samples) per query (default $sampleMedNegQty)"
"sample_easy_neg_qty"    "sampleEasyNegQty"    "A max. # of *EASY* negative samples (sampling arbitrary docs) per query (default $sampleEasyNegQty)"
"cand_train_qty"         "candTrainQty"        "A max. # of candidate records from which we sample medium negatives (default $candTrainQty)"
"cand_train_4pos_qty"    "candTrain4PosQty"    "A max. # of candidate records from which we select positive samples. (default $candTrain4PosQty)"
"cand_test_qty"          "candTestQty"         "A max. # of candidate records to generate test data (default $candTestQty)"
"max_num_query_train"    "maxNumQueryTrain"    "Optional max. # of train queries"
"max_num_query_test"     "maxNumQueryTest"     "Optional max. # of test/dev queries"
"max_doc_whitespace_qty" "maxDocWhitespaceQty" "Optional max. # of whitespace separated tokens to keep in a document"
"seed"                   "randSeed"            "A random seed (default $randSeed)"
)

usageMain="<collection> <name of the index field> \
<train subdir, e.g., $DEFAULT_TRAIN_SUBDIR> \
<test subdir, e.g., $DEV1_SUBDIR>"

parseArguments $@

if [ "$maxNumQueryTrain" != "" ] ; then
  maxNumQueryTrainParam=" -max_num_query_train $maxNumQueryTrain "
fi

if [ "$maxNumQueryTest" != "" ] ; then
  maxNumQueryTestParam=" -max_num_query_test $maxNumQueryTest "
fi

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

indexExportFieldName=${posArgs[1]}
if [ "$indexExportFieldName" = "" ] ; then
  genUsage "$usageMain" "Specify the name of the exported index field (2d arg)"
  exit 1
fi
if [ "$queryExportFieldName" = "" ] ; then
  queryFieldName="$queryExportFieldName"
fi

partTrain=${posArgs[2]}
if [ "$partTrain" = "" ] ; then
  genUsage "$usageMain" "Specify the training sub-dir, e.g., $DEFAULT_TRAIN_SUBDIR (3d arg)"
  exit 1
fi

partTest=${posArgs[3]}
if [ "$partTest" = "" ] ; then
  genUsage "$usageMain" "Specify the training sub-dir, e.g., $DEV1_SUBDIR (4th arg)"
  exit 1
fi


checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "MODEL1_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "QREL_FILE"

collectDir="$COLLECT_ROOT/$collect"
inputDataDir="$collectDir/$INPUT_DATA_SUBDIR"
fwdIndexDir="$FWD_INDEX_SUBDIR/"
embedDir="$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR/"
model1Dir="$DERIVED_DATA_SUBDIR/$MODEL1_SUBDIR"

commonResourceParams="\
-collect_dir $collectDir  \
-fwd_index_dir $fwdIndexDir \
-embed_dir $embedDir \
-model1_dir $model1Dir "

if [ "$providerURI" = "" ] ; then
  if [ "$candProv" = "$CAND_PROV_LUCENE" ] ; then
    providerURI="$LUCENE_INDEX_SUBDIR"
  else
    echo "Need to specify the candidate provider URI (-u) if the provider is not $CAND_PROV_LUCENE"
    exit 1
  fi
fi

candProvParams=" -cand_prov \"$candProv\" -u \"$providerURI\" "
if [ "$candProvAddConf" != "" ] ; then
  # Additional config is collection-location relative
  candProvParams="$candProvParams -cand_prov_add_conf \"$candProvAddConf\""
fi

outDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$outSubdir/$indexFieldName"

if [ ! -d "$outDir" ] ; then
  mkdir -p "$outDir"
fi


if [ "$keepCase" = "1" ] ; then
  handleCaseParam=" -keep_case "
fi

if [ "$queryExportFieldName" = "" ] ; then
  queryExportFieldName="$indexExportFieldName"
fi

echo "========================================================"
echo "Collection directory:      $collectDir"
echo "Train split: $partTrain"
echo "Eval split: $partTest"
echo "Random seed: $randSeed"
echo "Output directory: $outDir"
echo "# of threads: $threadQty"
echo "Index export field: $indexExportFieldName"
echo "Query export field: $queryExportFieldName"
echo "Candidate provider parameters: $candProvParams"
echo "Resource parameters: $commonResourceParams"
echo "A # of hard/medium/easy samples per query: $hardNegQty/$sampleMedNegQty/$sampleEasyNegQty"
echo "A max. # of candidate records to generate training data: $candTrainQty"
echo "A max. # of candidate records to generate test data: $candTestQty"
echo "Max train query # param.: $maxNumQueryTrainParam"
echo "Max test/dev query # param.: $maxNumQueryTestParam"
echo "Case handling param: $handleCaseParam"
echo "========================================================"
