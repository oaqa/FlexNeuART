#@IgnoreInspection BashAddShebang


checkVarNonEmpty "DEV1_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"

# This default is set by a calling script
checkVarNonEmpty "outSubdir"

threadQty=1
sampleNegQty=10
candTrainQty=500
candTestQty=10

boolOpts=("h" "help" "print help")

paramOpts=(
"thread_qty"          "threadQty"        "# of threads"
"out_subdir"          "outSubdir"        "output sub-directory (default $outSubdir)"
"sample_neg_qty"      "sampleNegQty"     "A # of negative samples per query or -1 to keep all candidate entries"
"cand_train_qty"      "candTrainQty"     "A max. # of candidate records to generate training data"
"cand_test_qty"       "candTestQty"      "A max. # of candidate records to generate test data"
"max_num_query_train" "maxNumQueryTrain" "Optional max. # of train queries"
"max_num_query_test"  "maxNumQueryTest"  "Optional max. # of test/dev queries"
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

indexFieldName=${posArgs[1]}
if [ "$indexFieldName" = "" ] ; then
  genUsage "$usageMain" "Specify the name of the index field (2d arg)"
  exit 1
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
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "QUERY_FIELD_NAME"
checkVarNonEmpty "QREL_FILE"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
luceneIndexDir="$COLLECT_ROOT/$collect/$LUCENE_INDEX_SUBDIR/"

outDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$outSubdir/$indexFieldName"

if [ ! -d "$outDir" ] ; then
  mkdir -p "$outDir"
fi

echo "========================================================"
echo "Train split: $partTrain"
echo "Eval split: $partTest"
echo "Output directory: $outDir"
echo "# of threads: $threadQty"
echo "A # of negative samples per query: $sampleNegQty"
echo "A max. # of candidate records to generate training data: $candTrainQty"
echo "A max. # of candidate records to generate test data: $candTestQty"
echo "Max train query # param.: $maxNumQueryTrainParam"
echo "Max test/dev query # param.: $maxNumQueryTestParam"
echo "========================================================"
