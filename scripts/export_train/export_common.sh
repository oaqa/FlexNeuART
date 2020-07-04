

checkVarNonEmpty "DEV1_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"

outSubdir=""
threadQty=1
sampleNegQty=10
candTrainQty=500
candTestQty=10

POS_ARGS=()

while [ $# -ne 0 ] ; do
  # set +e prevents the whole bash script from failing
  set +e
  echo $1|grep "^-" >/dev/null
  retStat=$?
  set -e
  if [ $retStat = 0 ] ; then
    OPT_NAME="$1"
    OPT_VALUE="$2"
    OPT="$1 $2"
    if [ "$OPT_VALUE" = "" ] ; then
      echo "Option $OPT_NAME requires an argument." >&2
      exit 1
    fi
    shift 2
    case $OPT_NAME in
      -thread_qty)
        outSubdir=$OPT_VALUE
        ;;
      -out_subdir)
        threadQty=$OPT_VALUE
        ;;
      -sample_neg_qty)
        sampleNegQty=$OPT_VALUE
        ;;
      -cand_train_qty)
        candTrainQty=$OPT_VALUE
        ;;
      -cand_test_qty)
        candTestQty=$OPT_VALUE
        ;;
      -max_num_query_train)
        maxNumQueryTrainParam=$OPT
        ;;
      -max_num_query_test)
        maxNumQueryTestParam=$OPT
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
  echo "Specify sub-collection (1st positional arg), e.g., squad"
  exit 1
fi
indexFieldName=${POS_ARGS[1]}
if [ "$indexFieldName" = "" ] ; then
  echo "Specify the name of the index field (2d positional arg)"
  exit 1
fi
partTrain=${POS_ARGS[2]}
if [ "$partTrain" = "" ] ; then
  echo "Specify the training sub-dir, e.g., $DEFAULT_TRAIN_SUBDIR (3d positional arg)"
  exit 1
fi
partTest=${POS_ARGS[3]}
if [ "$partTest" = "" ] ; then
  echo "Specify the training sub-dir, e.g., $DEV1_SUBDIR (4th positional arg)"
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
