#!/bin/bash -e

source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "DEFAULT_TRAIN_SUBDIR"
checkVarNonEmpty "DEFAULT_QUERY_FIELD_NAME"
checkVarNonEmpty "DEV1_SUBDIR"
checkVarNonEmpty "QUESTION_FILE_JSONL"
checkVarNonEmpty "QREL_FILE"

boolOpts=("h" "help" "print help")

usageMain="<collection> <token field name> <train part>"

partDev=""
minRelGrade="1"
queryFieldName="$DEFAULT_QUERY_FIELD_NAME"

paramOpts=(
"part_dev"               "partDev"             "development data sub-directory (optional)"
"part_test"              "partTest"            "test data sub-directory (optional)"
"query_field_name"       "queryFieldName"      "the name of the query field"
"min_rel_grade"          "minRelGrade"         "minimum grade of a relevant document (default $minRelGrade)"
)

parseArguments $@

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

tokFieldName=${posArgs[1]}
if [ "$tokFieldName" = "" ] ; then
  genUsage "$usageMain" "Specify the name of the token field (2d arg)"
  exit 1
fi

partTrain=${posArgs[2]}
if [ "$partTrain" = "" ] ; then
  genUsage "$usageMain" "Specify the training sub-dir, e.g., $DEFAULT_TRAIN_SUBDIR (3d arg)"
  exit 1
fi


colRoot="$COLLECT_ROOT/$collect"
fwdIndexHead="$colRoot/$FWD_INDEX_SUBDIR/$tokFieldName"
inpDataRoot="$colRoot/$INPUT_DATA_SUBDIR/"
inf0=`head -1 $fwdIndexHead`
dummy=' ' read -r -a indQtys <<< "$inf0"

docQty=${indQtys[0]}
totTokQty=${indQtys[1]}
tokPerDoct=$(div1 $totTokQty $docQty)

totQueryQty=0
totRelQty=0
totQueryTokQty=0

partQueryQtys=()

partArr=($partTrain $partDev $partTest)

for part in ${partArr[*]} ; do
  inpDir="$colRoot/$INPUT_DATA_SUBDIR/$part/"
  questFile="$inpDir/$QUESTION_FILE_JSONL"
  inf1=$(scripts/report/count_toks.py --input $questFile --field $queryFieldName)
  dummy=' ' read -r -a tokQtys1 <<< "$inf1"

  queryQty=${tokQtys1[0]}
  partQueryQtys+=($queryQty)

  totQueryQty=$(($totQueryQty+$queryQty))
  totQueryTokQty=$(($totQueryTokQty+${tokQtys1[1]}))

  qrelFile="$inpDir/$QREL_FILE"
  relQty=$(scripts/report/count_qrels.py --min_rel_grade $minRelGrade --input $qrelFile 2>/dev/null)
  totRelQty=$(($totRelQty+$relQty))
done

tokPerQuest=$(div1 $totQueryTokQty $totQueryQty)
relPerQuest=$(div1 $totRelQty $totQueryQty)

# HEADER
echo -ne "dataset"

for part in ${partArr[*]} ; do
  echo -ne "\t$part"
done

echo -en "\t#docs\t#rel per query\tquery #tok\tdoc #tok"

echo ""

# DATA
echo -ne "$collect"

pid=0
for part in ${partArr[*]} ; do
  echo -ne "\t"$(humnReadNums ${partQueryQtys[$pid]})
  pid=$(($pid+1))
done

echo -ne "\t"$(humnReadNums $docQty)

echo -ne "\t$relPerQuest\t$tokPerQuest\t$tokPerDoct"

echo ""
