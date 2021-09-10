#!/bin/bash
source ./common_proc.sh
source ./config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi
field_name=$2
if [ "$field_name" = "" ] ; then
    echo "Specify a field name (should be both query and index field), e.g., text (2d arg)"
  exit 1
fi

inputDataDir="$COLLECT_ROOT/$collect/"

questFile="$inputDataDir/$DERIVED_DATA_SUBDIR/$BITEXT_SUBDIR/question_$field_name"
answFile="$inputDataDir/$DERIVED_DATA_SUBDIR/$BITEXT_SUBDIR/answer_$field_name"

if [ ! -f "$questFile" ] ; then
    echo "Question/query file '$questFile' is missing, did you specify the correct field?"
    exit 1
fi
if [ ! -f "$answFile" ] ; then
    echo "Answer/passage file '$answFile' is missing, did you specify the correct field?"
    exit 1
fi

wcq=(`wc $questFile`)
wca=(`wc $answFile`)

echo -e "# of questions:\t${wcq[0]}"
echo -e "# of answers:\t${wca[0]}"
qt=`awk "BEGIN{printf(\"%.1f\", ${wcq[1]}/${wcq[0]})}"`
echo -e "# of terms per question:\t$qt"
at=`awk "BEGIN{printf(\"%.1f\", ${wca[1]}/${wca[0]})}"`
echo -e "# of terms per answer:\t$at"

