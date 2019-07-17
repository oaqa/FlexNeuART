#!/bin/bash
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

wcq=(`wc $inputDataDir/$BITEXT_SUBDIR/question_text`)
wca=(`wc $inputDataDir/$BITEXT_SUBDIR/answer_text`)
echo -e "# of questions:\t${wcq[0]}"
echo -e "# of answers:\t${wca[0]}"
qt=`awk "BEGIN{printf(\"%.1f\", ${wcq[1]}/${wcq[0]})}"`
echo -e "# of terms per question:\t$qt"
at=`awk "BEGIN{printf(\"%.1f\", ${wca[1]}/${wca[0]})}"`
echo -e "# of terms per answer:\t$at"

