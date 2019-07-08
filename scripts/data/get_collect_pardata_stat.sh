#!/bin/bash
. scripts/common.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

wcq=(`wc $collect/input_data/bitext/question_text`)
wca=(`wc $collect/input_data/bitext/answer_text`)
echo -e "# of questions:\t${wcq[0]}"
echo -e "# of answers:\t${wca[0]}"
qt=`awk "BEGIN{printf(\"%.1f\", ${wcq[1]}/${wcq[0]})}"`
echo -e "# of terms per question:\t$qt"
at=`awk "BEGIN{printf(\"%.1f\", ${wca[1]}/${wca[0]})}"`
echo -e "# of terms per answer:\t$at"

