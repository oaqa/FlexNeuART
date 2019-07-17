#!/bin/bash
source scripts/common_proc.sh
source scripts/config.sh
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

retVal=""
getIndexQueryDataInfo "$inputDataDir"
dataFileName=${retVal[1]}
queryFileName=${retVal[2]}
if [ "$dirList" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
if [ "$dataFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
else
  echo "Using the data input files: $dataFileName, $queryFileName"
fi
dirList=`echo ${retVal[0]}|sed 's/,/ /g'`

total_qty=0

h1=`echo -n "q-total\\ta-total\\t"`
h3=`echo -n "q-tot-#terms\\ta-tot-#terms\\t"`

for part in ${dirList[*]}  ; do
  h1=$h1`echo -n "q-$part-qty\\ta-$part-qty\\t"`
  h3=$h3`echo -n "q-$part-#term\\ta-$part-#term\\t"`
done

# We do have empty question/answers sometimes.
# Hence total # of question/answers is inferred from Solr*File.txt
# However, the latter cannot be used to compute average number
# of terms because they contain tags (in addition to text).

wcq1=(`wc $inputDataDir/*/$queryFileName|grep total`)
wca1=(`wc $inputDataDir/*/$dataFileName|grep total`)
h2=`echo -n "${wcq1[0]}\\t${wca1[0]}\\t"`

wcq=(`cat $inputDataDir/*/question_text|uniq|wc`)
wca=(`cat $inputDataDir/*/answer_text|uniq|wc`)
qt=`awk "BEGIN{printf(\"%.1f\", ${wcq[1]}/${wcq[0]})}"`
at=`awk "BEGIN{printf(\"%.1f\", ${wca[1]}/${wca[0]})}"`
h4=`echo -n "$qt\\t$at\\t"`

for part in ${dirList[*]}  ; do
  wcq1=(`wc $inputDataDir/$part/$queryFileName`)
  wca1=(`wc $inputDataDir/$part/$dataFileName`)
  h2=$h2`echo -n "${wcq1[0]}\\t${wca1[0]}\\t"`

  wcq=(`cat $inputDataDir/$part/question_text|uniq|wc`)
  wca=(`cat $inputDataDir/$part/answer_text|uniq|wc`)
  if [ "${wcq[0]}" != "0" ] ;  then
    qt=`awk "BEGIN{printf(\"%.1f\", ${wcq[1]}/${wcq[0]})}"`
  else
    qt=0
  fi
  at=`awk "BEGIN{printf(\"%.1f\", ${wca[1]}/${wca[0]})}"`
  h4=$h4`echo -n "$qt\\t$at\\t"`
done

echo -e $h1
echo -e $h2
echo -e $h3
echo -e $h4


