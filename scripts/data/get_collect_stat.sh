#!/bin/bash
. scripts/common.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr, stackoverflow, squad (1st arg)"
  exit 1
fi

PARTS=("train" "dev1" "dev2" "test")

if [ "$collect" = "manner" ] ; then
  PARTS+=("tran")
elif [ "$collect" = "compr" -o "$collect" = "stackoverflow" ] ; then
  PARTS+=("tran")
elif [ "$collect" = "squad" ] ; then
  PARTS+=("tran" "wiki")
  full_text_prefix="full_"
else
  echo "Unknown collection: $collect"
  exit 1
fi 

total_qty=0

h1=`echo -n "q-total\\ta-total\\t"`
h3=`echo -n "q-tot-#terms\\ta-tot-#terms\\t"`

for part in ${PARTS[*]}  ; do
  h1=$h1`echo -n "q-$part-qty\\ta-$part-qty\\t"`
  h3=$h3`echo -n "q-$part-#term\\ta-$part-#term\\t"`
done

# We do have empty question/answers sometimes.
# Hence total # of question/answers is inferred from Solr*File.txt
# However, the latter cannot be used to compute average number
# of terms because they contain tags (in addition to text).

wcq1=(`wc output/$collect/*/SolrQuestionFile.txt|grep total`)
wca1=(`wc output/$collect/*/SolrAnswerFile.txt|grep total`)
h2=`echo -n "${wcq1[0]}\\t${wca1[0]}\\t"`

wcq=(`cat output/$collect/*/${full_text_prefix}question_text|uniq|wc`)
wca=(`cat output/$collect/*/${full_text_prefix}answer_text|uniq|wc`)
qt=`awk "BEGIN{printf(\"%.1f\", ${wcq[1]}/${wcq[0]})}"`
at=`awk "BEGIN{printf(\"%.1f\", ${wca[1]}/${wca[0]})}"`
h4=`echo -n "$qt\\t$at\\t"`

for part in ${PARTS[*]}  ; do
  c=$collect
  if [ "$part" = "tran" -a "$collect" = "manner" ] ; then
    c=ComprMinusManner
  fi
  wcq1=(`wc output/$c/$part/SolrQuestionFile.txt`)
  wca1=(`wc output/$c/$part/SolrAnswerFile.txt`)
  h2=$h2`echo -n "${wcq1[0]}\\t${wca1[0]}\\t"`

  wcq=(`cat output/$c/$part/${full_text_prefix}question_text|uniq|wc`)
  wca=(`cat output/$c/$part/${full_text_prefix}answer_text|uniq|wc`)
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


