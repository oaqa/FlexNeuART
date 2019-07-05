#!/bin/bash -e

col=$1

[ "$col" != "" ] || ( echo "Specify input collection (1st arg)" ; exit 1;)

queryField=text
mainFieldName=text
candQty=100
negQtyArr=("10" "-1")

if [ "$col" = "manner" ] ; then
  partArr=("train" "dev1")
  negQtyDescArr=("neg10" "allCand")
  maxNumQueryQtyArr=("" "")
elif [ "$col" = "compr" ] ; then
  partArr=("tran" "dev1")
  negQtyDescArr=("neg10" "allCand")
  maxNumQueryQtyArr=("-max_num_query 500000" "")
fi

for partId in 0 1 ; do 
  negQty=${negQtyArr[$partId]}
  negQtyDesc=${negQtyDescArr[$partId]}
  part=${partArr[$partId]}
  maxNumQueryQty=${maxNumQueryQtyArr[$partId]}

  outDir="$PWD/matchZooTrain/$col"

  [ -d "$outDir" ] || (mkdir "$outDir" ; echo "Created output directory: $outDir")

  scripts/data/run_export_train_text_pairs.sh \
      -cand_qty $candQty \
      $maxNumQueryQty \
      -export_fmt match_zoo  \
      -memindex_dir memfwdindex/$col/ \
      -out_file $outDir/${part}_${negQtyDesc}.tsv \
      -thread_qty 8 \
      -sample_neg_qty $negQty \
      -query_field $queryField \
      -q output/$col/$part/SolrQuestionFile.txt \
      -u lucene_index/$col/ \
      -field_name $mainFieldName \
      -qrel_file output/$col/$part/qrels_all_binary.txt 
done
