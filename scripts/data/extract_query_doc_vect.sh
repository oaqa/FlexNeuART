#!/bin/bash
. scripts/common.sh
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow, squad"
  exit 1
fi

field=$2
if [ "$field" = "" ] ; then
  echo "Specify a field (2d arg), e.g., text"
  exit 1
fi

etype=$3
if [ "$etype" = "" ] ; then
  echo "Specify the output file type (3d arg), e.g., bm25, bm25_share_id, cosine"
  exit 1
fi

query_subsets=$4
if [ "$query_subsets" = "" ] ; then
  echo "Specify comma-separated query subsets (4th arg), e.g., \"dev1,dev2,test\" or \"@\" for empty query set"
  exit 1
fi
if [ "$query_subsets" = "@" ] ; then
  query_subsets=""
fi
query_subsets=`echo $query_subsets|sed 's/,/ /g'`

output_prefix=$5
if [ "$output_prefix" = "" ] ; then
  echo "Specify the output prefix (5th arg)"
  exit 1
fi

test_qty=$6
if [ "$test_qty" = "" ] ; then 
  echo "Specify a number of queries to test correctness, e.g., 100. Note this should be a small value"
  exit 1
fi

INDEX_DIR="memfwdindex/$collect/"

cmd="scripts/data/run_extr_query_doc_vect.sh -field $field  -memindex_dir memfwdindex/$collect/ -out_data ${output_prefix}_data_${etype}.txt  -extr_type $etype -test_qty $test_qty"
for part in $query_subsets ; do
  cmd="$cmd -in_queries output/$collect/$part/SolrQuestionFile.txt -out_queries ${output_prefix}_queries_${etype}_${part}.txt "
done

bash -c "$cmd"
check "$cmd"
