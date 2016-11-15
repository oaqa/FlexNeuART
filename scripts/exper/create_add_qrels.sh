#!/bin/bash
MAX_GRADE=4
for f_src in `find output -name qrels_all_binary.txt` ; do 
  echo "Processing $f_src -> $f_dst"
  f_dst=`echo $f_src|sed 's/qrels_all_binary/qrels_all_graded_same_score/'`
  cat "$f_src" | awk '{print $1" "$2" "$3" 4"}' > $f_dst
done
