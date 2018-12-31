#/bin/bash -e
N=5000000 
for col in compr stackoverflow ; do 
  for field in text text_unlemm ; do
    scripts/giza/sample_tran.py  output/ $col $field tran tran.sample5M $N 
  done
done

