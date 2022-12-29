#!/bin/bash

# A wrapper that compares all key metrics between our code and trec_eval
for metric in  ndcg@10 ndcg@20 ndcg@100 recip_rank mrr map map@10 map@100 recall@10 recall@20 recall@100 ; do
    echo "Testing metric $metric"
    ./check_utils/compare_eval_tools.py \
        --qrels data/sample_runs/trec2022/qrels.txt.bz2 \
        --run data/sample_runs/trec2022/run_100.txt.bz2  \
        --eval_metric $metric 
    if [ "$?" != "0" ] ; then
        echo "Check for metric=$metric failed!"
        exit 1
    fi
done

for metric in ndcg@10 ndcg@20 ndcg@100 recip_rank mrr map map@10 map@100 recall@10 recall@20 recall@100 ; do
    echo "Testing metric $metric"
    ./check_utils/compare_eval_tools.py \
        --qrels data/sample_runs/manner_dev1/qrels.txt.bz2 \
        --run data/sample_runs/manner_dev1/run.txt.bz2  \
        --eval_metric $metric 
    if [ "$?" != "0" ] ; then
        echo "Check for metric=$metric failed!"
        exit 1
    fi
done
echo "All tests are successful!"
