## Wikipedia DPR data

This directory contains scripts download and process
the [Wikipedia DPR data, created by
Facebook Research](https://github.com/facebookresearch/DPR). Although,
the code is fully functional 
detailed instructions are not available yet. In a nutshell, one needs to

1. Download data: [passages](download_dpr_passages.sh) and [queries](download_dpr_queries.sh). 
We suggest placing them in a collection sub-directory such as ``download``
2. Each DPR dataset comes with the training set, which we split into three subsets:
`bitext` (regular training data), `dev` (development), and `train_fusion` (a
set to learn a fusion model). Splitting and processing the queries can be done using 
the [following script](split_and_convert_dpr_queries.sh).