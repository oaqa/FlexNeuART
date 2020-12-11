# MSMARCO TREC 2019 document ranking
## Basic data preparation
Create raw-data directory and download data:
```
mkdir -p collections/msmarco_doc/input_raw

scripts/data_convert/msmarco/download_msmarco_doc.sh \
  collections/msmarco_doc/input_raw
```
Create the directory to store pre-processed data and run the conversion:
```
mkdir -p collections/msmarco_doc/input_data 

scripts/data_convert/msmarco/convert_msmarco_doc.sh \
  collections/msmarco_doc/input_raw  \
  msmarco_doc
```

## Indexing
Create a Lucene index:
```
scripts/index/create_lucene_index.sh msmarco_doc
```

Create a forward index:
```
scripts/index/create_fwd_index.sh msmarco_doc mapdb \
  'title_unlemm:parsedText url_unlemm:parsedText text:parsedText body:parsedText text_bert_tok:parsedText text_raw:raw'
```
 ## Optionally splitting training queries
 
This is necessary to carve out smaller subsets 
for fusion and development. The original
development set can be used for testing (then the
corresponding folder needs to be renamed accordingly):

```
scripts/data_convert/split_queries.sh \
  msmarco_doc train train_fusion tmp -part1_qty 10000
scripts/check_utils/check_split_queries.sh   \
  msmarco_doc train train_fusion tmp

scripts/data_convert/split_queries.sh \
  msmarco_doc tmp dev_add bitext -part1_qty 5000
scripts/check_utils/check_split_queries.sh \
  msmarco_doc tmp dev_add bitext
```


