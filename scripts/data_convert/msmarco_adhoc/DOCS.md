# Basic data preparation
Create raw-data directory and download data:
```
mkdir -p collections/msmarco_doc/input_raw

scripts/data_convert/msmarco_adhoc/download_msmarco_doc.sh \
  collections/msmarco_doc/input_raw
```
Create the directory to store pre-processed data and run the conversion:
```
mkdir -p collections/msmarco_doc/input_data 

scripts/data_convert/msmarco_adhoc/convert_msmarco_doc.sh \
  collections/msmarco_doc/input_raw  \
  msmarco_doc
```
# Optionally splitting the train and development parts
```
scripts/data_convert/split_queries.sh \
  msmarco_doc dev dev1 dev2 -part1_qty 3000
scripts/qa/check_split_queries.sh \
  msmarco_doc dev dev1 dev2

scripts/data_convert/split_queries.sh \
  msmarco_doc train train1 train_bitext -part1_qty 10000
scripts/qa/check_split_queries.sh   \
  msmarco_doc train train1 train_bitext
```


# Indexing
Create a Lucene index:
```
scripts/index/create_lucene_index.sh msmarco_doc
```

Create a forward index:
```
scripts/index/create_fwd_index.sh msmarco_doc mapdb \
  'title:parsedBOW text:parsedText body:parsedText text_raw:raw'
```
 