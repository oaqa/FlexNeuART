# Basic data preparation
Create raw-data directory and download data:
```
mkdir collections/msmarco_doc/input_raw -p 
scripts/data_convert/msmarco_adhoc/download_msmarco_doc.sh \
  collections/msmarco_doc/input_raw
```
Create the directory to store pre-processed data and run the conversion:
```
mkdir collections/msmarco_doc/input_data 
scripts/data_convert/msmarco_adhoc/convert_msmarco_doc.sh \
  collections/msmarco_doc/input_raw  \
  msmarco_doc
```
# Optionally splitting the train and development parts
scripts/data_convert/split_train4bitext.sh msmarco_doc 10000

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
 