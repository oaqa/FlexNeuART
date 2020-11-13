# MSMARCO TREC 2019 passage ranking
## Basic data preparation

```
mkdir -p collections/msmarco_pass/input_raw

scripts/data_convert/msmarco_adhoc/download_msmarco_pass.sh \
  collections/msmarco_pass/input_raw
```

Convert data:
```
scripts/data_convert/msmarco_adhoc/convert_msmarco_pass.sh \
  collections/msmarco_pass/input_raw/ msmarco_pass
```

## Indexing
Create a Lucene index:
```
scripts/index/create_lucene_index.sh msmarco_pass
```

Create a forward index:
```
scripts/index/create_fwd_index.sh msmarco_pass mapdb \
  'text:parsedText text_unlemm:parsedText text_bert_tok:parsedText text_raw:raw'
```

## Optionally splitting the train part
```
scripts/data_convert/split_queries.sh \
  msmarco_pass train train1 bitext -part1_qty 20000
scripts/qa/check_split_queries.sh   \
  msmarco_pass train train1 bitext
```