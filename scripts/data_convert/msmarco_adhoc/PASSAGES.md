# MSMARCO TREC 2019 passage ranking
## Basic data preparation

```
mkdir -p collections/msmarco_pass/input_raw

scripts/data_convert/msmarco_adhoc/download_msmarco_pass.sh \
  collections/msmarco_doc/input_raw
```

Convert data:
```
scripts/data_convert/msmarco_adhoc/convert_msmarco_pass.sh \
  collections/msmarco_pass/input_raw/ msmarco_pass
```