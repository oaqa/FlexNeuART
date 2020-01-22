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
  collections/msmarco_doc/input_data
```
