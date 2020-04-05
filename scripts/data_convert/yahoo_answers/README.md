# Basic data preparation
This example covers a Manner subset of the Yahoo Answers Comprehensive.
However, a similar procedure can be applied to a bigger collection.


Create raw-data directory and store raw data there:
```
mkdir -p collections/manner/input_pre_raw

cp <data path>/manner.xml.bz2 collections/manner/input_pre_raw/
```

Now, we need to split the data. The following command creates  several
training and testing subsets, including a ``bitext`` subset that can
be used to train either IBM Model 1 or a neural IR model. 
We would reserve a much smaller ``train`` data set to train
a fusion/LETOR model that could combine several signals:
```
mkdir -p collections/manner/input_raw/

scripts/data_convert/yahoo_answers/split_yahoo_answers_input.sh \
  -i collections/manner/input_pre_raw/manner.xml.bz2  \
  -o collections/manner/input_raw/manner-v2.0 \
  -n dev1,dev2,test,train,bitext \
  -p 0.05,0.05,0.1,0.1,0.7
```


Create input data in the JSON format, note the last argument defines a 
part of the collection that is used to create a parallel corpus (i.e,
a bitext):
```
scripts/data_convert/yahoo_answers/convert_yahoo_answers.sh \
  manner \
  dev1,dev2,test,train,bitext \
  bitext
```

# Indexing
Create a Lucene index:
```
scripts/index/create_lucene_index.sh manner
```


Create a forward index, mapdb, generates the fastest (but not the
smallest forward index), ``parsedBOW`` denotes 
indexing without keeping positional information (i.e.,
a bag-of-words indexing):
```
scripts/index/create_fwd_index.sh \
  manner \
  mapdb \
  'text:parsedBOW text_unlemm:parsedBOW text_raw:raw'
```

# Generating & using optional (derived) data


## Training CEDR neural ranking models

First, we need to export training data, one can optionally limit the number of
generated queries via `-max_num_query_test`. The following command
generates training data in the CEDR format for the collection `manner`
and the field `text_raw`. The traing data is generated from the split `bitext`, 
whereas split `dev1` is used to generate eval data:
```
scripts/export_train/export_cedr.sh \
  manner \
  text_raw \
  bitext \
  dev1 \
  -sample_neg_qty 20
```
In this case, the output goes to:
```
<collections root>/manner/derived_data/cedr_train/text_raw
```

## Generating a vocabulary file

**This would need to be refactored**:

```
scripts/cedr/build_vocab.py \
  --field text_unlemm \
  --input collections/manner/input_data/bitext/AnswerFields.jsonl.gz  \
          collections/manner/input_data/train/AnswerFields.jsonl.gz  \
  --output collections/manner/derived_data/vocab/text_unlemm.voc 
```

## Training an IBM Model 1 model

Here we create a model for the field ```text_unlemm```. To do
so one needs to download and compile MGIZA:
```
scripts/giza/create_tran.sh \
  manner \
  text_unlemm \
  <MGIZA DIRECTORY>/mgiza/
```





