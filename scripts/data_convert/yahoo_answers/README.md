# Basic data preparation
This example covers a Manner subset of the Yahoo Answers Comprehensive.
However, a similar procedure can be applied to a bigger collection. All
experiments assume the variable `COLLECT_ROOT` in the script `scripts/config.sh` 
is set to `collections` and that all collections are stored in the `collections`
sub-directory (relative to the source code root).


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

Finally, we can create input data in the JSON format. Note that the last argument defines a 
part of the collection that is used to create a parallel corpus (i.e,
a bitext), which is generated in addition to JSON input files:
```
scripts/data_convert/yahoo_answers/convert_yahoo_answers.sh \
  manner \
  dev1,dev2,test,train,bitext \
  bitext
```

# Indexing
Before creating Lucene index (if collections was resplit), please,
first delete Lucene caches:
```
rm -rf collections/manner/lucene_cache/
```

Then, you can create a Lucene index:
```
scripts/index/create_lucene_index.sh manner
```


Create a forward index, mapdb, generates the fastest (but not the
smallest forward index):
```
scripts/index/create_fwd_index.sh \
  manner \
  mapdb \
  'text:parsedBOW text_unlemm:parsedText text_bert_tok:parsedText text_raw:raw'
```
More detailed explanation of index types is below. Note that
there are two types of the field: a parsed text field and a raw field.
The indexer white-space tokenizes text fields and compiles token statistics. 
1. `parsedBOW` index keeps only a bag-of-words;
2. `parsedText` keeps the original word sequence;
3. `raw` is the index that stores text "as is" without any changes.

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
  dev2 \
  -thread_qty 4 \
  -sample_neg_qty 20
```
In this case, the output goes to:
```
collections/manner/derived_data/cedr_train/text_raw
```
Note that we use `dev2` here, so that we can use `dev1` **to evaluate fusion results**.

Afterwards, one can train using the following commands. Setting convenience variables:
```
export train_subdir=cedr_train/text_raw
export dpath=collections/manner
export mtype=vanilla_bert
export max_doc_len=512
export max_query_len=64
export grad_checkpoint_param=0
export backprop_batch_size=1
export batches_per_epoch=1024
export batch_size_val=16
export max_query_val=5000


```
Starting a training script:
```
python -u scripts/cedr/train.py \
    --model $mtype \
    --datafiles $dpath/derived_data/$train_subdir/data_query.tsv  \
                    $dpath/derived_data/$train_subdir/data_docs.tsv \
    --train_pairs $dpath/derived_data/$train_subdir/train_pairs.tsv \
   --valid_run $dpath/derived_data/$train_subdir/test_run.txt \
   --qrels $dpath/derived_data//$train_subdir/qrels.txt \
   --initial_bert_weights $dpath/derived_data/lm_finetune_model/pytorch_model.bin \
   --model_out_dir $dpath/derived_data/ir_models/$mtype \
   --batches_per_train_epoch $batches_per_epoch --init_lr 1e-3 --init_bert_lr 2e-5 \
    --epoch_qty 30 --epoch_lr_decay 0.9 \
    --backprop_batch_size $backprop_batch_size --batch_size 32 --batch_size_val $batch_size_val \
    --grad_checkpoint_param $grad_checkpoint_param \
    --max_doc_len $max_doc_len --max_query_len $max_query_len
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
  <MGIZA DIRECTORY>
```

It further needs to cleaned-up and converted to a binary format.
```
export min_tran_prob=0.001
export top_word_qty=100000


scripts/giza/filter_tran_table_and_voc.sh \
  manner \
  text_unlemm \
  $min_tran_prob \
  $top_word_qty
```


# Running basic experiments
## Generate descriptors to tune BM25

```
mkdir -p collections/manner/exper_desc
```

```
scripts/gen_exper_desc/gen_bm25_tune_json_desc.py \
  --outdir collections/manner/exper_desc/ \
  --exper_subdir bm25tune \
  --rel_desc_path exper_desc
```
The main experimental descriptor is going to be stored in 
`collections/manner/exper_desc/bm25tune.json`,
whereas auxiliary descriptors are stored in `collections/manner/exper_desc/bm25tune/`

Now we can run tuning experiments where we train on `train` and test on `dev1`:
```
scripts/exper/run_experiments.sh \
  manner \
  exper_desc/bm25tune.json \
  -test_part dev1
```

Obtain experimental results and find the best configuration 
with respect to the Mean Average Precision (MAP):
```
scripts/report/get_exper_results.sh \
  manner \
  exper_desc/bm25tune.json \
  bm25tune.tsv \
  -test_part dev1 \
  -flt_cand_qty 250 \
  -print_best_metr map
```
The results are saved to `bm25tune.tsv` and include only runs
with top *k*=250 candidates. The result extraction script
will print something like this:
```
Including only runs that generated 250 candidate records
Best results for metric map:
Value: 0.116200
Result sub-dir: bm25tune/bm25tune/bm25_k1=0.4_b=0.6
```

