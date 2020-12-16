# MSMARCO
1. [Document ranking](DOCS.md)
2. [Passage ranking](PASSAGES.md)
3. Step-by-step notebooks to reproduce our run submitted 
[to the MS MARCO leaderboard in December 2020](https://microsoft.github.io/msmarco/#docranking) (see [also our write-up](https://arxiv.org/abs/2012.08020)):
    * [One notebook](MSMARCO_docs_2020-12-06_complete.ipynb) reproduces all steps necessary to download the data, preprocess it, and train all the models.
    * [The second notebook](MSMARCO_docs_2020-12-06_processed_data_and_precomp_model1.ipynb) operates on preprocessed data in FlexNeuART JSONL format. It does not require running GIZA to generate IBM Model 1 (these models are already trained).
