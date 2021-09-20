[![Pypi version](https://img.shields.io/pypi/v/flexneuart.svg)](http://pypi.python.org/pypi/flexneuart)

## FlexNeuART (flex-noo-art)
Flexible classic and NeurAl Retrieval Toolkit, or shortly `FlexNeuART` (**intended pronunciation** flex-noo-art) 
is a substantially reworked [`knn4qa` package](knn4qa.md).  The overview can be found in our EMNLP OSS workshop paper: 
[Flexible retrieval with NMSLIB and FlexNeuART, 2020. Leonid Boytsov, Eric Nyberg](https://arxiv.org/abs/2010.14848).

In Aug-Dec 2020, we used this framework to generate best traditional and/or neural runs 
in the [MSMARCO Document ranking task](https://microsoft.github.io/msmarco/#docranking).
In fact, our best traditional (non-neural) run slightly outperformed a couple of neural submissions.
The code for the best-performing neural model will be published within 2-3 months. This model is described in our ECIR 2021 paper:
[Boytsov, Leonid, and Zico Kolter. "Exploring Classic and Neural Lexical Translation Models for Information Retrieval: Interpretability, Effectiveness, and Efficiency Benefits." ECIR 2021](https://arxiv.org/abs/2102.06815).

## Main features

* Dense, sparse, or dense-sparse retrieval using Lucene and NMSLIB.
* Multi-field multi-level forward indices (+parent-child field relations) that can store 
  parsed and "raw" text input as well as sparse and dense vectors.
* Pluggable generic rankers (via a server)
* SOTA neural (BERT-based) and non-neural models
* Multi-GPU training **and** inference with out-of-the box support for ensembling
* Basic experimentation framework (+LETOR)
* Python API to use retrievers and rankers as well as to access indexed data.


## Documentation

* [Usage notebooks covering installation & most functionality (including experimentation and Python API demo)](demo/README.md)
* [Legacy notebooks for MS MARCO and Yahoo Answers](legacy_docs/README.md)
* [Former life (as a knn4qa package), including acknowledgements and publications](knn4qa.md)

We support [a number of neural BERT-based ranking models](flexneuart/models) as well as strong traditional
ranking models including IBM Model 1 (description of non-neural rankers to follow).

The framework supports data in generic JSONL format. We provide conversion (and in some cases download) scripts for the following collections:
* Cranfield (a small toy collection)
* MS MARCO data v1 and v2 (documents and passages)
* Wikipedia DPR (Natural Questions, Trivia QA, SQuAD)
* Yahoo Answers collections


## Acknowledgements

For neural network training FlexNeuART incorporates
a substantially re-worked variant of CEDR ([MacAvaney et al' 2019](https://github.com/Georgetown-IR-Lab/cedr)).



