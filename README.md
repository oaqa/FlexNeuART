## FlexNeuART (flex-noo-art)
Flexible classic and NeurAl Retrieval Toolkit, or shortly `FlexNeuART` (**intended pronunciation** flex-noo-art) 
is a substantially reworked [`knn4qa` package](knn4qa.md).  The overview can be found in our EMNLP OSS workshop paper: 
[Flexible retrieval with NMSLIB and FlexNeuART, 2020. Leonid Boytsov, Eric Nyberg](https://arxiv.org/abs/2010.14848).


In Aug-Dec 2020, we used this framework to generate best traditional and/or neural runs 
in the [MSMARCO Document ranking task](https://microsoft.github.io/msmarco/#docranking).
In fact, our best traditional (non-neural) run slightly outperformed a couple of neural submissions.
The code for the best-performing neural model will be published within 2-3 months.


`FlexNeuART` is under active development. More detailed description and documentaion is to appear. Currently we have:

* [The installation instructions](INSTALL.md)
* [Python API for retrieval and re-reranking](scripts/py_flexneuart/README.md)
* [Former life (as a knn4qa package), including acknowledgements and publications](knn4qa.md)
* Collection-specific:
   * [MS MARCO](scripts/data_convert/msmarco/README.md)
   * [Yahoo Answers](scripts/data_convert/yahoo_answers/README.md)


For neural network training FlexNeuART incorporates
a re-worked variant of CEDR ([MacAvaney et al' 2019](https://github.com/Georgetown-IR-Lab/cedr)).



