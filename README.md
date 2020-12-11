## FlexNeuART
Flexible classic and NeurAl Retrieval Toolkit, or shortly `FlexNeuART` (**intended pronunciation** flex-noo-art) 
is a substantially reworked `knn4qa` (see below) package.  
The overview can be found in our EMNLP OSS workshop paper: 
[Flexible retrieval with NMSLIB and FlexNeuART, 2020. Leonid Boytsov, Eric Nyberg](https://arxiv.org/abs/2010.14848).


In Aug-Dec 2020, we used this framework to generate best traditional and/or neural runs 
in the [MSMARCO Document ranking task](https://microsoft.github.io/msmarco/#docranking).
In fact, our best traditional (non-neural) run slightly outperformed a couple of neural submissions.


`FlexNeuART` is under active development. More detailed description and documentaion is to appear. Currently we have:

* [The installation instructions](INSTALL.md)
* Collection-specific:
   * [MS MARCO](scripts/data_convert/msmarco/README.md)
   * [Yahoo Answers](scripts/data_convert/yahoo_answers/README.md)


For neural network training FlexNeuART incorporates
a re-worked variant of CEDR ([MacAvaney et al' 2019](https://github.com/Georgetown-IR-Lab/cedr)).

## Former life (as a knn4qa package)

This is a learning-to-rank pipeline, which is a part of the project where we study applicability of k-nearest neighbor search methods to IR and QA applications. This project is supported primarily by the NSF grant **#1618159** : "[Matching and Ranking via Proximity Graphs: Applications to Question Answering and Beyond](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1618159&HistoricalAwards=false)". 

Previous work can be found in the following branches branches:

1. [The ``bigger_reruns`` branch includes software](https://github.com/oaqa/knn4qa/tree/bigger_reruns) used in the dissertation of Leonid Boytsov: ["Efficient and Accurate Non-Metric k-NN Search with Applications to Text Matching"](http://boytsov.info/pubs/thesis_boytsov.pdf). A summary of this work is given in the [following blog post.](http://searchivarius.org/blog/efficient-and-accurate-non-metric-k-nn-search-applications-text-matching-we-need-more-k-nn).
2. [The ``cikm2016`` branch](https://github.com/oaqa/knn4qa/tree/cikm2016) includes software for the paper [L. Boytsov, D. Novak, Y. Malkov, E. Nyberg  (2016). *Off the Beaten Path: Let’s Replace Term-Based Retrieval
with k-NN Search*, CIKM'16](http://boytsov.info/pubs/cikm2016.pdf). This work is covered [in a blog post as well.](http://searchivarius.org/blog/text-retrieval-can-and-should-benefit-using-generic-k-nn-search-algorithms) **For more details on this branch software, please, check [the Wiki page](https://github.com/oaqa/knn4qa/wiki)**.

NB: Our [``cikm2016`` branch](https://github.com/oaqa/knn4qa/tree/cikm2016) can be also used to **partially** reproduce results from the paper: [M Surdeanu, M Ciaramita, H Zaragoza. *Learning to rank answers to non-factoid questions from web collections* 
Computational Linguistics, 2011 ](http://www.mitpressjournals.org/doi/pdfplus/10.1162/COLI_a_00051) 


