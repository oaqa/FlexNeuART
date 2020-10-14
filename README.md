## FlexNeuART
Flexible classic and NeurAl Retrieval Toolkit, or shortly `FlexNeuART` (**intended pronunciation** flex-noo-art) is a substantially reworked `knn4qa` package.  `FlexNeuART` is under active development. Detailed description and documentaion is to appear. Description of the `knn4qa` is given below.

## MSMARCO Document ranking task (leaderboard submissions)

Methods and models used in [MSMARCO Document ranking task](https://microsoft.github.io/msmarco/#docranking):

1. ``August 20th, 2020	MRR=0.256``  : a multi-field BM25 scores combined with IBM Model 1 scores. This is purely non-neural submission.
2. ``August 27th, 2020	MRR=0.368``  : the output of the traditional pipeline (with MRR=0.256) is re-ranked using BERT **BASE**.
3. ``October 1st, 2020  MRR=0.38``   : the output of the traditional pipeline is re-ranked using a variant of BERT **BASE** (this model code is not available yet). Data is augmented with [doc2query text](https://github.com/castorini/docTTTTTquery). However, it does not make a whole a lot of difference in this case compared to using just the traditional pipeline.
4. ``October 13,  2020, MRR=0.39``    : a slightly better tuned and trained variant of our October 1st solution.

## Former life (as a knn4qa package)

This is a learning-to-rank pipeline, which is a part of the project where we study applicability of k-nearest neighbor search methods to IR and QA applications. This project is supported primarily by the NSF grant **#1618159** : "[Matching and Ranking via Proximity Graphs: Applications to Question Answering and Beyond](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1618159&HistoricalAwards=false)". 

Previous work can be found in the following branches branches:

1. [The ``bigger_reruns`` branch includes software](https://github.com/oaqa/knn4qa/tree/bigger_reruns) used in the dissertation of Leonid Boytsov: ["Efficient and Accurate Non-Metric k-NN Search with Applications to Text Matching"](http://boytsov.info/pubs/thesis_boytsov.pdf). A summary of this work is given in the [following blog post.](http://searchivarius.org/blog/efficient-and-accurate-non-metric-k-nn-search-applications-text-matching-we-need-more-k-nn).
2. [The ``cikm2016`` branch](https://github.com/oaqa/knn4qa/tree/cikm2016) includes software for the paper [L. Boytsov, D. Novak, Y. Malkov, E. Nyberg  (2016). *Off the Beaten Path: Let’s Replace Term-Based Retrieval
with k-NN Search*, CIKM'16](http://boytsov.info/pubs/cikm2016.pdf). This work is covered [in a blog post as well.](http://searchivarius.org/blog/text-retrieval-can-and-should-benefit-using-generic-k-nn-search-algorithms) **For more details on this branch software, please, check [the Wiki page](https://github.com/oaqa/knn4qa/wiki)**.

NB: Our [``cikm2016`` branch](https://github.com/oaqa/knn4qa/tree/cikm2016) can be also used to **partially** reproduce results from the paper: [M Surdeanu, M Ciaramita, H Zaragoza. *Learning to rank answers to non-factoid questions from web collections* 
Computational Linguistics, 2011 ](http://www.mitpressjournals.org/doi/pdfplus/10.1162/COLI_a_00051) 


