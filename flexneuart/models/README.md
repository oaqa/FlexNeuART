## FlexNeuART neural-model ZOO

Model parameters can be specified in a JSON configuration file, which is specified during training. 
JSON parameter names should match constructor parameters of a respective models.

1. [Vanilla BERT (FirstP) ranker](cedr/cedr_vanilla_bert.py).
   This is a CEDR variant of FirstP ranker, which truncates input and pads queries. 

    Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT." arXiv:1901.04085 (2019).

2. [CEDR models](cedr).

    MacAvaney, Sean, et al. "CEDR: Contextualized embeddings for document ranking." SIGIR 2019.


3. [PARADE models](parade).
    
    Li, C., Yates, A., MacAvaney, S., He, B., & Sun, Y. (2020). PARADE:
    Passage representation aggregation for document reranking.
    arXiv:2008.09093.

4. [MaxP & SumP models](bert_aggreg_p.py).

    Dai, Zhuyun, and Jamie Callan. "Deeper text understanding for IR with contextual neural language modeling." SIGIR. 2019.