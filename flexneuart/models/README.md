## FlexNeuART neural-model ZOO

Model parameters can be specified in a JSON configuration file, which is specified during training. 
JSON parameter names should match constructor parameters of a respective models.

1. [Vanilla BERT (FirstP) ranker](cedr/cedr_vanilla_bert.py).
   This is a CEDR variant of FirstP ranker, which truncates input and pads queries. The truncation length and the backbone (flavor of BERT) are all configurable so it can be used with models such as [Longformer](https://huggingface.co/allenai/longformer-base-4096) and [Deberta V2/V3](https://huggingface.co/microsoft/deberta-v3-base).

    Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT." arXiv:1901.04085 (2019).
    
2. [Dot-product models from the Sentence Transformers library](biencoder/sbert.py)
   One can use any "dot-product" model from the [Sentence Transformers library](https://www.sbert.net/) out of the box and fine-tune it on their  data. The main ciation for this library is provided shortly, but make sure you also cite the model-specific citation if there is one: Reimers, Nils, et al. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." 


3. [COLBERT (v2) Re-ranking model](https://github.com/stanford-futuredata/ColBERT) A COLBERT (v2) model that can be used as an efficient re-ranker. It also has a strong zero-shot performance.


Various chunk-and-aggregate models for ranking of long documents, including:

1. [PARADE models, original and improved](parade).
    
    Li, C., Yates, A., MacAvaney, S., He, B., & Sun, Y. (2020). PARADE:
    Passage representation aggregation for document reranking.
    arXiv:2008.09093.
    
    Boytsov, L., Lin, T., Gao, F., Zhao, Y., Huang, J., & Nyberg, E. (2022). 
    [Understanding Performance of Long-Document Ranking Models through Comprehensive Evaluation and Leaderboarding](https://arxiv.org/abs/2207.01262). 
    arXiv:2207.01262.
    
   
5. [CEDR models](cedr).

    MacAvaney, Sean, et al. "CEDR: Contextualized embeddings for document ranking." SIGIR 2019.

4. [MaxP & SumP models](bert_aggreg_p.py).

    Dai, Zhuyun, and Jamie Callan. "Deeper text understanding for IR with contextual neural language modeling." SIGIR. 2019.

