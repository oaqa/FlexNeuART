This directory has notebooks covering all essential functionality of FlexNeuART. The notebooks should be explored in the following order:

1. [Installation](installation.ipynb)
2. [Basic data preparation and processing](basic_data_preparation_and_processing.ipynb)
3. [Basic indexing (inverted and forward indices)](basic_indexing.ipynb)
4. [Training IBM Model 1 (non-neural lexical translation model)](train_model1.ipynb)
5. [Training a neural model: vanilla BERT ranker](train_neural_model.ipynb)
6. Experimentaion:

    i. [Tuning BM25 and IBM Model 1](experimentation_tuning_bm25_and_bm25_model1.ipynb)
    
    ii. [Testing models that do not require training a fusion model (no learning to rank)](experimentation_testing_no_need_to_train_fusion.ipynb)
    
    iii. [Training & testing fusion models](experimentation_train_fusion_models.ipynb)
7. [Python API](py_api_demo.ipynb)
8. [Using k-NN search on dense and dense-sparse embeddings](cand_generator_nmslib.ipynb)
9. [Data processing: Special cases (need to cover answer-based QRELs and bitext from QRELs)](special_data_processing.ipynb)

