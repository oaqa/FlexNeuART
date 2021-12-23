#
# This code is based on the following repo:
# https://github.com/bmitra-msft/TREC-Deep-Learning-Quick-Start/
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import fasttext
import torch
import csv

from typing import List, Sequence

from flexneuart import models
from flexneuart.io import open_with_default_enc
from flexneuart.utils import DictToObject, clear_line_console
from flexneuart.models.base import BaseModel
from flexneuart.models.ndrm.models_main import NDRM1, NDRM2, NDRM3

NDRM_MODEL_DICT = {
    'ndrm1' : NDRM1,
    'ndrm2' : NDRM2,
    'ndrm3' : NDRM3
}

@models.register('ndrm')
class NDRMWrapperBase(BaseModel):
    """An NDRM wrapper class"""

    @staticmethod
    def pre_init(model_param_dict):
        """This function will load actual embeddings and IDF, which can be passed to the constructor of this
           model. This would allow a model holder to serialize and deserialize it properly by
           storing embeddings and IDFs jointly with model weights.

        :param model_param_dict: an original model parameter dictionary
        :return: a dictionary with modified parameters
        """
        model_param_dict['embeddings'] = NDRMWrapperBase.load_pretrained_embeddings(model_param_dict['embeddings'])
        model_param_dict['idfs'] = NDRMWrapperBase.load_idfs(model_param_dict['idfs'])

        return model_param_dict

    def featurize(self, max_query_len : int, max_doc_len : int,
                        query_texts : List[str],
                        doc_texts : List[str]) -> tuple:

        """
           "Featurizes" input. Convert input queries and texts to a set of features,
            which are compatible to the model's forward function.

            **ATTENTION!!!** This function *MUST* itself create a batch
            b/c training code does not use a standard PyTorch loader!
        """

        query_qty = len(query_texts)
        batch = {}
        curr_i = 0
        prev_i = 0
        query_doc_qtys = [] # number of documents generated per query

        assert query_qty > 0

        while curr_i <= query_qty:
            if curr_i == query_qty or query_texts[curr_i] != query_texts[prev_i]:
                query_doc_qtys.append(curr_i - prev_i)
                single_query_feat = self.__featurize(q=query_texts[prev_i],
                                                ds_orig=doc_texts,
                                                max_query_len=max_query_len,
                                                max_doc_len=max_doc_len,
                                                infer_mode=not self.training)
                feat_qty = len(single_query_feat)

                if prev_i == 0:
                    batch = {k : [] for k in range(feat_qty)}
                else:
                    # this is paranoid and it shouldn't happen, b/c the # of features
                    # depends only on the model type and this shouldn't change after
                    # the model is constructed
                    assert feat_qty == len(batch), "bug: inconsistent # of features!"

                for k in range(feat_qty):
                    batch[k].append(single_query_feat[k].unsqueeze(dim=0))

                prev_i = curr_i

            curr_i += 1


        assert batch
        #
        # at inference time, b/c the query length basically varies, there would be
        # incompatibilities among features produced by diff. queries.
        # hence mixed-query input is prohibited @ inference time
        if not self.training:
            assert len(query_doc_qtys) == 1, "Mixed-query input is prohibited during inference time!"

        features = [torch.LongTensor(query_doc_qtys)]
        # Concatenate all features to make the batch
        for i in range(feat_qty):
            assert len(batch[i]) == len(query_doc_qtys)
            features.append(torch.cat(batch[i]))

        return tuple(features)


    def __init__(self,
                 model_type,
                 embeddings,
                 idfs,
                 max_terms_doc=4000,
                 num_hidden_nodes=256,
                 num_encoder_layers=2,
                 conv_window_size=31,
                 num_attn_heads=32,
                 rbf_kernel_dim=10,
                 rbf_kernel_pool_size=300,
                 rbf_kernel_pool_stride=100,
                 dropout=0.2,
                 no_conformer=False):
        """

        :param model_type:              a type of the wrapped model
        :param embedding_data:          a (vocab, embedding matrix, embedding size) triple
        :param max_terms_doc:           maximum number of terms to consider for long text
        :param idfs:                    IDF dictionary
        :param num_hidden_nodes:        size of hidden layers
        :param num_encoder_layers:      number of document encoder layers
        :param conv_window_size:        window size for encoder convolution layer
        :param num_attn_heads:          number of self-attention heads
        :param rbf_kernel_dim:          number of RBF kernels
        :param rbf_kernel_pool_size:    window size for pooling layer in RBF kernels
        :param rbf_kernel_pool_stride:  stride for pooling layer in RBF kernels
        :param dropout:                 dropout rate
        :param no_conformer:            true to disable conformer
        :param qti_mode:                true for query-term independence (QTI_ model
        """
        super().__init__()

        self.qti_mode = False # This must be false, we cannot deal with query-specific scores returned by the model
        self.vocab, pretrained_embeddings, embed_size = embeddings
        self.idfs = idfs

        model_type = model_type.lower()

        arg_dict = {
            'model'            : model_type,
            'pretrained_embeddings' : pretrained_embeddings,
            'max_terms_doc'    : max_terms_doc,
            'vocab_size'       : pretrained_embeddings.size()[0],
            'num_hidden_nodes' : num_hidden_nodes,
            'num_encoder_layers' : num_encoder_layers,
            'conv_window_size' : conv_window_size,
            'num_attn_heads' : num_attn_heads,
            'rbf_kernel_dim' : rbf_kernel_dim,
            'rbf_kernel_pool_size' : rbf_kernel_pool_size,
            'rbf_kernel_pool_stride' : rbf_kernel_pool_stride,
            'dropout' : dropout,
            'no_conformer' : no_conformer
        }
        self.args = DictToObject(arg_dict)
        assert self.args.num_hidden_nodes == embed_size, \
            'pretrained embedding size ({}) does not match specified embedding size ({})'.format(embed_size,
                                                                                                 self.args.num_hidden_nodes)

        if not model_type in NDRM_MODEL_DICT:
            raise Exception(f'Unsupported NDRM model type: {model_type}, supported models: ' +
                            ', '.join(NDRM_MODEL_DICT.keys()))

        self.model = NDRM_MODEL_DICT[model_type](self.args)

    #
    # Admittedly this a rather hacky way to route input to different types of models, which
    # the current wrapper can hold. However, this seems to be the easiest way to incorporate
    # the original NDRM code without changing its core parts. Specifically, but keeping
    # the "ghost" forward function as generic as possible we can reuse the original
    # implemenation of the featurize function with minimum modifications.
    #
    def forward(self, *args):
        # Note that the first Tensor in the tuple must be the number of documents per query
        y = self.model(*args[1:], qti_mode=self.qti_mode)

        #
        # the output has shape [Q x D], which gives the score of each query vs every document,
        # but we will ignore scores for unrelated query-document pairs
        #

        query_doc_qtys = args[0].tolist()

        start=0

        res_mask = torch.zeros(y.shape, dtype=torch.bool)

        for qpos, qty in enumerate(query_doc_qtys):
            res_mask[qpos, start : start+qty] = True
            start += qty
        assert start == y.shape[1]

        return y[res_mask]

    def tokenize(self, s):
        return s.split()

    @staticmethod
    def load_pretrained_embeddings(file_name):
        model = fasttext.load_model(file_name)
        embed_size = model.get_input_matrix().shape[1] * 2
        clear_line_console()

        # FastText produces two sets of embeddings and we concatenate them
        pretrained_embeddings = torch.cat([torch.FloatTensor(model.get_input_matrix()),
                                           torch.FloatTensor(model.get_output_matrix())], dim=1)

        # Prepend embeddings for special symbols
        pretrained_embeddings = torch.cat([torch.zeros([3, embed_size], dtype=torch.float32),
                                           pretrained_embeddings], dim=0)
        terms = model.get_words(include_freq=False)
        vocab = {'UNK': 0, '<S>': 1, '</S>': 2}
        for i in range(len(terms)):
            vocab[terms[i]] = i + 3

        return vocab, pretrained_embeddings, embed_size

    @staticmethod
    def load_idfs(file_name):
        idfs = {}
        with open_with_default_enc(file_name) as f:
            reader = csv.reader(f, delimiter = '\t')
            for [term, idf] in reader:
                idfs[term] = float(idf)
        return idfs

    def __get_features_lat(self, terms: List[str], max_terms : int):
        terms = terms[:max_terms]
        num_terms = len(terms)
        num_pad = max_terms - num_terms
        features = [self.vocab.get(terms[i], self.vocab['UNK']) for i in range(num_terms)] + [0]*num_pad
        masks = [1]*num_terms + [0]*num_pad
        return features, masks

    def __get_features_exp(self, q: List[str], d: List[str], max_q_terms: int):
        q = q[:max_q_terms]
        features = [d.count(term) for term in q]
        pad_len = max_q_terms - len(q)
        features.extend([0]*pad_len)
        return features

    def __get_features_dlen(self, ds):
        features = [len(d) for d in ds]
        return features

    def __get_features_idf(self, terms: List[str], max_terms: int):
        terms = terms[:max_terms]
        num_terms = len(terms)
        num_pad = max_terms - num_terms
        features = [self.idfs.get(terms[i], 0) for i in range(num_terms)] + [0]*num_pad
        return features

    #
    # This function deviates from the original one in a couple of aspects:
    # 1. removed explicit support for ORCAS fields
    # 2. max # of terms for docs and queries now is a minimum of two values
    # 3. this function now returns torch tensors instead of numpy arrays
    #
    def __featurize(self, q: str, ds_orig: List[str],
                          max_query_len: int, max_doc_len: int,
                          infer_mode: bool =False) -> Sequence[torch.Tensor]:
        max_doc_len = min(self.args.max_terms_doc, max_doc_len)
        q = self.tokenize(q)
        max_q_terms = len(q) if infer_mode else max_query_len
        ds_tok = [['<S>'] + self.tokenize(ds_orig[i]) + ['</S>']for i in range(len(ds_orig))]
        feat_q, feat_mask_q = self.__get_features_lat(q, max_q_terms)
        feat_q = torch.tensor(feat_q, dtype=torch.int64)
        feat_mask_q = torch.tensor(feat_mask_q, dtype=torch.float32)
        if self.args.model != 'ndrm2':
            features = [self.__get_features_lat(doc, max_doc_len) for doc in ds_tok]
            feat_d = [feat[0] for feat in features]
            feat_d = torch.tensor(feat_d, dtype=torch.int64)
            feat_mask_d = [feat[1] for feat in features]
            feat_mask_d = torch.tensor(feat_mask_d, dtype=torch.float32)
        if self.args.model != 'ndrm1':
            feat_qd = [self.__get_features_exp(q, doc, max_q_terms) for doc in ds_tok]
            feat_qd = torch.tensor(feat_qd, dtype=torch.float32)
            feat_idf = self.__get_features_idf(q, max_q_terms)
            feat_idf = torch.tensor(feat_idf, dtype=torch.float32)
            feat_dlen = self.__get_features_dlen(ds_tok)
            feat_dlen = torch.tensor(feat_dlen, dtype=torch.float32)
        if self.args.model == 'ndrm1':
            return feat_q, feat_d, feat_mask_q, feat_mask_d
        if self.args.model == 'ndrm2':
            return feat_qd, feat_mask_q, feat_idf, feat_dlen
        return feat_q, feat_d, feat_qd, feat_mask_q, feat_mask_d, feat_idf, feat_dlen



