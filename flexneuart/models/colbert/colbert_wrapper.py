#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
    A wrapper for ColBERT (new_api branch), which currently is used only as a re-ranker.

    https://github.com/stanford-futuredata/ColBERT/tree/new_api/colbert
"""
import os

from typing import List
from .colbert import ColBERT
from .tokenization import DocTokenizer, QueryTokenizer

from flexneuart import models
from flexneuart.models.base import BaseModel

INNER_MODEL_ATTR = 'model'


@models.register('colbert')
class ColbertLoader(BaseModel):
    """
        A wrapper class to load externally trained COLBERT.
        It will load only a previously saved model, or a
        model created by converting a COLBERT model using our
        script: scripts/models/colbert/convert_model.py

        This model should be trained using original COLBERT software:
        https://github.com/stanford-futuredata/ColBERT/tree/new_api/colbert
    """
    def __init__(self, bert_flavor, colbert_config):
        """
        :param bert_flavor: The name of the underlying Transformer/BERT.
                            It should *NOT* be a path to save Transformer model!
        """
        super().__init__()
        assert bert_flavor is not None
        if os.path.exists(bert_flavor):
            raise Exception('bert_flavor cannot point a previously saved model, use colbert_checkpoint instead')

        model = ColBERT(bert_flavor=bert_flavor, colbert_config=colbert_config)
        setattr(self, INNER_MODEL_ATTR, model)

        self.query_tok = QueryTokenizer(bert_flavor, model.colbert_config)
        self.doc_tok = DocTokenizer(bert_flavor, model.colbert_config)

    def bert_param_names(self):
        # This returns more parameters than necessary, which means that the linear projection
        # layer will be updated using a small learning rate only. But, unfortunately, it is
        # hard to extract non-BERT specific parameters in a generic way. Plus, we would
        # really want to only fine-tune this model rather than train it from scratch.
        # Anyways, training from scratch would require support for in-batch negatives,
        # which we do not provide.
        return set([k for k in self.state_dict().keys() if k.startswith( f'{INNER_MODEL_ATTR}.')])

    def get_colbert_model(self) -> ColBERT:
        """
        :return: an underlying COLBERT model.
        """
        return getattr(self, INNER_MODEL_ATTR)

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
        assert query_qty == len(doc_texts)

        q_inp_ids, q_attn_mask = self.query_tok.tensorize(query_texts, max_query_len=max_query_len)
        d_inp_ids, d_attn_mask = self.doc_tok.tensorize(doc_texts, max_doc_len=max_doc_len)

        return q_inp_ids, q_attn_mask, d_inp_ids, d_attn_mask

    def forward(self, q_inp_ids, q_attn_mask, d_inp_ids, d_attn_mask):
        assert len(q_inp_ids) == len(d_inp_ids)
        assert len(q_inp_ids) == len(q_attn_mask)
        assert len(q_inp_ids) == len(d_attn_mask)

        model : ColBERT = self.get_colbert_model()

        q = model.query(q_inp_ids, q_attn_mask)
        d, d_mask = model.doc(d_inp_ids, d_attn_mask, keep_dims='return_mask')

        return model.score(q, d, d_mask)




