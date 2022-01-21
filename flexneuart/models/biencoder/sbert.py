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
    A convenience wrapper for sentence-bert bi-encoder models.
    https://github.com/UKPLab/sentence-transformers
"""
import torch

from flexneuart import models
from flexneuart.models.base import BaseModel
from typing import List

from sentence_transformers import SentenceTransformer

@models.register('biencoder_sbert')
class BiEncoderSBERT(BaseModel):
    """A sentence-bert bi-encoder wrapper class"""
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def bert_param_names(self):
        # This returns more parameters than necessary, which means that the linear projection
        # layer will be updated using a small learning rate only. But, unfortunately, it is
        # hard to extract non-BERT specific parameters in a generic way. Plus, we would
        # really want to only fine-tune this model rather than train it from scratch.
        # Anyways, training from scratch would require support for in-batch negatives,
        # which we do not provide.
        return self.model.parameters()

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
        batch = {}

        assert batch

        query_texts_trunc = [s[0:max_query_len] for s in query_texts]
        doc_texts_trunc = [s[0:max_doc_len] for s in doc_texts]

        return (self.model.encode(query_texts_trunc), self.model.encode(doc_texts_trunc))

    def forward(self, query_embed, doc_embed):
        assert query_embed.shape == doc_embed.shape # B x D

        assert len(query_embed.shape) == 2

        return torch.sum(doc_embed * query_embed, dim=-1)
