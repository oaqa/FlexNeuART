#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Using some bits from CEDR (for padding and masking):
#  https://github.com/Georgetown-IR-Lab/cedr
#  which has MIT, i.e., Apache 2 compatible license.
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
from typing import List

import torch

from flexneuart.models.base import BaseModel
from flexneuart.models.utils import init_t5, T5_ATTR

USE_BATCH_COEFF = True
DEFAULT_T5_DROPOUT = 0.1
PAD_CODE = 0


class T5BaseRanker(BaseModel):
    """
       The base class for all Transformer-based ranking models.

       We generally/broadly consider these models to be T5-variants, hence, the name of the base class.
    """

    def __init__(self, t5_flavor):
        """Bart ranker constructor.

            :param t5_flavor:   The name of the underlying Transformer/T5 or a path
                                  to a previously stored model. This will will be passed
                                  to AutoModel.from_pretrained().
                                  One can use quite a few Transformer models as long as
                                  they return an object of the type:
                                  BaseModelOutputWithPoolingAndCrossAttentions.
        """
        super().__init__()
        init_t5(self, t5_flavor)

    # Keep this function name for compatibility with the old code
    def bert_param_names(self):
        """
        :return: a list of the main T5-parameters. Because we assigned the main T5 model
                 to an attribute with the name T5_ATTR, all parameter keys must start with this
                 value followed by a dot.
        """
        return set([k for k in self.state_dict().keys() if k.startswith( f'{T5_ATTR}.')])

    def featurize(self, max_query_len : int, max_doc_len : int,
                        query_texts : List[str],
                        doc_texts : List[str]) -> tuple:
        """
           "Featurizes" input. Convert input queries and texts to a set of features,
            which are compatible to the model's forward function.

            **ATTENTION!!!** This function *MUST* itself create a batch
            b/c training code does not use a standard PyTorch loader!
        """
        if self.config.model_type == 't5':
            query_tok_ids = [self._tokenize_and_encode("Query:"+ query) for query in query_texts]
            doc_tok_ids = [self._tokenize_and_encode("Document:" + doc) for doc in doc_texts]
        elif self.config.model_type == 'longt5':
            query_tok_ids = [self._tokenize_and_encode(query) for query in query_texts]
            doc_tok_ids = [self._tokenize_and_encode(doc) for doc in doc_texts]
        else:
            raise Exception(f'Unknown model type: {self.t5.config.model_type}')

        query_tok = self._pad_crop(query_tok_ids, max_len=max_query_len)
        query_mask = self._mask(query_tok_ids, max_len=max_query_len)


        doc_tok = self._pad_crop(doc_tok_ids, max_len=max_doc_len)
        doc_mask = self._mask(doc_tok_ids, max_len=max_doc_len)

        return (query_tok, query_mask, doc_tok, doc_mask)

    def forward(self, **inputs):
        raise NotImplementedError

    def _tokenize_and_encode(self, text):
        """Tokenizes the text and converts tokens to respective IDs

        :param text:  input text
        :return:      an array of token IDs
        """
        toks = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(toks)

    @staticmethod
    def _pad_crop(items, max_len, pad_code=PAD_CODE):
        result = []
        for item in items:
            if len(item) < max_len:
                item = item + [pad_code] * (max_len - len(item))
            if len(item) > max_len:
                item = item[:max_len]
            result.append(item)

        return torch.tensor(result).long()

    @staticmethod
    def _mask(items, max_len):
        result = []
        for e in items:
            elen = min(len(e), max_len)
            result.append([1.] * elen + [0.] * (max_len - elen))

        return torch.tensor(result).float()




