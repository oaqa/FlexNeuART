#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#

import flexneuart.models.base as base
import flexneuart.models.utils as modeling_util

USE_BATCH_COEFF = True
DEFAULT_BERT_DROPOUT = 0.1


class BertBaseRanker(base.BaseModel):
    """
       The base class for all Transformer-based ranking models.

       We generally/broadly consider these models to be BERT-variants, hence, the name of the base class.
    """

    def __init__(self, bert_flavor):
        """Bert ranker constructor.

            :param bert_flavor:   the name of the underlying Transformer/BERT. Various
                                  Transformer models are possible as long as they return
                                  the object BaseModelOutputWithPoolingAndCrossAttentions.

        """
        super().__init__()
        modeling_util.init_model(self, bert_flavor)

    def tokenize_and_encode(self, text):
        """Tokenizes the text and converts tokens to respective IDs

        :param text:  input text
        :return:      an array of token IDs
        """
        toks = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(toks)

    def forward(self, **inputs):
        raise NotImplementedError


