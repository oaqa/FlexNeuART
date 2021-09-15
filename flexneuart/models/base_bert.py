#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
from flexneuart.models.base import BaseModel
from flexneuart.models.utils import init_model, BERT_ATTR

USE_BATCH_COEFF = True
DEFAULT_BERT_DROPOUT = 0.1


class BertBaseRanker(BaseModel):
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
        init_model(self, bert_flavor)

    def bert_param_names(self):
        """
        :return: a list of the main BERT-parameters. Because we assigned the main BERT model
                 to an attribute with the name BERT_ATTR, all parameter keys must start with this
                 value followed by a dot.
        """
        return set([k for k in self.state_dict().keys() if k.startswith( f'{BERT_ATTR}.')])

    def tokenize_and_encode(self, text):
        """Tokenizes the text and converts tokens to respective IDs

        :param text:  input text
        :return:      an array of token IDs
        """
        toks = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(toks)

    def forward(self, **inputs):
        raise NotImplementedError


