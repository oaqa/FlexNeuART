#!/usr/bin/env python
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
from pathlib import Path

import torch
import json

from typing import List

from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers import AutoTokenizer

from flexneuart.models import register
from flexneuart.models.base import BaseModel
from flexneuart.config import BERT_BASE_MODEL

from .bert_layers import BertModel
from .configuration_bert import BertConfig

PAD_CODE = 0
DEFAULT_BERT_DROPOUT = 0.1

BERT_ATTR = 'bert'

# Get the absolute path of the file
file_path = Path(__file__).resolve()

# Get the parent directory
parent_dir = file_path.parent

def init_mosaic(obj_ref, bert_flavor):
    # Create config object
    config = BertConfig.from_pretrained(bert_flavor)
    model = BertModel(config, add_pooling_layer=True)

    model.load_state_dict(BertModel.from_pretrained(bert_flavor).state_dict())

    obj_ref.BERT_MODEL = bert_flavor

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(BERT_BASE_MODEL)
    setattr(obj_ref, BERT_ATTR, model)
    obj_ref.config = config
    obj_ref.tokenizer = tokenizer
    obj_ref.no_token_type_ids = not 'token_type_ids' in tokenizer.model_input_names

    obj_ref.CHANNELS = config.num_hidden_layers + 1
    obj_ref.BERT_SIZE = config.hidden_size
    obj_ref.MAXLEN = config.max_position_embeddings

    obj_ref.CLS_TOK_ID = tokenizer.cls_token_id
    obj_ref.SEP_TOK_ID = tokenizer.sep_token_id

    print('Model type:', obj_ref.BERT_MODEL,
        '# of channels:', obj_ref.CHANNELS,
        'hidden layer size:', obj_ref.BERT_SIZE,
        'input window size:', obj_ref.MAXLEN,
        'no token type IDs:', obj_ref.no_token_type_ids)
    return

@register('mosaic_bert')
class MosaicBert(BaseModel):
    """
        A special wrapper for MOSAIC BERT code, e.g., mosaicml/mosaic-bert-base-seqlen-2048
    """
    def __init__(self,
                 bert_flavor="mosaicml/mosaic-bert-base-seqlen-2048",
                 dropout=DEFAULT_BERT_DROPOUT):
        """

        :param bert_flavor: a name / location of a MOSAIC BERT model, e.g., mosaicml/mosaic-bert-base-seqlen-2048
        :param dropout:
        """
        super().__init__()
        assert bert_flavor.startswith("mosaicml/mosaic-bert"), 'bert_flavor_parameter: you should use one of the mosaic BERT models'
        init_mosaic(self, bert_flavor)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def bert_param_names(self):
        """
        :return: a list of the main BERT-parameters. Because we assigned the main BERT model
                 to an attribute with the name BERT_ATTR, all parameter keys must start with this
                 value followed by a dot.
        """
        return set([k for k in self.state_dict().keys() if k.startswith( f'{BERT_ATTR}.')])

    def featurize(self, max_query_len : int, max_doc_len : int,
                        query_texts : List[str],
                        doc_texts : List[str]) -> tuple:
        """
           "Featurizes" input. This function itself create a batch
            b/c training code does not use a standard PyTorch loader!
        """
        tok : PreTrainedTokenizerBase = self.tokenizer
        assert len(query_texts) == len(doc_texts), \
            "Document array length must be the same as query array length!"
        input_list = list(zip(query_texts, doc_texts))

        # With only_second truncation, sometimes this will fail b/c the first sequence is already longer
        # than the maximum allowed length so batch_encode_plus will fail with an exception.
        # In many IR applications, a document is much longer than a string, so effectively
        # only_second truncation strategy will be used most of the time.
        res : BatchEncoding = tok.batch_encode_plus(batch_text_or_text_pairs=input_list,
                                   padding='longest',
                                   #truncation='only_second',
                                   truncation='longest_first',
                                   max_length=3 + max_query_len + max_doc_len,
                                   return_tensors='pt')

        # token_type_ids may be missing
        return (res.input_ids, getattr(res, "token_type_ids", None), res.attention_mask)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = \
            self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    
        cls_reps = outputs[0][:, 0]
        out = self.cls(self.dropout(cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)
