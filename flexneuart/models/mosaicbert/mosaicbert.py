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

from flexneuart.models.mosaicbert.bert_layers import BertModel
from flexneuart.models.mosaicbert.configuration_bert import BertConfig

PAD_CODE = 0
DEFAULT_BERT_DROPOUT = 0.1

BERT_ATTR = 'bert'

# Get the absolute path of the file
file_path = Path(__file__).resolve()

# Get the parent directory
parent_dir = file_path.parent

# Assumes the pytorch model and the config.json files are stored in the current directory
# To use the mosaic bert type e.g mosaic-bert-base-seqlen-2048, go to Huggingface and download the pytorch model and config.json
# and copy to the current directory

def init_mosaic(obj_ref):
    json_file_path = parent_dir / "config.json"
    with open(json_file_path, 'r') as file:
        config_params = json.load(file)

    # Create config object
    config = BertConfig.from_dict(config_params)
    model = BertModel(config, add_pooling_layer=False)

    # Load the model's state dictionary
    state_dict = torch.load(parent_dir / 'pytorch_model.bin')
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("bert.", "", 1)  # the "1" ensures only the first occurrence is replaced
        new_state_dict[new_key] = state_dict[key]

    # Remove keys that start with "cls"
    keys_to_delete = [key for key in new_state_dict.keys() if key.startswith("cls")]
    for key in keys_to_delete:
        del new_state_dict[key]

    # Apply the loaded state dictionary to the model
    model.load_state_dict(new_state_dict)

    obj_ref.BERT_MODEL = "mosaic_bert"

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


class MosaicBertBaseRanker(BaseModel):
    """
       The base class for all Transformer-based ranking models.

       We generally/broadly consider these models to be BERT-variants, hence, the name of the base class.
    """

    def __init__(self):
        """Mosaic Bert ranker constructor.
        """
        super().__init__()
        init_mosaic(self)
        

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
           "Featurizes" input. Convert input queries and texts to a set of features,
            which are compatible to the model's forward function.

            **ATTENTION!!!** This function *MUST* itself create a batch
            b/c training code does not use a standard PyTorch loader!
        """

        query_tok_ids = [self._tokenize_and_encode(query) for query in query_texts]
        query_tok = self._pad_crop(query_tok_ids, max_len=max_query_len)
        query_mask = self._mask(query_tok_ids, max_len=max_query_len)

        doc_tok_ids = [self._tokenize_and_encode(doc) for doc in doc_texts]
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


@register('mosaic_bert')
class MosaicBert(MosaicBertBaseRanker):
    """
        A standard vanilla BERT Ranker, which does not pad queries (unlike CEDR version of FirstP).
    """
    def __init__(self, dropout=DEFAULT_BERT_DROPOUT):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

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
