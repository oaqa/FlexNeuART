#
# Slightly modified code from  pyserini/encode/_tct_colbert.py, pyserini/encode/_base.py
# ONNX support removed
#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import BertModel, BertTokenizer, BertTokenizerFast

class DocumentEncoder:
    def encode(self, texts, **kwargs):
        pass

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class QueryEncoder:
    def encode(self, text, **kwargs):
        pass


class TctColBertDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name: str, device, amp=False, tokenizer_name=None):
        self.device = device

        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name or model_name)
        self.amp = amp

    def encode(self, texts, max_length = 512, titles=None, **kwargs):
        if titles is not None:
            texts = [f'[CLS] [D] {title} {text}' for title, text in zip(titles, texts)]
        else:
            texts = ['[CLS] [D] ' + text for text in texts]
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding="longest",
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        
        inputs.to(self.device)
        if self.amp:
            with autocast():
                with torch.no_grad():
                    outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        embeddings = self._mean_pooling(outputs["last_hidden_state"][:, 4:, :], inputs['attention_mask'][:, 4:])

        return embeddings.detach().cpu().numpy()


class TctColBertQueryEncoder(QueryEncoder):
    def __init__(self, model_name: str, device, tokenizer_name: str = None):
        self.device = device
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name or model_name)

    def encode(self, query: str, **kwargs):
        max_length = 36  # hardcode for now
        inputs = self.tokenizer(
            '[CLS] [Q] ' + query + '[MASK]' * max_length,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        inputs.to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        return np.average(embeddings[:, 4:, :], axis=-2).flatten()
