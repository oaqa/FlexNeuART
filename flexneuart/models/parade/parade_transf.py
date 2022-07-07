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
import torch
import math

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart.config import BERT_BASE_MODEL, MSMARCO_MINILM_L2
from flexneuart.models.utils import init_model
from flexneuart.models import register
from flexneuart.models.base_bert_split_slide_window import \
        BertSplitSlideWindowRanker, DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, \
        CLS_AGGREG_STACK
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT

class Empty:
    pass

RAND_SPECIAL_INIT_DEFAULT=True


class ParadeTransfPretrAggregRankerBase(BertSplitSlideWindowRanker):
    """
        PARADE Transformer base ranker. Contributed by Tianyi Lin, reworked by Leonid Boytsov.

        Child classes can use either randomly initialized or pre-trained aggregating Transformer.
        When the transformer is pre-trained, as an option we can use pre-trained representations of CLS and SEP tokens,
        but it does not seem to work better, though.

        We also implemented a modification, which feeds query representations into an aggregating transformer.

        Main Transformer paper:

        Li, C., Yates, A., MacAvaney, S., He, B., & Sun, Y. (2020). PARADE:
        Passage representation aggregation for document reranking.
        arXiv preprint arXiv:2008.09093.

        Modification with integrated query embeddings:

        Understanding Performance of Long-Document Ranking Models through Comprehensive Evaluation and Leaderboarding (2022).
        Leonid Boytsov, Tianyi Lin, Fangwei Gao, Yutian Zhao, Jeffrey Huang, Eric Nyberg.

        https://arxiv.org/abs/2207.01262

    """

    def __init__(self,
                 bert_flavor=BERT_BASE_MODEL,
                 bert_aggreg_flavor=MSMARCO_MINILM_L2,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 rand_special_init=RAND_SPECIAL_INIT_DEFAULT,
                 dropout=DEFAULT_BERT_DROPOUT):
        """Constructor.

        :param bert_flavor:             name of the main BERT model.
        :param bert_aggreg_flavor:      name of the aggregating BERT model.
        :param window_size:             the size of the aggregating window (a number of document tokens)
        :param stride:                  aggregating window stride
        :param rand_special_init:       true to initialize aggregator CLS token randomly
        :param dropout:                 dropout before CLS
        """
        super().__init__(bert_flavor, cls_aggreg_type=CLS_AGGREG_STACK,
                         window_size=window_size, stride=stride)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)

        # Let's create an aggregator BERT
        init_data = Empty()
        init_model(init_data, bert_aggreg_flavor)
        # Must memorize this as a class attribute
        self.bert_aggreg = init_data.bert

        self.BERT_AGGREG_SIZE = init_data.BERT_SIZE


        # Sometimes SEP embeddings isn't used, but there's not much harm to always init
        if not rand_special_init and hasattr(self.bert_aggreg, 'embeddings'):
            print(f'Initializing special token CLS using pre-trained embeddings of {bert_aggreg_flavor}')

            embeds = self.bert_aggreg.embeddings.word_embeddings.weight.data
            self.bert_aggreg_cls_embed = torch.nn.Parameter(embeds[self.tokenizer.cls_token_id].clone())
        else:
            print(f'Initializing special token CLS randomly')

            norm = 1.0 / math.sqrt(self.BERT_AGGREG_SIZE)
            self.bert_aggreg_cls_embed = torch.nn.Parameter(norm * torch.randn(self.BERT_AGGREG_SIZE))


        # If there's a mismatch between the embedding size of the aggregating BERT and the
        # hidden size of the main BERT, a projection is required
        if self.BERT_SIZE != self.BERT_AGGREG_SIZE:
            print('Projecting embeddings before aggregation')
            self.proj_out = torch.nn.Linear(self.BERT_SIZE, self.BERT_AGGREG_SIZE)
            torch.nn.init.xavier_uniform_(self.proj_out.weight)
        else:
            self.proj_out = None

        self.cls = torch.nn.Linear(self.BERT_AGGREG_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)


@register('parade_transf_pretr')
class ParadeTransfPretrAggregRanker(ParadeTransfPretrAggregRankerBase):
    """
        Pre-trained aggregator Transformer (PARADE paper).
    """

    def __init__(self, bert_flavor=BERT_BASE_MODEL, bert_aggreg_flavor=MSMARCO_MINILM_L2,
                 rand_special_init=RAND_SPECIAL_INIT_DEFAULT,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor=bert_flavor, bert_aggreg_flavor=bert_aggreg_flavor,
                         rand_special_init=rand_special_init,
                         window_size=window_size, stride=stride,
                         dropout=dropout)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask).cls_results
        last_layer_cls_rep = torch.transpose(cls_reps[-1], 1, 2)  # [B, N, BERT_SIZE]

        B, N, _ = last_layer_cls_rep.shape

        if self.proj_out is not None:
            last_layer_cls_rep_proj = self.proj_out(last_layer_cls_rep) # [B, N, BERT_AGGREG_SIZE]
        else:
            last_layer_cls_rep_proj = last_layer_cls_rep

        # +two singletown dimensions before the CLS embedding
        aggreg_cls_tok_exp = self.bert_aggreg_cls_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B, 1, self.BERT_AGGREG_SIZE)

        # We need to prepend a CLS token vector as the classifier operation depends on the existence of such special token!
        last_layer_cls_rep_proj = torch.cat([last_layer_cls_rep_proj, aggreg_cls_tok_exp], dim=1) #[B, N+1, BERT_AGGREG_SIZE]

        # run aggregating BERT and get the last layer output
        # note that we pass directly vectors (CLS vector including!) without carrying out an embedding, b/c
        # it's pointless at this stage
        outputs : BaseModelOutputWithPoolingAndCrossAttentions = self.bert_aggreg(inputs_embeds=last_layer_cls_rep_proj)
        result = outputs.last_hidden_state

        # The cls vector of the last Transformer output layer
        parade_cls_reps = result[:, 0, :] #

        out = self.cls(self.dropout(parade_cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)


@register('parade_transf_rand')
class ParadeTransfRandAggregRanker(BertSplitSlideWindowRanker):
    """
        Randomly intialized aggregator Transformer (PARADE paper).
    """

    def __init__(self, bert_flavor=BERT_BASE_MODEL,
                 aggreg_layer_qty=2, aggreg_head_qty=4,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor, cls_aggreg_type=CLS_AGGREG_STACK,
                         window_size=window_size, stride=stride)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)

        # Let's create an aggregator Transformer
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.BERT_SIZE, nhead=aggreg_head_qty)
        norm_layer = torch.nn.LayerNorm(self.BERT_SIZE)
        self.transf_aggreg = torch.nn.TransformerEncoder(encoder_layer, aggreg_layer_qty, norm=norm_layer)

        self.bert_aggreg_cls_embed = torch.nn.Parameter(torch.randn(self.BERT_SIZE))

        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask).cls_results
        last_layer_cls_rep = torch.transpose(cls_reps[-1], 1, 2)  # [B, N, BERT_SIZE]

        B, N, _ = last_layer_cls_rep.shape

        # +two singletown dimensions before the CLS embedding
        aggreg_cls_tok_exp = self.bert_aggreg_cls_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B, 1,
                                                                                                 self.BERT_SIZE)

        # We need to prepend a CLS token vector as the classifier operation depends on the existence of such special token!
        last_layer_cls_rep = torch.cat([last_layer_cls_rep, aggreg_cls_tok_exp], dim=1)  # [B, N+1, BERT_AGGREG_SIZE]

        # run aggregating BERT and get the last layer output
        # note that we pass directly vectors (CLS vector including!) without carrying out an embedding, b/c
        # it's pointless at this stage
        result = self.transf_aggreg(last_layer_cls_rep)

        # The cls vector of the last Transformer output layer
        parade_cls_reps = result[:, 0, :]  #

        out = self.cls(self.dropout(parade_cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)


@register('parade_transf_wquery_pretr')
class ParadeTransfWithQueryPretrAggregRanker(ParadeTransfPretrAggregRankerBase):
    """
        Pre-trained aggregator Transformer with integrated query embeddings (our modification of PARADE).
    """

    def __init__(self,
                 bert_flavor=BERT_BASE_MODEL,
                 bert_aggreg_flavor=MSMARCO_MINILM_L2,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 separate_query_proj=False,
                 rand_special_init=RAND_SPECIAL_INIT_DEFAULT,
                 dropout=DEFAULT_BERT_DROPOUT):
        """Constructor.

        :param bert_flavor:             name of the main BERT model.
        :param bert_aggreg_flavor:      name of the aggregating BERT model.
        :param window_size:             the size of the aggregating window (a number of document tokens)
        :param stride:                  aggregating window stride
        :param rand_special_init:       true to initialize aggregator CLS token randomly
        :param dropout:                 dropout before CLS
        :param separate_query_proj:     true to enable a separate projection matrix for query embeddings (passed to the
                                        aggregator Transformer).
        """
        super().__init__(bert_flavor=bert_flavor, bert_aggreg_flavor=bert_aggreg_flavor,
                         rand_special_init=rand_special_init,
                         window_size=window_size, stride=stride,
                         dropout=dropout)
        if separate_query_proj:
            print('Using a separate projection matrix for query embeddings')
            self.proj_query = torch.nn.Linear(self.BERT_SIZE, self.BERT_AGGREG_SIZE)
            torch.nn.init.xavier_uniform_(self.proj_out.weight)
        else:
            self.proj_query = None
            if self.proj_query is not None:
                print('Sharing query projection matrix with window CLS token embeddings')

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        enc_res = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        cls_reps = enc_res.cls_results
        last_layer_cls_rep = torch.transpose(cls_reps[-1], 1, 2)  # [B, N, BERT_SIZE]

        B, N, BERT_SIZE = last_layer_cls_rep.shape

        query_reps = enc_res.query_results
        last_layer_query_rep = query_reps[-1]
        assert len(last_layer_query_rep.shape) == 3 # [B, Q, BERT_SIZE]
        _, Q, _ = last_layer_query_rep.shape
        assert last_layer_query_rep.size(0) == B
        assert last_layer_query_rep.size(2) == BERT_SIZE

        # Trying to nitialize query projection matrix using the query-specific matrix
        proj_query = self.proj_query
        # However if it's None try to use the general projection matrix
        if proj_query is None:
            assert self.proj_query is None
            proj_query = self.proj_out

        if self.proj_out is not None:
            last_layer_cls_rep_proj = self.proj_out(last_layer_cls_rep)  # [B, N, BERT_AGGREG_SIZE]
        else:
            last_layer_cls_rep_proj = last_layer_cls_rep

        if proj_query is not None:
            assert (self.proj_query is not None and proj_query == self.proj_query) or \
                   (self.proj_query is     None and proj_query == self.proj_out)

            last_layer_query_rep_proj = proj_query(last_layer_query_rep)
        else:
            last_layer_query_rep_proj = last_layer_query_rep

        # +two singletown dimensions before the CLS embedding
        aggreg_cls_tok_exp = self.bert_aggreg_cls_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B, 1,
                                                                                                 self.BERT_AGGREG_SIZE)

        EXPECT_TOT_N = N + 1 + Q
        # We need to prepend a CLS token vector as the classifier operation depends on the existence of such special token!
        aggreg_repr = torch.cat([aggreg_cls_tok_exp,
                                 last_layer_query_rep_proj,
                                 last_layer_cls_rep_proj],
                                             dim=1)  # [B, N + 1 + Q, BERT_AGGREG_SIZE]
        assert aggreg_repr.shape == (B, EXPECT_TOT_N, self.BERT_AGGREG_SIZE)

        ONES = torch.ones_like(query_mask[:, :1]) # Bx1
        NILS = torch.zeros_like(query_mask[:, :1]) # Bx1

        assert ONES.shape == (B, 1)
        assert NILS.shape == (B, 1)

        doc_mask_aggreg = torch.ones_like(last_layer_cls_rep[:,:,0])
        assert doc_mask_aggreg.shape == (B, N), f'Internal error, expected shape ({B},{N}) actual shape: {doc_mask_aggreg.shape} last_layer_cls_rep.shape: {last_layer_cls_rep.shape}'

        mask = torch.cat([ONES, query_mask, doc_mask_aggreg], dim=1)
        assert mask.shape == (B, EXPECT_TOT_N)

        segment_ids = torch.cat([NILS] * (1 + Q) + [ONES] * (N), dim=1)

        assert segment_ids.shape == (B, EXPECT_TOT_N)

        # run aggregating BERT and get the last layer output
        # note that we pass directly vectors (CLS vector including!) without carrying out an embedding, b/c
        # it's pointless at this stage
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_aggreg(inputs_embeds=aggreg_repr,
                                                                                 token_type_ids=segment_ids.long(),
                                                                                 attention_mask=mask,)
        result = outputs.last_hidden_state

        # The cls vector of the last Transformer output layer
        parade_cls_reps = result[:, 0, :]  #

        out = self.cls(self.dropout(parade_cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)
