from typing import Optional
import torch
import torch.nn as nn
import math

from flexneuart.config import BART_BASE_MODEL, MSMARCO_MINILM_L2
from flexneuart.models import utils as modeling_util
from flexneuart.models.utils import init_model, AGGREG_ATTR
from flexneuart.models.base_bart import BartBaseRanker

from flexneuart.models import register

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bart.modeling_bart import BartClassificationHead


DEFAULT_WINDOW_SIZE = 150
DEFAULT_STRIDE = 100
DEFAULT_BART_DROPOUT = 0.1
DEFAULT_OUTPUT_HIDDEN_STATE = True
DEFAULT_OUTPUT_ATTENTIONS = True
RAND_SPECIAL_INIT_DEFAULT=True
DEFAULT_USE_SEP=True


class BartMoresModule(BartBaseRanker):
    def __init__(self, bart_flavor, window_size,
                 stride, output_attentions,
                 output_hidden_states):
        """Constructor.
        :param bart_flavor:     the name of the underlying Transformer/BART.
        :param window_size:     a size of the window (in # of tokens)
        :param stride:          a step
        :param output_attentions:   whether to output attentions

        """
        super().__init__(bart_flavor)
        self.window_size = window_size
        self.stride = stride
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

    def _prepare_tokens(self, text_tok, text_mask):
        """ This function applies BART to a query or document text.
        If this concatenation is too long to fit into a specified window, the
        text is split into (possibly overlapping) chunks.

        :param text_tok:            batched and encoded tokens (queries or text)
        :param text_mask:           token mask (0 for padding, 1 for actual tokens)

        :return: representations of each token in the last layer of encoded chunks,
        mask of the same size as the representation, and the number of subbatches.
        """

        split_toks, subbatch_count_toks = modeling_util.sliding_window_subbatch(text_tok,
                                                                                self.window_size,
                                                                                self.stride)
        split_masks, subbatch_count_masks = modeling_util.sliding_window_subbatch(text_mask,
                                                                                  self.window_size,
                                                                                  self.stride)

        assert subbatch_count_toks == subbatch_count_masks
        CLSS = torch.full_like(split_toks[:, :1], self.CLS_TOK_ID)
        EOSS = torch.full_like(split_toks[:, :1], self.EOS_TOK_ID)
        ONES = torch.ones_like(split_masks[:, :1])

        # build BART input sequences
        toks = torch.cat([CLSS, split_toks, EOSS], dim=1)
        mask = torch.cat([ONES, split_masks, ONES], dim=1)
        return toks, mask, subbatch_count_toks


@register('mores_plus')
class BartMoresRanker(BartMoresModule):
    def __init__(self, bart_flavor=BART_BASE_MODEL,
                 window_size=DEFAULT_WINDOW_SIZE,
                 stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BART_DROPOUT,
                 output_attentions=DEFAULT_OUTPUT_ATTENTIONS,
                 output_hidden_states=DEFAULT_OUTPUT_HIDDEN_STATE):

        """Constructor

        :param bart_flavor:     the name of the underlying Transformer/BART.
        :param window_size:     a size of the window (in # of tokens)
        :param stride:          a step
        :param dropout:         dropout probability
        :param output_attentions:   whether to output attentions
        :param output_hidden_states:    whether to output hidden states
        """

        super().__init__(bart_flavor=bart_flavor,
                 window_size=window_size,
                 stride=stride,
                 output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states)

        self.classification_head = BartClassificationHead(input_dim=self.BART_SIZE, 
                                                          inner_dim=self.BART_SIZE,
                                                          num_classes=1,
                                                          pooler_dropout=dropout,)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        query_toks, query_masks, _ = self._prepare_tokens(query_tok, query_mask)
        doc_toks, doc_masks, _ = self._prepare_tokens(doc_tok, doc_mask)

        # Here we assume the inputs are already chunked
        encoder_outputs = self.bart.encoder(
            input_ids=doc_toks,
            attention_mask=doc_masks,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        hiddens = encoder_outputs.last_hidden_state
        encoder_mask = doc_masks

        B, _ = query_toks.shape
        hiddens = hiddens.reshape((B, -1, hiddens.shape[-1]))
        encoder_mask = encoder_mask.reshape((B, -1))

        # Check for batch size consistency
        assert hiddens.size(0) == B
        assert encoder_mask.size(0) == B

        decoder_outputs = self.bart.decoder(
            input_ids=query_toks,
            attention_mask=query_masks,
            encoder_hidden_states=hiddens,
            encoder_attention_mask=encoder_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        last_hidden_state = decoder_outputs.last_hidden_state

        eos_mask = torch.where(query_toks == self.config.eos_token_id, 1, 0).to(last_hidden_state.dtype)
        sentence_representation = (eos_mask.view(eos_mask.shape + (1,)) * last_hidden_state).sum(1)
        logits = self.classification_head(sentence_representation)
        
        return logits.squeeze(dim=-1)

@register('mores_plus_transf')
class BartMoresTransfRanker(BartMoresModule):
    def __init__(self, bart_flavor=BART_BASE_MODEL,
                 bert_aggreg_flavor=MSMARCO_MINILM_L2,
                 window_size=DEFAULT_WINDOW_SIZE,
                 stride=DEFAULT_STRIDE,
                 rand_special_init=RAND_SPECIAL_INIT_DEFAULT,
                 dropout=DEFAULT_BART_DROPOUT,
                 output_attentions=DEFAULT_OUTPUT_ATTENTIONS,
                 output_hidden_states=DEFAULT_OUTPUT_HIDDEN_STATE,
                 use_sep=DEFAULT_USE_SEP):

        """Constructor
        :param bart_flavor:     the name of the underlying Transformer/BART.
        :param bert_aggreg_flavor:  the name of the underlying Transformer/BERT used for aggregation.
        :param window_size:     a size of the window (in # of tokens)
        :param stride:          a step
        :param rand_special_init:   true to initialize aggregator CLS AND SEP tokens randomly
        :param dropout:         dropout probability
        :param output_attentions:   whether to output attentions
        :param output_hidden_states:    whether to output hidden states
        :param use_sep:         whether to use SEP tokens for aggregation
        """

        super().__init__(bart_flavor=bart_flavor,
                 window_size=window_size,
                 stride=stride,
                 output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states)

        self.use_sep = use_sep

        # Let's create an aggregator BERT
        #init_aggreg = Empty()
        init_model(self, bert_aggreg_flavor, is_aggreg=True)
        # Must memorize this as a class attribute
        #self.bert_aggreg = init_aggreg.bert

        self.BERT_AGGREG_SIZE = self.bert_aggreg.config.hidden_size


        if not rand_special_init and hasattr(self.bert_aggreg, 'embeddings'):
            print(f'Initializing special token CLS using pre-trained embeddings of {bert_aggreg_flavor}')

            embeds = self.bert_aggreg.embeddings.word_embeddings.weight.data
            self.bert_aggreg_cls_embed = torch.nn.Parameter(embeds[self.aggreg_tokenizer.cls_token_id].clone())
            self.bert_aggreg_sep_embed = torch.nn.Parameter(embeds[self.aggreg_tokenizer.sep_token_id].clone())
        else:
            print(f'Initializing special token CLS randomly')

            norm = 1.0 / math.sqrt(self.BERT_AGGREG_SIZE)
            self.bert_aggreg_cls_embed = torch.nn.Parameter(norm * torch.randn(self.BERT_AGGREG_SIZE))
            self.bert_aggreg_sep_embed = torch.nn.Parameter(norm * torch.randn(self.BERT_AGGREG_SIZE))

        if self.BART_SIZE != self.BERT_AGGREG_SIZE:
            print('Projecting embeddings before aggregation')
            self.proj_out = torch.nn.Linear(self.BART_SIZE, self.BERT_AGGREG_SIZE)
            torch.nn.init.xavier_uniform_(self.proj_out.weight)
        else:
            self.proj_out = None

        self.cls = torch.nn.Linear(self.BERT_AGGREG_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)
        self.dropout = nn.Dropout(dropout)

    def aggreg_param_names(self):
        """
        :return: a list of the aggregate BERT-parameters. Because we assigned the aggregate model
                 to an attribute with the name AGGREG_ATTR, all parameter keys must start with this
                 value followed by a dot.
        """
        return set([k for k in self.state_dict().keys() if k.startswith( f'{AGGREG_ATTR}.')])


    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        batch_qty, _ = query_tok.shape
        query_toks, query_masks, _ = self._prepare_tokens(query_tok, query_mask)
        doc_toks, doc_masks, sbcount = self._prepare_tokens(doc_tok, doc_mask)
        query_toks = torch.cat([query_toks] * sbcount, dim=0)
        query_masks = torch.cat([query_masks] * sbcount, dim=0)

        encoder_outputs = self.bart.encoder(
            input_ids=doc_toks,
            attention_mask=doc_masks,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        hiddens = encoder_outputs.last_hidden_state
        encoder_mask = doc_masks

        # Check for batch size consistency
        B, _ = query_toks.shape
        assert hiddens.size(0) == B
        assert encoder_mask.size(0) == B

        decoder_outputs = self.bart.decoder(
            input_ids=query_toks,
            attention_mask=query_masks,
            encoder_hidden_states=hiddens,
            encoder_attention_mask=encoder_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        cls_output = decoder_outputs.last_hidden_state[:, 0, :]

        cls_result = []
        for i in range(cls_output.shape[0] // batch_qty):
            cls_result.append(cls_output[i*batch_qty:(i+1)*batch_qty])

        cls_result = torch.stack(cls_result, dim=2)


        last_layer_cls_rep = torch.transpose(cls_result, 1, 2)  #  [B, EXPECT_TOT_INTERACT_N, BERT_INTERACT_SIZE]

        B, N, _ = last_layer_cls_rep.shape

        if self.proj_out is not None:
            last_layer_cls_rep_proj = self.proj_out(last_layer_cls_rep) # [B, N, BERT_AGGREG_SIZE]
        else:
            last_layer_cls_rep_proj = last_layer_cls_rep

        # +two singletown dimensions before the CLS embedding
        aggreg_cls_tok_exp = self.bert_aggreg_cls_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B, 1,
                                                                                                 self.BERT_AGGREG_SIZE)
        ONES = torch.ones_like(query_mask[:, :1]) # Bx1
        NILS = torch.zeros_like(query_mask[:, :1]) # Bx1

        assert ONES.shape == (B, 1)
        assert NILS.shape == (B, 1)

        mask = torch.ones_like(last_layer_cls_rep_proj[..., 0])

        if self.use_sep:
            aggreg_sep_tok_exp = self.bert_aggreg_sep_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B,1,
                                                                                                     self.BERT_AGGREG_SIZE)
            EXPECT_TOT_N = N + 3

            # We need to prepend a CLS token vector as the classifier operation depends on the existence of such special token!
            last_layer_cls_rep_proj = torch.cat([aggreg_cls_tok_exp, aggreg_sep_tok_exp,
                                             last_layer_cls_rep_proj, aggreg_sep_tok_exp], dim=1) #[B, N+3, BERT_AGGREG_SIZE]

            mask = torch.cat([ONES, ONES, mask, ONES], dim=1)
            segment_ids = torch.cat([NILS] * 2 + [ONES] * (N) + [NILS], dim=1)

        else:
            # We need to prepend a CLS token vector as the classifier operation depends on the existence of such special token!
            last_layer_cls_rep_proj = torch.cat([aggreg_cls_tok_exp,
                                             last_layer_cls_rep_proj], dim=1) #[B, N+1, BERT_AGGREG_SIZE]

            EXPECT_TOT_N = N + 1

            mask = torch.cat([ONES, mask], dim=1)
            segment_ids = torch.cat([NILS] + [ONES] * (N), dim=1)

        assert last_layer_cls_rep_proj.shape == (B, EXPECT_TOT_N, self.BERT_AGGREG_SIZE)
        assert mask.shape == (B, EXPECT_TOT_N)
        assert segment_ids.shape == (B, EXPECT_TOT_N)

        # run aggregating BERT and get the last layer output
        # note that we pass directly vectors (CLS vector including!) without carrying out an embedding, b/c
        # it's pointless at this stage
        outputs : BaseModelOutputWithPoolingAndCrossAttentions = self.bert_aggreg(inputs_embeds=last_layer_cls_rep_proj,
                                                                                  token_type_ids=segment_ids.long(),
                                                                                  attention_mask=mask)
        result = outputs.last_hidden_state

        # The cls vector of the last Transformer output layer
        parade_cls_reps = result[:, 0, :] #

        out = self.cls(self.dropout(parade_cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)
