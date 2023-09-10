import torch
import math

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart.config import BERT_BASE_MODEL, MSMARCO_MINILM_L2
from flexneuart.models.utils import init_model, AGGREG_ATTR, INTERACT_ATTR
from flexneuart.models import register

from flexneuart.models.base_bert_late_interaction import \
        BertLateInteraction, DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE

from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT

class Empty:
    pass

RAND_SPECIAL_INIT_DEFAULT=True

@register('parade_lateir_transf')
class ParadeLateirTransfAggregRanker(BertLateInteraction):
    def __init__(self, bert_flavor=BERT_BASE_MODEL,
                 bert_interact_flavor=MSMARCO_MINILM_L2,
                 bert_aggreg_flavor=MSMARCO_MINILM_L2,
                 window_size=DEFAULT_WINDOW_SIZE, 
                 stride=DEFAULT_STRIDE,
                 rand_special_init=RAND_SPECIAL_INIT_DEFAULT,
                 dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor=bert_flavor, 
                         window_size=window_size, stride=stride)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)

        # Let's create an interaction BERT
        #init_interact = Empty()
        init_model(self, bert_interact_flavor, is_interact=True)
        # Must memorize this as a class attribute
        #self.bert_interact = init_interact.bert

        self.BERT_INTERACT_SIZE = self.bert_interact.config.hidden_size
        
        # Let's create an aggregator BERT
        #init_data = Empty()
        init_model(self, bert_aggreg_flavor, is_aggreg=True)
        # Must memorize this as a class attribute
        #self.bert_aggreg = init_data.bert_aggreg

        self.BERT_AGGREG_SIZE = self.bert_aggreg.config.hidden_size

        if not rand_special_init and hasattr(self.bert_interact, 'embeddings'):
            print(f'Initializing special token CLS using pre-trained embeddings of {bert_interact_flavor}')

            embeds = self.bert_interact.embeddings.word_embeddings.weight.data
            self.bert_interact_cls_embed = torch.nn.Parameter(embeds[self.interact_tokenizer.cls_token_id].clone())
            self.bert_interact_sep_embed = torch.nn.Parameter(embeds[self.interact_tokenizer.sep_token_id].clone())
        else:
            print(f'Initializing special token CLS randomly')

            norm = 1.0 / math.sqrt(self.BERT_INTERACT_SIZE)
            self.bert_interact_cls_embed = torch.nn.Parameter(norm * torch.randn(self.BERT_INTERACT_SIZE))
            self.bert_interact_sep_embed = torch.nn.Parameter(norm * torch.randn(self.BERT_INTERACT_SIZE))

        if self.BERT_SIZE != self.BERT_INTERACT_SIZE:
            print('Projecting embeddings before aggregation')
            self.proj_agg = torch.nn.Linear(self.BERT_SIZE, self.BERT_INTERACT_SIZE)
            torch.nn.init.xavier_uniform_(self.proj_agg.weight)
        else:
            self.proj_out = None

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

        if self.BERT_INTERACT_SIZE != self.BERT_AGGREG_SIZE:
            print('Projecting embeddings before aggregation')
            self.proj_out = torch.nn.Linear(self.BERT_INTERACT_SIZE, self.BERT_AGGREG_SIZE)
            torch.nn.init.xavier_uniform_(self.proj_out.weight)
        else:
            self.proj_out = None

        self.cls = torch.nn.Linear(self.BERT_AGGREG_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)
    
    def aggreg_param_names(self):
        """
        :return: a list of the aggregate BERT-parameters. Because we assigned the aggregate model
                 to an attribute with the name AGGREG_ATTR, all parameter keys must start with this
                 value followed by a dot.
        """
        return set([k for k in self.state_dict().keys() if k.startswith( f'{AGGREG_ATTR}.')])
    
    def interact_param_names(self):
        """
        :return: a list of the interact BERT-parameters. Because we assigned the interaction model
                 to an attribute with the name INTERACT_ATTR, all parameter keys must start with this
                 value followed by a dot.
        """
        return set([k for k in self.state_dict().keys() if k.startswith( f'{INTERACT_ATTR}.')])


    def forward(self, query_tok, query_mask, doc_tok, doc_mask):

        batch_qty, _ = query_tok.shape

        last_layer_doc_rep, doc_masks, sbcount = self.encode_bert(doc_tok, 
                                                      doc_mask, 
                                                      is_query=False)
        
        B, N, BERT_SIZE = last_layer_doc_rep.shape
        
        last_layer_query_rep, query_masks, _ = self.encode_bert(query_tok, 
                                                            query_mask, 
                                                            is_query=True)
        last_layer_query_rep = torch.cat([last_layer_query_rep] * sbcount, dim=0)
        query_masks = torch.cat([query_masks] * sbcount, dim=0)
        assert len(last_layer_query_rep.shape) == 3
        _, Q, _ = last_layer_query_rep.shape
        assert last_layer_query_rep.size(0) == B
        assert last_layer_query_rep.size(2) == BERT_SIZE
      
        if self.proj_agg is not None:
            last_layer_doc_rep = self.proj_agg(last_layer_doc_rep)  # [B, Doc_SUB_BATCH_SEQUENCE LENGTH , Aggreg_HIDDEN_SIZE]
            last_layer_query_rep = self.proj_agg(last_layer_query_rep)
            
        interact_cls_tok_exp = self.bert_interact_cls_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B, 1, 
                                                                                          self.BERT_INTERACT_SIZE)
        interact_sep_tok_exp = self.bert_interact_sep_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B, 1, 
                                                                                          self.BERT_INTERACT_SIZE)
    
        EXPECT_TOT_INTERACT_N = N + 4 + Q
        # [B, 4 + DOC_SUB_BATCH_SEQUENCE LENGTH + QUERY_SEQUENCE_LENGTH, Aggreg_HIDDEN_SIZE]
        interact_repr = torch.cat([interact_cls_tok_exp, interact_sep_tok_exp,
                                 last_layer_query_rep, interact_sep_tok_exp,
                                 last_layer_doc_rep, interact_sep_tok_exp], dim=1)

        assert interact_repr.shape == (B, EXPECT_TOT_INTERACT_N, self.BERT_INTERACT_SIZE)

        ONES = torch.ones_like(doc_masks[:, :1]) # B*sbcountx1
        NILS = torch.zeros_like(doc_masks[:, :1]) # B*sbcountx1

        # Token type IDs for segment tokens
        segment_ids = torch.cat([NILS]*2 + [NILS] * (1 + last_layer_query_rep.shape[1])
                              + [NILS] * (1 + last_layer_doc_rep.shape[1]), dim=1)

        # Generate positional encodings manually
        position_ids = torch.arange(interact_repr.size(1),
                                    device=interact_repr.device).unsqueeze(0).expand(B, -1)

        # Assemble attention mask
        attention_mask = torch.cat([ONES, ONES, query_masks, ONES, doc_masks, ONES], dim=1)

        interact_output = self.bert_interact(inputs_embeds=interact_repr,
                              token_type_ids=segment_ids.long(),
                              position_ids=position_ids,
                              attention_mask=attention_mask)

        last_layer_interact_rep = interact_output.last_hidden_state

        cls_output = last_layer_interact_rep[:, 0, :]
        cls_result = []
        for i in range(cls_output.shape[0] // batch_qty):
            cls_result.append(cls_output[i*batch_qty:(i+1)*batch_qty])

        cls_result = torch.stack(cls_result, dim=2)


        last_layer_cls_rep = torch.transpose(cls_result, 1, 2)  #  [B, EXPECT_TOT_INTERACT_N, BERT_INTERACT_SIZE]

        B, N, _ = last_layer_cls_rep.shape

        if self.proj_out is not None:
            last_layer_cls_rep_proj = self.proj_out(last_layer_cls_rep) # [B, EXPECT_TOT_INTERACT_N, BERT_AGGREG_SIZE]
        else:
            last_layer_cls_rep_proj = last_layer_cls_rep

        aggreg_cls_tok_exp = self.bert_aggreg_cls_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B, 1, 
                                                                                                 self.BERT_AGGREG_SIZE)
        aggreg_sep_tok_exp = self.bert_aggreg_sep_embed.unsqueeze(dim=0).unsqueeze(dim=0).expand(B, 1, 
                                                                                                 self.BERT_AGGREG_SIZE)
        EXPECT_TOT_AGGREG_N = N + 3

        # We need to prepend a CLS token vector as the classifier operation depends on the existence of such special token!
        mask = torch.ones_like(last_layer_cls_rep_proj[..., 0])
        last_layer_cls_rep_proj = torch.cat([aggreg_cls_tok_exp, aggreg_sep_tok_exp, 
                                             last_layer_cls_rep_proj, aggreg_sep_tok_exp], dim=1) #[B, EXPECT_TOT_INTERACT_N+3, BERT_AGGREG_SIZE]

        ONES = torch.ones_like(doc_mask[:, :1]) # Bx1
        NILS = torch.zeros_like(doc_mask[:, :1]) # Bx1

        mask = torch.cat([ONES, ONES, mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * 2 + [ONES] * (N) + [NILS], dim=1)

        assert last_layer_cls_rep_proj.shape == (B, EXPECT_TOT_AGGREG_N, self.BERT_AGGREG_SIZE)
        assert mask.shape == (B, EXPECT_TOT_AGGREG_N)
        assert segment_ids.shape == (B, EXPECT_TOT_AGGREG_N)

        outputs : BaseModelOutputWithPoolingAndCrossAttentions = self.bert_aggreg(inputs_embeds=last_layer_cls_rep_proj,
                                                                                  token_type_ids=segment_ids.long(),
                                                                                  attention_mask=mask) 
        result = outputs.last_hidden_state

        # The cls vector of the last Transformer output layer
        cls_reps = result[:, 0, :]

        out = self.cls(self.dropout(cls_reps))

        return out.squeeze(dim=-1)
