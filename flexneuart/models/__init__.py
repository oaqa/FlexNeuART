from flexneuart import Registry
model_registry = Registry()
register = model_registry.register

VANILLA_BERT='vanilla_bert'
VANILLA_BERT_FLASH_ATTN='vanilla_bert_flash_attn'

from flexneuart.models import vanilla_bert_standard, \
                            bert_aggreg_p, \
                            parade, \
                            longformer, \
                            cedr, \
                            ndrm, \
                            biencoder, \
                            colbert, \
                            mores
