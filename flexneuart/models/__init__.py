from flexneuart import Registry
model_registry = Registry()
register = model_registry.register

VANILLA_BERT='vanilla_bert'

from flexneuart.models import vanilla_bert_standard, \
                            bert_aggreg_p, \
                            parade, \
                            longformer, \
                            cedr, \
                            ndrm, \
                            biencoder, \
                            colbert
