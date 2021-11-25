from flexneuart import Registry
model_registry = Registry(default='models')
register = model_registry.register

VANILLA_BERT='vanilla_bert'

from flexneuart.models import vanilla_bert_no_qry_pad, \
                            bert_aggreg_p, \
                            parade, \
                            longformer, \
                            cedr, \
                            ndrm
from flexneuart.models import model1
