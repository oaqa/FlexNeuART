from flexneuart.data_augmentation.rule_based.data_augment import DataAugment
from flexneuart.data_augmentation import get_registered_name, register_augmentation

@register_augmentation("dummy_augmentation")
class DummyAugmentation(DataAugment):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        print(get_registered_name(self.__class__))
    
    def augment(self, text, query=None):
        print(text)
        print(query)
        return 0
