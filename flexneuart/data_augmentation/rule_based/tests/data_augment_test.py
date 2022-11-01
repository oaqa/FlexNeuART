from abc import abstractmethod

class DataAugmentTest:
    def __init__(self):
        return
        
    @abstractmethod
    def augment(self, text, **kwargs):
        pass

    @classmethod
    def build_augmentation(cls, conf):
        return cls(conf)