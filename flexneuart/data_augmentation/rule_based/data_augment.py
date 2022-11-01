from abc import abstractmethod

class DataAugment:
    def __init__(self, name):
        self.augmentation_name = name
        return
        
    @abstractmethod
    def augment(self, text, **kwargs):
        pass

    @classmethod
    def build_augmentation(cls, conf):
        return cls(conf)
     
