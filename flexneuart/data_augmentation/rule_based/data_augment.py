from abc import abstractmethod
class DataAugment:
    def __init__(self, name):
        self.augmentation_name = name
        self.types = "both"
        return
        
    @abstractmethod
    def augment(self, text, **kwargs):
        pass
    
    def augment_select(self, query, doc, **kwargs):
        if self.types=="document":
            doc = self.augment(doc, **kwargs)
        elif self.types=="query":
            query = self.augment(query, **kwargs)
        elif self.types=="both":
            doc = self.augment(doc, **kwargs)
            query = self.augment(query, **kwargs)
        else:
            raise NotImplementedError("Augmentation type needs to be one of -> [document, query, both]. Check config file")
        
        return query, doc
        
            

    @classmethod
    def build_augmentation(cls, conf):
        return cls(conf)
     
