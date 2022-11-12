import random
from flexneuart.data_augmentation import get_augmentation_method
import json

class DataAugmentModule:
    def __init__(self, augment_type, random_seed=42, augment_p=0.25):
        self.p = augment_p
        random.seed(random_seed)
        self.doc_augment = None
        config_path = "/home/ubuntu/efs/capstone/data_aug/FlexNeuART/flexneuart/data_augmentation/rule_based/parameters.json"
        conf_file = open(config_path)
        conf = json.load(conf_file)
        conf_file.close()        
        try:
            self.doc_augment = get_augmentation_method(augment_type, conf)
        except:
            print("No Augmentation Registered Augmentation Found by the name {0}".format(augment_type))
            pass

    def augment(self, query_text, doc_text):
        if self.doc_augment is not None and random.random() < self.p:
            doc_text = self.doc_augment.augment(doc_text)
        return query_text, doc_text
