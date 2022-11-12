import random
from flexneuart.data_augmentation import get_augmentation_method
from flexneuart.data_augmentation.rule_based.utils.parameters import conf as default_conf
import json

class DataAugmentModule:
    def __init__(self, augment_type, config_path, augment_p=0.25):
        self.p = augment_p
        random.seed(random_seed)
        self.doc_augment = None
        if config_path is None:
            conf = default_conf
        else:
            try:
                conf_file = open(config_path)
                conf = json.load(conf_file)
                conf_file.close()
            except:
                raise Exception("Please ensure that the path to the config file is correct.")
        try:
            self.doc_augment = get_augmentation_method(augment_type, conf)
        except:
            raise Exception("No Augmentation Registered Augmentation Found by the name {0}".format(augment_type))
            pass

    def augment(self, query_text, doc_text):
        if self.doc_augment is not None and random.random() < self.p:
            doc_text = self.doc_augment.augment(doc_text)
        return query_text, doc_text
