import random
from flexneuart.data_augmentation import get_augmentation_method
from flexneuart.data_augmentation.rule_based.utils.parameters import conf as default_conf
import json

class DataAugmentModule:
    def __init__(self, augment_type, config_path, augment_p):
        self.p = augment_p
        self.doc_augment_techniques = list()
        if config_path is None:
            conf = default_conf
        else:
            try:
                conf_file = open(config_path)
                conf = json.load(conf_file)
                conf_file.close()
            except:
                raise Exception("Please ensure that the path to the config file is correct.")

        for technique in augment_type:
            self.doc_augment_techniques.append(get_augmentation_method(technique, conf))
                

    def augment(self, query_text, doc_text):
        for technique in self.doc_augment_techniques:
            if random.random() < self.p:
                query_text, doc_text = technique.augment_select(query_text, doc_text)
        return query_text, doc_text
