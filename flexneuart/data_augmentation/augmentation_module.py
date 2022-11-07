import random
# from abc import abstractmethod
from flexneuart.data_augmentation import get_augmentation_method
import json
# from flexneuart.data_augmentation.rule_based.abnirml_transformation import *
# from flexneuart.data_augmentation.rule_based.random_word_transformations import *
# # from flexneuart.data_augmentation.utils.synonym_hypernym_transformations import *
# from flexneuart.data_augmentation.rule_based.document_level_transformation import *
# from flexneuart.data_augmentation.rule_based.character_transformation import *
# from flexneuart.data_augmentation.rule_based.synonym_hypernym_transformations import *

class DataAugmentModule:
    def __init__(self, augment_type, random_seed=42, augment_p=0.25):
        self.p = augment_p
        random.seed(random_seed)
        self.doc_augment = None
        config_path = "/home/ubuntu/efs/capstone/data_aug/FlexNeuART/flexneuart/data_augmentation/rule_based/parameters.json"
        conf_file = open(config_path)
        conf = json.load(conf_file)
        conf_file.close()
        
        # if augment_type == 'random_word_deletion':
        #     self.doc_augment = RandomWordDeletion(p = 0.05)
        # elif augment_type == 'random_word_insertion':
        #     self.doc_augment = RandomWordInsertion(alpha_ri = 0.05)
        # elif augment_type == 'random_word_swap':
        #     self.doc_augment = RandomWordSwap(alpha_rs = 0.05)
        # elif augment_type == 'synonym_word_replacement':
        #     self.doc_augment = SynonymWordReplacement(alpha_sr = 0.05)
        # # elif augment_type == 'hypernym_word_replacement':
        # #    self.doc_augment = HypernymWordReplacement(alpha_hr = 0.05)
        # # elif augment_type == 'constant_document_length':
        # #     self.doc_augment = ConstantDocLength(doc_length=500) # doc length needs to be decided
        # elif augment_type == 'random_character_insertion':
        #     self.doc_augment = AddCharacterTransformation(word_add_probability=0.05, character_add_probability=0.1)
        # elif augment_type == 'random_character_deletion':
        #     self.doc_augment = RemoveCharacterTransformation(word_remove_probability=0.05, character_remove_probability=0.1)
        # elif augment_type == 'random_character_swap':
        #     self.doc_augment = SwapCharacterTransformation(word_swap_probability=0.05, character_swap_probability=0.1)
        # elif augment_type == 'random_character_replace':
        #     self.doc_augment = ReplaceCharacterTransformation(word_replace_probability=0.05, character_replace_probability=0.1)
        # elif augment_type == 'keyboard_character_insertion': 
        #     self.doc_augment = AddCharacterKeyboardAdjacentTransformation(word_add_probability=0.05, character_add_probability=0.1)  
        # elif augment_type == 'keyboard_character_replace':    
        #     self.doc_augment = ReplaceCharacterKeyboardTransformation(word_replace_probability=0.05, character_replace_probability=0.1) 
        # elif augment_type == 'document_cut_out':
        #     self.doc_augment = DocuemntCutOut(p=0.1, span_p=0.2)
        # elif augment_type == 'query_cut_out':
        #     self.doc_augment = QueryTextDrop(p=0.5)
        # temp_conf = {"random_character_insertion":{"word_add_probability": 0.1, "character_add_probability":0.1}}
        try:
            self.doc_augment = get_augmentation_method(augment_type, conf)
        except:
            print("No Augmentation Registered Augmentation Found by the name {0}".format(augment_type))
            pass

    def augment(self, query_text, doc_text):
        if self.doc_augment is not None and random.random() < self.p:
            doc_text = self.doc_augment.augment(doc_text)
        return query_text, doc_text
