from flexneuart.data_augmentation.utils.base_class import DataAugment
from nltk.corpus import wordnet
import re
import spacy
import random
class SynonymReplacement(DataAugment):

    def __init__(self, alpha_sr=0.1, random_seed=42):
        super().__init__(random_seed)
        self.alpha_sr = alpha_sr # percentage of synonym replacement
        self.en = spacy.load('en_core_web_sm')
        self.stopwords = self.en.Defaults.stop_words

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        number_sr = max(1, int(self.alpha_sr*num_words))
        augmented_words = self.synonym_replacement(words, number_sr)
        return ' '.join(augmented_words)

    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stopwords]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                #print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n: #only replace up to n words
                break

        #this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

class HypernymReplacement(DataAugment):
    def augment(self, text, **kwargs):
        pass
