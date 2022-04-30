from flexneuart.data_augmentation.utils.base_class import DataAugment
from nltk.corpus import wordnet
import re
import spacy
import random
class SynonymWordReplacement(DataAugment):

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

class HypernymWordReplacement(DataAugment):
    def __init__(self, alpha_hr=0.1, random_seed=42):
        super().__init__(random_seed)
        self.alpha_hr = alpha_hr # percentage of synonym replacement
    def augment(self, text, **kwargs):
<<<<<<< HEAD
        words = re.split('\s+', text)
        num_words = len(words)
        number_hr = max(1, int(self.alpha_hr*num_words))
        augmented_words = self.hypernym_replacement(words, number_hr)
        return ' '.join(augmented_words)

    def hypernym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stopwords]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            hypernyms = self.get_hypernyms(random_word)
            if len(hypernyms) >= 1:
                hypernym = random.choice(list(hypernyms))
                new_words = [hypernym if word == random_word else word for word in new_words]
                #print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n: #only replace up to n words
                break

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words

    def get_hypernyms(word):
        hypers_lst = []
        try:
            s = wordnet.synsets(word)[0]
        except:
            return hypers_lst
        if s.name() == 'restrain.v.01':
            print("RESTRAIN ENCOUNTERED (hypers)")
            return hypers_lst
        hypers = lambda s:s.hypernyms()
        hypers = list(s.closure(hypers))
        for syn in hypers:
            for l in syn.lemmas():
                if l.name().lower() != word:
                    hypers_lst.append(l.name().lower())
        return list(dict.fromkeys(hypers_lst))
=======
        pass
>>>>>>> b9b8ea161560b8b303bb4a83bbdbc5a3cad18744
