from flexneuart.data_augmentation.utils.base_class import DataAugment

import re
import random
import nltk
from nltk.corpus import wordnet, stopwords

nltk.download('wordnet') 
nltk.download('stopwords')
nltk.download('omw-1.4')

class SynonymWordReplacement(DataAugment):
    """                                                                                                                                                                                                     
    A class used for replacing words with their synonyms using nltk                                                                                                                                                                                                                                                                                                                      
    ...                                                                                                                                                                                                     
                                                                                                                                                                                                            
    Attributes                                                                                                                                                                                              
    ----------                                                                                                                                                                                              
    alpha_sr : float                                                                                                                                                                        
        percentage of words to be replaced by their synonyms                                                                                                                                                 
    stopwords: list(string)  
        list of the english stopwords from the nltk corpus                                                                                                                 
                                                                                                                                                                                                            
    Methods                                                                                                                                                                                                 
    -------                                                                                                                                                                                                 
    augment(text)                                                                                                                                                                                           
        returns the augmented text

    """

    def __init__(self, alpha_sr=0.1):
        """                                                                                                                                                                                                                                                                                                                                                               
        Parameters                                                                                                                                                                                          
        ----------                                                                                                                                                                                          
        alpha_sr : float                                                                                                                                                                                          
            percentage of words to be replaced by their synonym                                                                                                                                                                          
        """
        super().__init__()
        self.alpha_sr = alpha_sr 
        self.stopwords = stopwords.words('english')

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        number_sr = max(1, int(self.alpha_sr*num_words))
        augmented_words = self.__synonym_replacement(words, number_sr)
        return ' '.join(augmented_words)

    def __synonym_replacement(self, words, n):
        """  
        Given a list of words, n words in the list are replaced by synonyms 
        Parameters                                                                                                                                                                                          
        ----------                                                                                                                                                                                          
        words : list(string)  
            list of words in which n words are to be replaced by synonyms
        n: int
            number of words to replace with synonym

        """
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stopwords]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.__get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n: 
                break


        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words

    def __get_synonyms(self, word):
        """      
        Given a word, nltk wordnet synonym set is used to identify and 
        return a synonym list for the word

        Parameters                                                                                                                                                                                          
        ----------                                                                                                                                                                                          
        word : string                                                                                                                                                                                          
            word for which synonyms are to be fetched
        """
        
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
    """                                                                                                                                                                                                     
    A class used for replacing words with their hypernyms using nltk                                                                                                                                                                                                                                                                                                                      
    ...                                                                                                                                                                                                     
                                                                                                                                                                                                            
    Attributes                                                                                                                                                                                              
    ----------                                                                                                                                                                                              
    alpha_hr : float                                                                                                                                                                        
        percentage of words to be replaced by their hypernym                                                                                                                                                  
    stopwords: list(string)   
        list of the english stopwords from the nltk corpus                                                                                                                 
                                                                                                                                                                                                            
    Methods                                                                                                                                                                                                 
    -------                                                                                                                                                                                                 
    augment(text)                                                                                                                                                                                           
        returns the augmented text
                                                                                                                                                                              
    """
    
    def __init__(self, alpha_hr=0.1):
        """                                                                                                                                                                                                                                                                                                                                                               
        Parameters                                                                                                                                                                                          
        ----------                                                                                                                                                                                          
        alpha_hr : float                                                                                                                                                                                          
            percentage of words to be replaced by their hypernym                                                                                                                                                                          
        """
        super().__init__()
        self.alpha_hr = alpha_hr 
        self.stopwords = stopwords.words('english')

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        number_hr = max(1, int(self.alpha_hr*num_words))
        augmented_words = self.__hypernym_replacement(words, number_hr)
        return ' '.join(augmented_words).replace('_', ' ')

    def __hypernym_replacement(self, words, n):
        """                                    
        Given a list of words, n words in the list are replaced by hypernyms                                                                                                                                                                                                                                                                                                                        
        Parameters                                                                                                                                                                                          
        ----------                                                                                                                                                                                          
        words : list(string)   
            list of words in which n words are to be replaced by hypernyms                                                                                                                                                                                       
        n : int 
            number of words to be replaced by their hypernyms                                                                                                                                                                       
        """
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stopwords]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            hypernyms = self.__get_hypernyms(random_word)
            if len(hypernyms) >= 1:
                hypernym = random.choice(list(hypernyms))
                new_words = [hypernym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n: 
                break

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words

    def __get_hypernyms(self, word):
        """      
        Given a word, nltk wordnet synsets is used to identify and 
        return a hypernym list for the word

        Parameters                                                                                                                                                                                          
        ----------                                                                                                                                                                                          
        word : string                                                                                                                                                                                          
            word for which hypernyms are to be fetched
        """
        hypers_lst = []
        try:
            s = wordnet.synsets(word)[0]
        except:
            return hypers_lst
        if s.name() == 'restrain.v.01':
            return hypers_lst
        hypers = lambda s:s.hypernyms()
        hypers = list(s.closure(hypers))
        for syn in hypers:
            for l in syn.lemmas():
                if l.name().lower() != word:
                    hypers_lst.append(l.name().lower())
        return list(dict.fromkeys(hypers_lst))
