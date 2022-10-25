from flexneuart.data_augmentation.utils.base_class import DataAugment
import random
import re

class ConstantDocLength(DataAugment):
    """
    Randomly delete words reduce the document length
    ...

    Attributes
    ----------
    doc_length : int 
        The maximum number of words in document

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, doc_length):
        super().__init__()
        self.doc_length = doc_length

    def augment(self, text):
        tokens = text.split()
        old_length = len(tokens)
        constant_length_text = []

        # check if current length of the document is less than equal max specified length
        if old_length<=self.doc_length:
            return text
        
        #sample indices that will be deleted from the document
        indices_to_delete = random.sample(range(old_length),old_length-self.doc_length)

        for ind,word in enumerate(tokens):
            if ind not in indices_to_delete:
                constant_length_text.append(word)        
        
        return " ".join(constant_length_text)

class DocuemntCutOut(DataAugment):
    """
    Randomly drop a span of the document
    ...

    Attributes
    ----------
    p: probability of dropping a span in the document
    span_p: percentage of the document to cut out

    Methods
    -------
    augment(text)
        returns the augmented text
    """

    def __init__(self, p=0.1, span_p=0.2):
        super().__init__()
        self.p = p
        self.span_p = span_p
    
    def augment(self, text):
        r = random.uniform(0, 1)
        if r > self.p:
            return text
        
        words = re.split('\s+', text)
        num_words_to_drop = len(words)*self.span_p
        index_to_cut = random.randint(0, len(words))
        augmented_words = words[:index_to_cut] + words[index_to_cut+num_words_to_drop:]
        
        return ' '.join(augmented_words)


class QueryTextDrop(DataAugment):
    """
    Randomly drop a span of the document
    ...

    Attributes
    ----------
    p: probability of removing query words from a document

    Methods
    -------
    augment(text, query)
        returns the augmented text
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def augment(self, text, query=None):
        r = random.uniform(0, 1)
        if r > self.p:
            return text        
        doc_words = re.split('\s+', text)
        query_words = set(re.split('\s+', query))

        augmented_words = [w for w in doc_words if w not in query_words]

        return ' '.join(augmented_words)

        