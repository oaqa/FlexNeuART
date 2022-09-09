from flexneuart.data_augmentation.utils.base_class import DataAugment
import random

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

        # check if current length of the document is greater than max specified length
        if old_length>self.doc_length:
            #sample indices that will be deleted from the document
            indices_to_delete = random.sample(range(old_length),old_length-self.doc_length)

            for ind,word in enumerate(tokens):
                if ind not in indices_to_delete:
                    constant_length_text.append(word)          
        
        return " ".join(constant_length_text)