import unittest
from document_level_transformation import ConstantDocLength

class TestConstantDocLength(unittest.TestCase):
    def test_constant_doc_length(self):
        sentence = "This is a test sentence to check constant doc length."
        constant_doc_length = ConstantDocLength(doc_length=7)
        truncated_sentence = constant_doc_length.augment(sentence)
        assert len(truncated_sentence.split())==7
        print("Sentence of constant length is:", truncated_sentence)
    
if __name__ == "__main__":
    tester = TestConstantDocLength()
    tester.test_constant_doc_length()
    