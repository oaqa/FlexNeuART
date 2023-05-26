from flexneuart.data_augmentation.rule_based.synonym_hypernym_transformations import SynonymWordReplacement, HypernymWordReplacement
from test_conf import conf

class TestSynonymTransformations():
    def test_synonym_transformation(self):
        text = "Testing random deletions in this sentence. Do you think words are correctly deleted"
        random_insertion = SynonymWordReplacement("synonym_word_replacement", conf)
        print(random_insertion.augment(text))
    
    def test_hypernym_transformation(self):
        text = "Testing random deletions in this sentence. Do you think words are correctly deleted"
        random_insertion = HypernymWordReplacement("hypernym_word_replacement", conf)
        print(random_insertion.augment(text))

if __name__ == "__main__":
    tester = TestSynonymTransformations()
    tester.test_synonym_transformation()
    tester.test_hypernym_transformation()
