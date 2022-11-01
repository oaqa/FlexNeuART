from synonym_hypernym_transformations import SynonymWordReplacement, HypernymWordReplacement

class TestSynonymTransformations():
    def test_synonym_transformation(self):
        text = "Testing random deletions in this sentence. Do you think words are correctly deleted"
        random_insertion = SynonymWordReplacement(alpha_sr=0.2)
        print(random_insertion.augment(text))
    
    def test_hypernym_transformation(self):
        text = "Testing random deletions in this sentence. Do you think words are correctly deleted"
        random_insertion = HypernymWordReplacement(alpha_hr=0.2)
        print(random_insertion.augment(text))

if __name__ == "__main__":
    tester = TestSynonymTransformations()
    tester.test_synonym_transformation()
    tester.test_hypernym_transformation()
