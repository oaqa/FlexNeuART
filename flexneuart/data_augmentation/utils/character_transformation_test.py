
from character_transformation import AddCharacterTransformation, RemoveCharacterTransformation, SwapCharacterTransformation

class TestCharacterTransformations():
    def test_add_character_insertions(self):
        # answer = True
        # self.assertEqual(answer, True)
        sentence = "This is a test sentence"
        add_character = AddCharacterTransformation(word_add_probability=0.3, character_add_probability=0.1)
        print(add_character.augment(sentence))
    def test_remove_character_insertions(self):
        # answer = True
        # self.assertEqual(answer, True)
        sentence = "This is a test sentence. Writing this to get a longer sentence"
        remove_character = RemoveCharacterTransformation(word_remove_character_probability=0.3, character_remove_probability=0.1)
        print(remove_character.augment(sentence))
    def test_swap_character_insertions(self):
        # answer = True
        # self.assertEqual(answer, True)
        sentence = "This is a test sentence. Writing this to get a longer sentence"
        remove_character = SwapCharacterTransformation(word_level_probability=0.3, character_swap_probability=0.1)
        print(remove_character.augment(sentence))

if __name__ == "__main__":
    tester = TestCharacterTransformations()
    tester.test_swap_character_insertions()
