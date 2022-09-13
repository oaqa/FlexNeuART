from character_transformation import *

sentence = "If you're looking for random paragraphs, you've come to the right place. When a random word or a random sentence isn't quite enough, the next logical step is to find a random paragraph. We created the Random Paragraph Generator with you in mind. The process is quite simple. Choose the number of random paragraphs you'd like to see and click the button. Your chosen number of paragraphs will instantly appear."

class TestCharacterTransformations():
    def test_add_character_transformation(self):
        add_character = AddCharacterTransformation(word_add_probability=0.1, character_add_probability=0.1)
        print(add_character.augment(sentence))

    def test_remove_character_transformation(self):
        remove_character = RemoveCharacterTransformation(word_remove_probability=0.1, character_remove_probability=0.1)
        print(remove_character.augment(sentence))

    def test_swap_character_transformation(self):
        remove_character = SwapCharacterTransformation(word_swap_probability=0.1, character_swap_probability=0.1)
        print(remove_character.augment(sentence))
        
    def test_replace_character_transformation(self):
        replace_character = ReplaceCharacterTransformation(word_replace_probability=0.1, character_replace_probability=0.1)
        print(replace_character.augment(sentence))

    def test_keyboard_replace_character_transformation(self):
        replace_character = ReplaceCharacterKeyboardTransformation(word_replace_probability=0.1, character_replace_probability=0.1)
        print(replace_character.augment(sentence))

    def test_add_character_keyboard_transformation(self):
        add_character = AddCharacterKeyboardAdjacentTransformation(word_add_probability=0.1, character_add_probability=0.1)
        print(add_character.augment(sentence))

if __name__ == "__main__":
    tester = TestCharacterTransformations()
    tester.test_add_character_keyboard_transformation()
