from flexneuart.data_augmentation.rule_based.character_transformation import *
from test_conf import conf

sentence = "If you're looking for random paragraphs, you've come to the right place. When a random word or a random sentence isn't quite enough, the next logical step is to find a random paragraph. We created the Random Paragraph Generator with you in mind. The process is quite simple. Choose the number of random paragraphs you'd like to see and click the button. Your chosen number of paragraphs will instantly appear."

class TestCharacterTransformations():
    def test_add_character_transformation(self):
        add_character = AddCharacterTransformation("random_character_insertion", conf)
        print(add_character.augment(sentence))

    def test_remove_character_transformation(self):
        remove_character = RemoveCharacterTransformation("random_character_deletion", conf)
        print(remove_character.augment(sentence))

    def test_swap_character_transformation(self):
        remove_character = SwapCharacterTransformation("random_character_swap", conf)
        print(remove_character.augment(sentence))
        
    def test_replace_character_transformation(self):
        replace_character = ReplaceCharacterTransformation("random_character_replace", conf)
        print(replace_character.augment(sentence))

    def test_keyboard_replace_character_transformation(self):
        replace_character = ReplaceCharacterKeyboardTransformation("keyboard_character_replace", conf)
        print(replace_character.augment(sentence))

    def test_add_character_keyboard_transformation(self):
        add_character = AddCharacterKeyboardAdjacentTransformation("keyboard_character_insertion", conf)
        print(add_character.augment(sentence))

if __name__ == "__main__":
    tester = TestCharacterTransformations()
    tester.test_add_character_transformation()
    tester.test_remove_character_transformation()
    tester.test_swap_character_transformation()
    tester.test_replace_character_transformation()
    tester.test_keyboard_replace_character_transformation()
    tester.test_add_character_keyboard_transformation()
