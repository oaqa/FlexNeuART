import re

def is_equal(text_str_1, text_str_2):
    return text_str_1 == text_str_2


def tokenized_equal(text_str_1, text_str_2):
    """Tokenizes input strings and matches the token sets
    """
    token_list1 = list(map(lambda x: x.lower(), re.sub('[^a-zA-Z0-9 ]', ' ', text_str_1).split()))
    token_list2 = list(map(lambda x: x.lower(), re.sub('[^a-zA-Z0-9 ]', ' ', text_str_2).split()))
    return set(token_list1) == set(token_list2)