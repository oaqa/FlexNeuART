#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import re

def is_equal(text_str_1, text_str_2):
    return text_str_1 == text_str_2


def tokenized_equal(text_str_1, text_str_2):
    """Tokenizes input strings and matches the token sets
    """
    token_list1 = list(map(lambda x: x.lower(), re.sub('[^a-zA-Z0-9 ]', ' ', text_str_1).split()))
    token_list2 = list(map(lambda x: x.lower(), re.sub('[^a-zA-Z0-9 ]', ' ', text_str_2).split()))
    return set(token_list1) == set(token_list2)

