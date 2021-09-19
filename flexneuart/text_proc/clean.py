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


"""
    Basic text cleaning helpers.
"""

def remove_tags(str):
    """Just remove anything that looks like a tag"""
    return re.sub(r'</?[a-z]+\s*/?>', '', str)


def replace_chars_nl(s):
    """Replace \n and \r characters with spaces"""
    return re.sub(r'[\n\r]', ' ', s)


def replace_tab(orig_str, repl_str=' '):
    return orig_str.replace('\t', repl_str)



