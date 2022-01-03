#!/usr/bin/env python
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
"""
    Adding a field stemmed using a Krovetz stemmer.
    Based on https://github.com/bmitra-msft/TREC-Deep-Learning-Quick-Start/blob/master/model_utils.py

    It requires installing an extra package: krovetzstemmer
"""
import argparse
import json
import re

from tqdm import tqdm

from flexneuart.io import jsonl_gen, FileWrapper
import krovetzstemmer

from flexneuart.config import TEXT_RAW_FIELD_NAME

STEMMED_FIELD_NAME='text_krovetz'

parser = argparse.ArgumentParser(description='Add doc2query fields to the existing JSONL data entries')

parser.add_argument('--input', metavar='input JSONL file', help='input JSONL file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--output', metavar='output JSONL file', help='output JSONL file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--src_field', metavar='source field',
                    help='the name of field whose context is stemmed and stopped',
                    type=str, default=TEXT_RAW_FIELD_NAME)
parser.add_argument('--dst_field', metavar='target field',
                    help='the name of field to store the stemmed and stopped text',
                    type=str, default=STEMMED_FIELD_NAME)



args = parser.parse_args()
print(args)

stop_words = ['a', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although',
                           'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'aren', 'around',
                           'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below',
                           'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes',
                           'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', 'couldn', 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', 'didn', 'different',
                           'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every',
                           'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from',
                           'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'hadn', 'happens', 'hardly', 'has', 'hasn', 'have', 'haven', 'having', 'he',
                           'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', 'ie', 'if', 'ignored',
                           'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'isn', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know',
                           'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 'likely', 'little', 'll', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe',
                           'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never',
                           'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once',
                           'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please',
                           'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said',
                           'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she',
                           'should', 'shouldn', 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub',
                           'such', 'sup', 'sure', 't', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby',
                           'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took',
                           'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using',
                           'usually', 'uucp', 'v', 've', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', 'wasn', 'way', 'we', 'welcome', 'well', 'went', 'were', 'weren', 'what', 'whatever', 'when', 'whence', 'whenever',
                           'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within',
                           'without', 'won', 'wonder', 'would', 'would', 'wouldn', 'x', 'y', 'yes', 'yet', 'you', 'youve', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'z', 'zero']
print(stop_words)

stop_words = set(stop_words)

src_field = args.src_field
dst_field = args.dst_field

print(f'Source field: {src_field} target field: {dst_field}')

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')
stemmer = krovetzstemmer.Stemmer()


def clean_text(s):
    s = regex_multi_space.sub(' ', regex_drop_char.sub(' ', s.lower())).strip()
    s = ' '.join([stemmer(t) for t in s.split() if t not in stop_words])
    return s


with FileWrapper(args.output, 'w') as outf:
    for doce in tqdm(jsonl_gen(args.input), desc='adding '):
        doce[dst_field] = clean_text(doce[src_field])

        outf.write(json.dumps(doce) + '\n')
