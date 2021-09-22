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

from flexneuart.config import TEXT_BERT_TOKENIZED_NAME
from flexneuart.text_proc.parse import get_bert_tokenizer

MAX_NUM_QUERY_OPT = 'max_num_query'
MAX_NUM_QUERY_OPT_HELP = 'maximum # of queries to generate'
BERT_TOK_OPT = 'bert_tokenize'
BERT_TOK_OPT_HELP = 'Apply the BERT tokenizer and store result in a separate field'
ENABLE_POS_OPT = 'enable_pos'
ENABLE_POS_OPT_HELP = 'Enable POS tagging for more accurate lemmatization'
OUT_BITEXT_PATH_OPT = 'out_bitext_path'
OUT_BITEXT_PATH_OPT_META = 'optional bitext path'
OUT_BITEXT_PATH_OPT_HELP = 'An optional output directory to store bitext'

MSMARCO_DOC_V2_FILE_PATTERN = "^msmarco_doc_.*"
MSMARCO_PASS_V2_FILE_PATTERN = "^msmarco_passage_.*"


def unique(arr):
    return list(set(arr))


def build_query_id_to_partition(query_ids, sizes):
    """Partition a given list of query IDs.

    :param query_ids:   an input array of query IDs.
    :param sizes:       partion sizes

    :return:  a dictionary that maps each query ID to its respective partition ID
    """
    assert sum(sizes) == len(query_ids)
    query_id_to_partition = dict()
    start = 0
    for part_id in range(len(sizes)):
        end = start + sizes[part_id]
        for k in range(start, end):
            query_id_to_partition[query_ids[k]] = part_id
        start = end

    return query_id_to_partition


def add_bert_tok_args(parser):
    """Add an argument for optional BERT tokenization"""
    parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)


def create_bert_tokenizer_if_needed(args):
    """Create the BERT tokenizer if specified in command-line arguments.

    :param args:    *PARSED* arguments
    :return:   a BERT tokenizer object or None if BERT-based tokenization was not requested.
    """
    arg_vars = vars(args)
    bert_tokenizer = None
    if arg_vars[BERT_TOK_OPT]:
        print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
        bert_tokenizer = get_bert_tokenizer()

    return bert_tokenizer


def add_basic_query_split_args(parser):
    parser.add_argument('--seed',
                        metavar='random seed',
                        help='random seed',
                        type=int, default=0)
    parser.add_argument('--partitions_names',
                        metavar='names of partitions to split at',
                        help='names of partitions to split at separated by comma',
                        required=True,
                        type=str)
    parser.add_argument('--partitions_sizes',
                        metavar='sizes of partitions to split at',
                        help="sizes (in queries) of partitions to split at separated by comma (one of the values can be missing, "
                             "in that case all left queries go to that partition)",
                        required=True,
                        type=str)


class QuerySplitArgumentsBase:
    def __init__(self, raw_args):
        self.raw_args = raw_args

    @property
    def src_dir(self):
        return self.raw_args.src_dir

    @property
    def dst_dir(self):
        return self.raw_args.dst_dir

    @property
    def seed(self):
        return self.raw_args.seed

    @property
    def partitions_names(self):
        return self.raw_args.partitions_names.split(',')

    def partitions_sizes(self, queries_count):
        raw_values = []
        for value in self.raw_args.partitions_sizes.split(','):
            if value == '':
                raw_values.append(-1)
            else:
                raw_values.append(int(value))
        nondefined_count = 0
        defined_sum = 0
        for value in raw_values:
            if value != -1:
                if value <= 0:
                    raise Exception('One query list is empty!')
                if value >= queries_count:
                    raise Exception(f'A partition is too big, the total number of queries is only: {queries_count}')
                defined_sum += value
            else:
                nondefined_count += 1

        if nondefined_count == 0 and defined_sum == queries_count:
            return raw_values
        elif nondefined_count == 1 and defined_sum < queries_count:
            raw_values[raw_values.index(-1)] = queries_count - defined_sum
            return raw_values
        else:
            raise ValueError("invalid --partitions_sizes argument")


