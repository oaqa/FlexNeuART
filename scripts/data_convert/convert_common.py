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
import gzip, bz2
import collections
import re
import os
import bson
import torch
import json
import struct
import urllib
import urllib.parse
from bs4 import BeautifulSoup

from scripts.config import DEFAULT_ENCODING, STOPWORD_FILE, DOCID_FIELD, QUESTION_FILE_JSON

YahooAnswerRecParsed = collections.namedtuple('YahooAnswerRecParsed',
                                              'uri subject content best_answer_id answer_list')

MAX_NUM_QUERY_OPT = 'max_num_query'
MAX_NUM_QUERY_OPT_HELP = 'maximum # of queries to generate'
BERT_TOK_OPT = 'bert_tokenize'
BERT_TOK_OPT_HELP = 'Apply the BERT tokenizer and store result in a separate field'
ENABLE_POS_OPT = 'enable_pos'
ENABLE_POS_OPT_HELP = 'Enable POS tagging for more accurate lemmatization'
OUT_BITEXT_PATH_OPT = 'out_bitext_path'
OUT_BITEXT_PATH_OPT_META = 'optional bitext path'
OUT_BITEXT_PATH_OPT_HELP = 'An optional output directory to store bitext'


ENDIANNES_TYPE = '<'
PACKED_TYPE_DENSE = 0

# Replace \n and \r characters with spaces
def replace_chars_nl(s):
    return re.sub(r'[\n\r]', ' ', s)


class FileWrapper:

    def __enter__(self):
        return self

    def __init__(self, file_name, flags='r'):
        """Constructor, which opens a regular or gzipped-file

          :param  file_name a name of the file, it has a '.gz' or '.bz2' extension, we open a compressed stream.
          :param  flags    open flags such as 'r' or 'w'
        """
        dir_name = os.path.dirname(file_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if file_name.endswith('.gz'):
            self._file = gzip.open(file_name, flags)
            self._isCompr = True
        elif file_name.endswith('.bz2'):
            self._file = bz2.open(file_name, flags)
            self._isCompr = True
        else:
            self._file = open(file_name, flags)
            self._isCompr = False

    def write(self, s):
        if self._isCompr:
            self._file.write(s.encode())
        else:
            self._file.write(s)

    def read(self, s):
        if self._isCompr:
            return self._file.read().decode()
        else:
            return self._file.read()

    def close(self):
        self._file.close()

    def __exit__(self, type, value, tb):
        self._file.close()

    def __iter__(self):
        for line in self._file:
            yield line.decode() if self._isCompr else line


def read_stop_words(file_name=STOPWORD_FILE, lower_case=True):
    """Reads a list of stopwords from a file. By default the words
       are read from a standard repo location and are lowercased.

      :param file_name a stopword file name
      :param lower_case  a boolean flag indicating if lowercasing is needed.

      :return a list of stopwords
    """
    stop_words = []
    with open(file_name) as f:
        for w in f:
            w = w.strip()
            if w:
                if lower_case:
                    w = w.lower()
                stop_words.append(w)

    return stop_words


def SimpleXmlRecIterator(file_name, rec_tag_name):
    """A simple class to read XML records stored in a way similar to
      the Yahoo Answers collection. In this format, each record
      occupies a certain number of lines, but no record "share" the same
      line. The format may not be fully proper XML, but each individual
      record may be. It always starts with a given tag name ends with
      the same tag, e.g.,:

      <record_tag_name ...>
      </record_tag_name>

    :param file_name:  input file name (can be compressed).
    :param rec_tag_name:   a record tag name (for the tag that encloses the record)

    :return:   it yields a series of records
    """

    with FileWrapper(file_name) as f:

        rec_lines = []

        start_entry = '<' + rec_tag_name
        end_entry = '</' + rec_tag_name + '>'

        seen_end = True
        seen_start = False

        ln = 0
        for line in f:
            ln += 1
            if not seen_start:
                if line.strip() == '':
                    continue  # Can skip empty lines
                if line.startswith(start_entry):
                    if not seen_end:
                        raise Exception(f'Invalid format, no previous end tag, line {ln} file {file_name}')
                    assert (not rec_lines)
                    rec_lines.append(line)
                    seen_end = False
                    seen_start = True
                else:
                    raise Exception(f'Invalid format, no previous start tag, line {ln} file {file_name}')
            else:
                rec_lines.append(line)
                no_space_line = line.replace(' ', '').strip()  # End tags may contain spaces
                if no_space_line.endswith(end_entry):
                    if not seen_start:
                        raise Exception(f'Invalid format, no previous start tag, line {ln} file {file_name}')
                    yield ''.join(rec_lines)
                    rec_lines = []
                    seen_end = True
                    seen_start = False

        if rec_lines:
            raise Exception(f'Invalid trailing entries in the file {file_name} %d entries left' % (len(rec_lines)))


def remove_tags(str):
    """Just remove anything that looks like a tag"""
    return re.sub(r'</?[a-z]+\s*/?>', '', str)


def pretokenize_url(url):
    """A hacky procedure to "pretokenize" URLs.

    :param  url:  an input URL
    :return a URL with prefixes (see below) removed and some characters replaced with ' '
    """
    remove_pref = ['http://', 'https://', 'www.']
    url = urllib.parse.unquote(url)
    changed = True
    while changed:
        changed = False
        for p in remove_pref:
            assert len(p) > 0
            if url.startswith(p):
                changed = True
                url = url[len(p):]
                break

    return re.sub(r'[.,:!\?/"+\-\'=_{}()|]', " ", url)


def proc_yahoo_answers_record(rec_str):
    """A procedure to parse a single Yahoo-answers format entry.

    :param rec_str: Answer content including enclosing tags <document>...</document>
    :return:  parsed data as YahooAnswerRecParsed entry
    """
    doc = BeautifulSoup(rec_str, 'lxml')

    doc_root = doc.find('document')
    if doc_root is None:
        raise Exception('Invalid format, missing <document> tag')

    uri = doc_root.find('uri')
    if uri is None:
        raise Exception('Invalid format, missing <uri> tag')
    uri = uri.text

    subject = doc_root.find('subject')
    if subject is None:
        raise Exception('Invalid format, missing <subject> tag')
    subject = remove_tags(subject.text)

    content = doc_root.find('content')
    content = '' if content is None else remove_tags(content.text)  # can probably be missing

    best_answer = doc_root.find('bestanswer')
    best_answer = '' if best_answer is None else best_answer.text  # is missing occaisionally

    best_answer_id = -1

    answ_list = []
    answers = doc_root.find('nbestanswers')
    if answers is not None:
        for answ in answers.find_all('answer_item'):
            answ_text = answ.text
            if answ_text == best_answer:
                best_answer_id = len(answ_list)
            answ_list.append(remove_tags(answ_text))

    return YahooAnswerRecParsed(uri=uri, subject=subject.strip(), content=content.strip(),
                                best_answer_id=best_answer_id, answer_list=answ_list)


def jsonl_gen(file_name):
    """A generator that produces parsed doc/query entries one by one.
      :param file_name: an input file name
    """

    with FileWrapper(file_name) as f:
        for i, line in enumerate(f):
            ln = i + 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except:
                raise Exception('Error parsing JSON in line: %d' % ln)

            if not DOCID_FIELD in data:
                raise Exception('Missing %s field in JSON in line: %d' % (DOCID_FIELD, ln))

            yield data


def multi_file_linegen(dir_name, pattern):
    """A generator that reads all files from a given directory matching the pattern
       and yields their contents line by line.

    :param dir_name:   a source directory name
    :param pattern:    a pattern should match fully (we use fullmatch)
    """

    for fn in os.listdir(dir_name):
        if re.fullmatch(pattern, fn):
            full_path = os.path.join(dir_name, fn)
            print('Processing: ' + full_path)
            with FileWrapper(full_path) as inp:
                for line in inp:
                    yield line


def read_queries(file_name):
    """Read queries from a JSONL file and checks the document ID is set.

    :param file_name: an input file name
    :return: an array where each entry is a parsed query JSON.
    """
    return list(jsonl_gen(file_name))


def write_queries(query_list, file_name):
    """Write queries to a JSONL file.

    :param query_list: an array of parsed JSON query entries
    :param file_name: an output file
    """
    with open(file_name, 'w') as f:
        for e in query_list:
            f.write(json.dumps(e))
            f.write('\n')


def unique(arr):
    return list(set(arr))


def get_retokenized(tokenizer, text):
    """Obtain a space separated re-tokenized text.
    :param tokenizer:  a tokenizer that has the function
                       tokenize that returns an array of tokens.
    :param text:       a text to re-tokenize.
    """
    return ' '.join(tokenizer.tokenize(text))


def add_retokenized_field(data_entry,
                        src_field,
                        dst_field,
                        tokenizer):
    """
    Create a re-tokenized field from an existing one.

    :param data_entry:   a dictionary of entries (keys are field names, values are text items)
    :param src_field:    a source field
    :param dst_field:    a target field
    :param tokenizer:    a tokenizer to use, if None, nothing is done
    """
    if tokenizer is not None:
        dst = ''
        if src_field in data_entry:
            dst = get_retokenized(tokenizer, data_entry[src_field])

        data_entry[dst_field] = dst


def read_doc_ids_from_forward_file_header(fwd_file_name):
    """Read document IDs from the textual header
       of a forward index. Some basic integrity checkes are done.

       :param   fwd_file_name: input file name
       :return  a set of document IDs.
    """
    f = open(fwd_file_name)
    lines = [s.strip() for s in f]
    assert len(lines) > 3, f"File {fwd_file_name} is too short"
    f.close()
    doc_qty, _ = lines[1].split()
    doc_qty = int(doc_qty)

    assert len(lines) > doc_qty + 4, f"File {fwd_file_name} is too short: length isn't consistent with the header info"
    assert lines[2] == "", f"The second line in {fwd_file_name} isn't empty as expected!"
    assert lines[-1] == "", f"The last line in {fwd_file_name} isn't empty as expected!"
    k = 3
    while k < len(lines) and lines[k] != '':
        k = k + 1
    assert lines[k] == ''  # We check that the last line is empty, we must find the empty line!
    k = k + 1

    assert k + doc_qty + 1 == len(lines)
    res = lines[k:len(lines) - 1]
    assert len(res) == doc_qty

    return set(res)


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


def dense_vect_pack_mask(dim):
    """Generate a packing masking for an integer + floating point array (little endian layout).

    :param dim:     dimensionality
    :return:        packing mask
    """
    # The endianness mark applies to the whole string (Python docs are unclear about this, but we checked this out
    # using real examples)
    return f'{ENDIANNES_TYPE}I' + ''.join(['f'] * dim)


def pack_dense_batch(data):
    """Pack a bach of dense vectors.

    :param data: a PyTorch tensor or a numpy 2d array
    :return: a list of byte arrays where each array represents one dense vector
    """
    if type(data) == torch.Tensor:
        data = data.cpu()
    bqty, dim = data.shape

    mask = dense_vect_pack_mask(dim)

    return [struct.pack(mask, PACKED_TYPE_DENSE, *list(data[i])) for i in range(bqty)]


def write_json_to_bin(data_elem, out_file):
    """Convert a json entry to a BSON format and write it to a file.

    :param data_elem: an input JSON dictionary.
    :param out_file: an output file (should be binary writable)
    """
    assert type(data_elem) == dict

    bson_data = bson.dumps(data_elem)
    out_file.write(struct.pack(f'{ENDIANNES_TYPE}I', len(bson_data)))
    out_file.write(bson_data)


def read_json_from_bin(inp_file):
    """Read a BSON entry (previously written by write_json_to_bin) from a
       file.

    :param input file  (should be binary read-only)
    :param a parased JSON entry or None when we reach the end of file.
    """
    data_len_packed = inp_file.read(4)
    if len(data_len_packed) == 0:
        return None
    assert len(data_len_packed) == 4, "possibly truncated file, not enough input data to read the entry length"
    data_len = struct.unpack(f'{ENDIANNES_TYPE}I', data_len_packed)[0]
    data_packed = inp_file.read(data_len)
    assert len(data_packed) == data_len, "possibly truncated file, not enough input data to read BSON entry"
    return bson.loads(data_packed)


def is_json_query_file(file_name):
    """Checks if the input is a JSONL query file (using name only)."""
    return os.path.split(file_name)[1] == QUESTION_FILE_JSON
