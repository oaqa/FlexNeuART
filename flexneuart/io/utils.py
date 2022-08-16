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
import json
import re
import os
import tempfile

from flexneuart.config import DOCID_FIELD, DEFAULT_ENCODING

def create_temp_file():
    """"Create a temporary file
    :return temporary file name
    """
    f, file_name = tempfile.mkstemp()
    os.close(f)
    return file_name


def open_with_default_enc(file, mode='r', buffering=-1, encoding=DEFAULT_ENCODING,
                          errors=None, newline=None, closefd=True, opener=None):
    """A simple wrapper that opens files using the default encoding.
    :param file:   a file name
    :param mode:   flags (r, w, b, etc .._
    :param buffering: is an optional integer used to set the buffering policy.
    :param newline: controls how universal newlines mode works (it only applies to text mode).
                    It can be None, '', '\n', '\r', and '\r\n'
    :param encoding: encoding
    :return:
    """
    # In the binary mode encoding cannot be specify
    if 'b' in mode:
        encoding = None
    return open(file=file, mode=mode, buffering=buffering, encoding=encoding,
                errors=errors, newline=newline, closefd=closefd, opener=opener)


class FileWrapper:

    def __enter__(self):
        return self

    def __init__(self, file_name, flags='r', encoding=DEFAULT_ENCODING, decode_errors='strict'):
        """Constructor, which opens a regular or gzipped-file

          :param  file_name a name of the file, it has a '.gz' or '.bz2' extension, we open a compressed stream.
          :param  flags    open flags such as 'r' or 'w'
          :param  encoding file encoding: will be ignored for binary files!
          :param  how to treat the decoding errors, this value is passed as the parameter 'errors' to the function decode
        """
        self.decode_errors=decode_errors
        # In the binary mode encoding cannot be specify
        dir_name = os.path.dirname(file_name)
        # create a directory only if the file is in the write mode
        if dir_name and 'w' in list(flags):
            os.makedirs(dir_name, exist_ok=True)
        if file_name.endswith('.gz'):
            self._file = gzip.open(file_name, flags, encoding=None)
            self._is_compr = True
        elif file_name.endswith('.bz2'):
            self._file = bz2.open(file_name, flags, encoding=None)
            self._is_compr = True
        else:
            if 'b' in flags:
                encoding = None
            self._file = open(file_name, flags, encoding=encoding)
            self._is_compr = False

    def write(self, s):
        if self._is_compr:
            self._file.write(s.encode())
        else:
            self._file.write(s)

    def read(self, qty=-1):
        if self._is_compr:
            return self._file.read(qty).decode(encoding=DEFAULT_ENCODING, errors=self.decode_errors)
        else:
            return self._file.read(qty)

    def close(self):
        self._file.close()

    def __exit__(self, type, value, tb):
        self._file.close()

    def __iter__(self):
        for line in self._file:
            yield line.decode(encoding=DEFAULT_ENCODING, errors=self.decode_errors) if self._is_compr else line


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

