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

from flexneuart.config import DOCID_FIELD

def create_temp_file():
    """"Create a temporary file
    :return temporary file name
    """
    f, file_name = tempfile.mkstemp()
    os.close(f)
    return file_name


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
            self._is_compr = True
        elif file_name.endswith('.bz2'):
            self._file = bz2.open(file_name, flags)
            self._is_compr = True
        else:
            self._file = open(file_name, flags)
            self._is_compr = False

    def write(self, s):
        if self._is_compr:
            self._file.write(s.encode())
        else:
            self._file.write(s)

    def read(self, qty):
        if self._is_compr:
            return self._file.read(qty).decode()
        else:
            return self._file.read(qty)

    def close(self):
        self._file.close()

    def __exit__(self, type, value, tb):
        self._file.close()

    def __iter__(self):
        for line in self._file:
            yield line.decode() if self._is_compr else line


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

