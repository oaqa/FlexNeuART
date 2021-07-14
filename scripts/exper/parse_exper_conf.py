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
# Carry out a key-value data conversion
# of a specified entry from an array of JSON entries 
# to a stupid one-entry-per-line format, 
# where each line has the format:
#
# key:value
#
# Note the following:
# 1. The script expects the input JSON to contain an array of dictionaries
# 2. If the user specifies a dictionary id outside the range, we generate the out-of-range string #OOR
# 3. newlines in JSON keys and values will be replaced with spaces
#
import json
import sys
import re

ERROR_STR = '#ERR'
END_STR = '#END'


# Replace \n and \r characters with spaces
# This is a copy-pasted function, but I want to keep
# this simple script to be really light-weight without dependencies
def replace_chars_nl(s):
    return re.sub(r'[\n\r]', ' ', s)


if len(sys.argv) != 4:
    sys.stderr.write('Usage <input JSON> <entry ID> <out file>\n')
    print(ERROR_STR)
    sys.exit(1)

out_file = sys.argv[3]

inp_data = json.load(open(sys.argv[1]))
if type(inp_data) != type([]):
    sys.stderr.write('Wrong root-level data type, expecting a list')
    print(ERROR_STR)
    sys.exit(1)

entry_id = int(sys.argv[2])

if entry_id < 0 or entry_id >= len(inp_data):
    print(END_STR)
    sys.exit(0)
else:

    with open(out_file, 'w') as of:
        for key, value in inp_data[entry_id].items():
            key = replace_chars_nl(str(key))
            if type(value) == bool:
                value = int(value)

            value = replace_chars_nl(str(value))
            of.write(f'{key}:{value}\n')
