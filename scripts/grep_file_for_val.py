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
# grep*ForVal functions extract values of a given key
# from a string/file in a stupid one-entry-per-line format,
# where each line has the format (there can be spaces around :)
# key:value
#
import sys

from flexneuart.io import open_with_default_enc

file_name=sys.argv[1]
metr_name=sys.argv[2]
ignore_case=False
if len(sys.argv) >= 4:
    ignore_case=str(sys.argv[3]) == '1'

if ignore_case:
    metr_name = metr_name.lower()

with open_with_default_enc(file_name) as f:
    for line in f:
        line = line.strip()
        sepi = line.find(':')
        if sepi >= 0:
            metr = line[0:sepi].strip()
            if ignore_case:
                metr = metr.lower()
            if metr == metr_name:
                print(line[sepi+1:].strip())
                sys.exit(0)

print('')
      
