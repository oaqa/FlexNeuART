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
import subprocess as sp

seen_core = set()
output = sp.check_output(["cat", "/proc/cpuinfo"]).decode("utf-8").replace('\t', ' ').split('\n')
start = 0
output.append('')  # There would always a trailing empty line
num_cores = 0
while (start < len(output)):
    line = output[start].strip()
    if line != '':
        end = start + 1
        while (end < len(output)):
            if output[end].strip() == '': break
            end = end + 1
        # print start, end
        id = 0
        non_repeat = False
        for i in range(start, end):
            line = output[i].strip()
            fields = line.split(':')
            if (len(fields) == 2 and fields[0].strip() == 'physical id'):
                id = fields[1].strip()
                if not id in seen_core:
                    seen_core.add(id)
                    non_repeat = True
                    break
        if non_repeat:
            for i in range(start, end):
                line = output[i].strip()
                fields = line.split(':')
                if (len(fields) == 2 and fields[0].strip() == 'siblings'):
                    num_cores += int(fields[1].strip())
        start = end + 1
    start = start + 1

print(num_cores)
