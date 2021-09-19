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
import json


def get_passage_id(ctx_entry):
    """Retrieve a passage ID from the positive or negative context entry:
       an element in an array with a key positive_ctxs, negative_ctxs, or
       hard_negative_ctxs. The problem is that some entries encode them
       using the key psg_id and some use the key passage_id.
    """
    for psg_key in ['psg_id', 'passage_id']:
        if psg_key in ctx_entry:
            return ctx_entry[psg_key]

    raise Exception('No passage keys in the entry: ' + json.dumps(ctx_entry))


def dpr_json_reader(file_to_read):
    """A simple streaming json reader. It assumes the file is well formated,
       which is the case of Facebook DPR data, but it cannot be used as a generic
       JSON stream reader, where blocks start/end at arbitrary positions
       (unlike Facebook DPR data).

    :param file_to_read:

    :return: yields an unparsed textual representation of all entries for one question
    """
    current_depth = 0
    buffer = []
    for i, line in enumerate(map(lambda line: line.strip(), file_to_read)):
        if current_depth == 0 and line in ("[", "]"):
            continue

        if line == "{":
            current_depth += 1

        if line == "}" or line == "},":
            current_depth -= 1
            if current_depth == 0:
                buffer.append("}")
                yield "\n".join(buffer)
                buffer = []
            else:
                buffer.append(line)
        else:
            buffer.append(line)
