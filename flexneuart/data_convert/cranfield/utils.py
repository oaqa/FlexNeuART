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
from flexneuart.io import FileWrapper
from flexneuart.config import TITLE_FIELD_NAME, DOCID_FIELD, TEXT_RAW_FIELD_NAME

AUTHOR_FIELD_NAME = 'author'
BODY_FIED_NAME = 'body'
VENUE_FIELD_NAME = 'venue'

FIELD_MAP = {
    '.T' : TITLE_FIELD_NAME,
    '.W' : BODY_FIED_NAME,
    '.B' : VENUE_FIELD_NAME,
    '.A' : AUTHOR_FIELD_NAME
}


def read_cranfield_data(file):
    res = []
    curr_entry = None
    curr_text = None
    all_text = None
    prev_field = None
    with FileWrapper(file) as f:
        for line in f:
            if line.startswith('.I '):
                if curr_entry:
                    assert curr_text is not None
                    curr_entry[FIELD_MAP[prev_field]] = curr_text.strip()
                    assert all_text is not None
                    curr_entry[TEXT_RAW_FIELD_NAME] = all_text
                    res.append(curr_entry)
                curr_entry = {DOCID_FIELD : line[3:].strip()}
                curr_text = ''
                all_text = ''
                prev_field = None
            else:
                all_text += line
                line_stripped = line.strip()
                if line_stripped in FIELD_MAP:
                    if  prev_field is not None:
                        assert curr_text is not None
                        curr_entry[FIELD_MAP[prev_field]] = curr_text.strip()
                    prev_field = line_stripped
                    curr_text = ''
                else:
                    curr_text += line

    if curr_entry:
        assert curr_text is not None
        curr_entry[FIELD_MAP[prev_field]] = curr_text.strip()
        assert all_text is not None
        curr_entry[TEXT_RAW_FIELD_NAME] = all_text
        res.append(curr_entry)

    return res