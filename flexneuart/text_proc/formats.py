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
import collections

from bs4 import BeautifulSoup
from flexneuart.io import FileWrapper
from flexneuart.text_proc.clean import remove_tags

"""
    Tools to work with data in non-standard (not JSONL and not binary JSONL) formats.
"""

YahooAnswerRecParsed = collections.namedtuple('YahooAnswerRecParsed',
                                              'uri subject content best_answer_id answer_list')


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

