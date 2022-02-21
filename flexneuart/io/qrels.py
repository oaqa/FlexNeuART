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
from typing import Dict, List
from tqdm import tqdm
from flexneuart.io import FileWrapper

QrelEntry = collections.namedtuple('QrelEntry',
                                   'query_id doc_id rel_grade')
#
# Important note: currently we have multiple ways to represent QREL dictionaries.
#


def gen_qrel_str(query_id : str, doc_id: str, rel_grade : int) -> str:
    """Produces a string representing one QREL entry

    :param query_id:   question/query ID
    :param doc_id:     relevanet document/answer ID
    :param rel_grade:  relevance grade

    :return: a string representing one QREL entry
    """
    return f'{query_id} 0 {doc_id} {rel_grade}'


def qrel_entry2_str(qrel_entry : QrelEntry) -> str:
    """Convert a parsed QREL entry to string.

    :param qrel_entry: input of the type QrelEntry
    :return:  string representation.
    """
    return gen_qrel_str(qrel_entry.query_id, qrel_entry.doc_id, qrel_entry.rel_grade)


def parse_qrel_entry(line) -> QrelEntry:
    """Parse one QREL entry
    :param line  a single line with a QREL entry.
            Relevance graded is expected to be integer.

    :return a parsed QrelEntry entry.
    """

    line = line.strip()
    parts = line.split()
    if len(parts) != 4:
        raise Exception('QREL entry format error, expecting just 4 white-space separted field in the entry: ' + line)

    return QrelEntry(query_id=parts[0], doc_id=parts[2], rel_grade=int(parts[3]))


def read_qrels(file_name : str) -> List[QrelEntry]:
    """Read and parse QRELs.

    :param file_name: input file name
    :return: an array of parsed QREL entries
    """
    ln = 0
    res = []

    with FileWrapper(file_name) as f:
        for line in tqdm(f, desc='loading qrels (by line)', leave=False):
            ln += 1
            line = line.strip()
            if not line:
                continue
            try:
                e = parse_qrel_entry(line)
                res.append(e)
            except:
                raise Exception('Error parsing QRELs in line: %d' % ln)

    return res


def write_qrels(qrel_list : List[QrelEntry], file_name : str):
    """Write a list of QRELs to a file.

    :param qrel_list:  a list of parsed QRELs
    :param file_name:  an output file name
    """
    with FileWrapper(file_name, 'w') as f:
        for e in qrel_list:
            f.write(qrel_entry2_str(e))
            f.write('\n')


def write_qrels_dict(qrel_dict : Dict[str, Dict[str, int]],
                     file_name : str):
    """Write a QREL dictionary stored in the format produced by the
       function read_qrels_dict.

    :param qrel_dict:  dictionary of dictionaries (see read_qrels_dict).
    :param file_name:  output file name
    """
    with FileWrapper(file_name, 'w') as f:
        for qid, doc_rel_dict in qrel_dict.items():
            for did, grade in doc_rel_dict.items():
                f.write(gen_qrel_str(query_id=qid, doc_id=did, rel_grade=grade))
                f.write('\n')


def read_qrels_dict(file_name : str) -> Dict[str, Dict[str, int]]:
    """Read QRELs in the form of a dictionary where keys are query IDs.

    :param file_name: QREL file name
    :return: a dictionary of dictionaries
    """
    result = {}
    for e in read_qrels(file_name):
        result.setdefault(e.query_id, {})[e.doc_id] = int(e.rel_grade)
    return result


def add_qrel_entry(qrel_dict, qid, did, grade):
    """Add a QREL entry to a QREL dictionary. Repeated entries are ignored. However if they
       have a different grade, an exception is thrown.

    :param qrel_dict:  a QREL dictionary
    :param qid:        query id
    :param did:        document id
    :param grade:      QREL grade
    """
    qrel_key = (qid, did)
    if qrel_key in qrel_dict:
        prev_grade = qrel_dict[qrel_key].rel_grade
        if prev_grade != grade:
            raise Exception(f'Repeating inconsistent QREL values for query {qid} and document {did}, got grades: ',
                            grade, prev_grade)
    qrel_dict[qrel_key] = QrelEntry(query_id=qid, doc_id=did, rel_grade=grade)
