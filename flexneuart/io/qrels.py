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
from tqdm import tqdm
from flexneuart.io import open_with_default_enc

QrelEntry = collections.namedtuple('QrelEntry',
                                   'query_id doc_id rel_grade')


def gen_qrel_str(query_id, doc_id, rel_grade):
    """Produces a string representing one QREL entry

    :param query_id:   question/query ID
    :param doc_id:     relevanet document/answer ID
    :param rel_grade:  relevance grade

    :return: a string representing one QREL entry
    """
    return f'{query_id} 0 {doc_id} {rel_grade}'


def qrel_entry2_str(qrel_entry):
    """Convert a parsed QREL entry to string.

    :param qrel_entry: input of the type QrelEntry
    :return:  string representation.
    """
    return gen_qrel_str(qrel_entry.query_id, qrel_entry.doc_id, qrel_entry.rel_grade)


def parse_qrel_entry(line):
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


def read_qrels(file_name):
    """Read and parse QRELs.

    :param file_name: input file name
    :return: an array of parsed QREL entries
    """
    ln = 0
    res = []

    with open_with_default_enc(file_name) as f:
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


def write_qrels(qrel_list, file_name):
    """Write a list of QRELs to a file.

    :param qrel_list:  a list of parsed QRELs
    :param file_name:  an output file name
    """
    with open_with_default_enc(file_name, 'w') as f:
        for e in qrel_list:
            f.write(qrel_entry2_str(e))
            f.write('\n')


def write_qrel_dict(qrel_dict, file_name):
    """Write a QREL dictionary where entries are added using, e.g., add_qrel_entry

    :param qrel_dict:  dictionary of QRELs.
    :param file_name:  output file name
    """
    qrel_list = [qrel_entry for qrel_key, qrel_entry in qrel_dict.items()]
    write_qrels(qrel_list, file_name)


def read_qrels_dict(file_name):
    """Read QRELs in the form of a dictionary where keys are query IDs.

    :param file_name: QREL file name
    :return: a dictionary of dictionaries
    """
    result = {}
    for e in read_qrels(file_name):
        result.setdefault(e.query_id, {})[e.doc_id] = int(e.rel_grade)
    return result


def add_qrel_entry(qrel_dict, qid, did, grade):
    """Add a QREL entry to the QREL dictionary. Repeated entries are ignored. However if they
       have a different grade, an exception is throw.

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
