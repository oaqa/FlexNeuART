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
import struct
import torch
import bson


# '<' stands for Little-endian
ENDIANNES_TYPE = '<'
PACKED_TYPE_DENSE = 0
PACKED_TYPE_SPARSE = 1


def dense_vect_pack_mask(dim):
    """Generate a packing masking for an integer + floating point array (little endian layout).

    :param dim:     dimensionality
    :return:        packing mask
    """
    # The endianness mark applies to the whole string (Python docs are unclear about this, but we checked this out
    # using real examples)
    return f'{ENDIANNES_TYPE}I' + ''.join(['f'] * dim)


def pack_dense_batch(data):
    """Pack a bach of dense vectors.

    :param data: a PyTorch tensor or a numpy 2d array
    :return: a list of byte arrays where each array represents one dense vector
    """
    if type(data) == torch.Tensor:
        data = data.cpu()
    bqty, dim = data.shape

    mask = dense_vect_pack_mask(dim)

    return [struct.pack(mask, PACKED_TYPE_DENSE, *list(data[i])) for i in range(bqty)]

def pack_sparse_vect(vect):
    """Pack the sparse vector that comes in as a

    :param vect: tuple or list containing alternating integer ids and float values.
    :return:  a packed byte sequence
    """
    dim = len(vect) // 2
    assert dim * 2 == len(vect)
    mask = f'{ENDIANNES_TYPE}II' + ''.join(['If'] * dim)

    return struct.pack(mask, PACKED_TYPE_SPARSE, dim, *(vect))


def unpack_int(b):
    """Just unpack an integer value"""
    assert len(b) == 4
    return struct.unpack(f'{ENDIANNES_TYPE}I', b)[0]


def read_and_unpack_int(f):
    return unpack_int(f.read(4))


def read_ascii_str(f):
    id_len = read_and_unpack_int(f)
    b = struct.unpack(''.join(['s'] * id_len), f.read(id_len))
    return ''.join(c.decode() for c in b)


def write_json_to_bin(data_elem, out_file):
    """Convert a json entry to a BSON format and write it to a file.

    :param data_elem: an input JSON dictionary.
    :param out_file: an output file (should be binary writable)
    """
    assert type(data_elem) == dict

    bson_data = bson.dumps(data_elem)
    out_file.write(struct.pack(f'{ENDIANNES_TYPE}I', len(bson_data)))
    out_file.write(bson_data)


def read_json_from_bin(inp_file):
    """Read a BSON entry (previously written by write_json_to_bin) from a
       file.

    :param input file  (should be binary read-only)
    :param a parased JSON entry or None when we reach the end of file.
    """
    data_len_packed = inp_file.read(4)
    rqty = len(data_len_packed)
    if rqty == 0:
        return None
    assert rqty == 4, f"possibly corrputed/truncated file, asked to read 4 bytes, but read {rqty}"
    data_len = struct.unpack(f'{ENDIANNES_TYPE}I', data_len_packed)[0]
    data_packed = inp_file.read(data_len)
    assert len(data_packed) == data_len, "possibly truncated file, not enough input data to read BSON entry"
    return bson.loads(data_packed)
