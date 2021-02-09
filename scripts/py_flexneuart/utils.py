"""Misc FlexNeuART utils."""
from jnius import autoclass

JHashMap = autoclass('java.util.HashMap')


def dict_to_hash_map(dict_obj):
    """Convert a Python dictionary to a Java HashMap object. Caution:
       values in the dictionary need to be either simple types, or
       proper Java object references created through jnius autoclass.

    :param dict_obj:   a Python dictionary whose values and keys are either simple types
                       or Java objects creates via jnius autoclass
    :return: a Java HashMap
    """

    res = JHashMap()
    for k, v in dict_obj.items():
        res.put(k, v)

    return res
