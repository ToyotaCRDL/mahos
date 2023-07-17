#!/usr/bin/env python3

"""
IO functions for Data Timing Generator.

.. This file is a part of MAHOS project.

"""

import struct

import numpy as np

from .dtg_recids import is_interior, RECIDS, LEAF_TYPES


def read_tree(data, max_bytes=10000000):
    """read 'max_bytes' from a byte string 'data' and build up a tree."""

    def read(size, data):
        tree = []

        while size > 0 and len(data) > 0:
            recid = struct.unpack("<H", data[:2])[0]

            data = data[2:]
            size -= 2

            if is_interior(recid):
                lint = struct.unpack("<I", data[:4])[0]
                data, subtree = read(lint, data[4:])
                hr_recid = [key for key in RECIDS.keys() if RECIDS[key] == recid]
                tree.append([hr_recid[0], lint, subtree])
                size -= lint + 4
            else:  # leaf
                hr_recid = [key for key in RECIDS.keys() if RECIDS[key] == recid]
                d, lleaf, data = read_leaf(hr_recid[0], data)

                tree.append([hr_recid[0], lleaf - 4, d])
                size -= lleaf

        return data, tree

    return read(max_bytes, data)[1]


def read_leaf(recid_hr, data):
    """read data from a leaf node.

    data must start with an entry of type "length, data" (RECID is dropped before)

    """

    t = LEAF_TYPES[recid_hr]
    l = struct.unpack("<I", data[:4])[0]

    nbytes, label = {
        np.bytes_: (1, "s"),
        np.int8: (1, "b"),
        np.int16: (2, "h"),
        np.int32: (4, "i"),
        np.float64: (8, "d"),
    }[t]
    # np.int16().nbytes returns 2. but this method doesn't work for np.bytes_.

    fmt = "<%d%s" % (int(l / nbytes), label)

    d = np.array(struct.unpack(fmt, data[4 : 4 + l]), dtype=t)

    return d, l + 4, data[4 + l :]


def dump_tree(tree, f):
    """dump a DTG setup tree 'tree' to the file 'f'."""

    for node in tree:
        recid_hr, length, data = node
        recid = RECIDS[recid_hr]
        if is_interior(recid):
            f.write(struct.pack("<H", recid))
            f.write(struct.pack("<I", length))
            dump_tree(data, f)
        else:  # leaf node
            f.write(struct.pack("<H", recid))
            f.write(struct.pack("<I", length))
            f.write(data.tobytes())


def recalculate_space(tree):
    """given a DTG setup tree 'tree', update the length info of each node."""

    total = 0
    for i, node in enumerate(tree):
        recid_hr, length, data = node
        recid = RECIDS[recid_hr]

        if is_interior(recid):
            node[1], node[2] = recalculate_space(data)
            tree[i] = node
        else:  # leaf node
            node[1] = len(data.tobytes())
            tree[i] = node
        # add length of the treated node to total length.
        # 6 is for recid (2 bytes) and length (4 bytes).
        total += node[1] + 6

    return total, tree


def get_leaf(tree, leaf_id: str):
    """search a DTG tree 'tree' for a leaf with recid 'leaf_id' and return it."""

    for node in tree:
        recid_hr, length, data = node
        recid = RECIDS[recid_hr]
        if is_interior(recid):
            res = get_leaf(data, leaf_id)
            if len(res):
                return res

        elif recid_hr == leaf_id:
            return [length, data]

    return []


def replace_leaf(leaves, leaf_id: str, newlen, newval):
    """replace the leaf.

    Given list of 'leaves', replace all leaves of recid 'leaf_id' with 'newlen', 'newval'.

    """

    for i, leaf in enumerate(leaves):
        if leaf[0] == leaf_id:
            leaves[i][1] = newlen
            leaves[i][2] = newval
    return leaves


def find_record(tree, name: str):
    """search 'tree' for given recid 'name'."""

    res = []

    for node in tree:
        recid_hr, length, data = node
        recid = RECIDS[recid_hr]

        if recid_hr == name:
            res.append(node)
        elif is_interior(recid):
            res.extend(find_record(data, name))

    return res
