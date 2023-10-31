#!/usr/bin/env python3

"""
Graph algorithm utilities.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import networkx as nx


def sort_dependency(dep_dict: dict) -> list:
    """use topological sort to resolve dependency.

    :param dep_dict: a dict to represent nodes (any hashable objects) and their dependencies.
        {"a": ("b", "c"), "b": ("c",)} reads as "'a' depends on 'b' & 'c'",
        and "'b' depends on 'c'".
    :type dep_dict: dict[Hashable, tuple[Hashable]]

    :returns: list[Hashable] topological-sorted nodes (dep_dict's keys and value elements).
        ["c", "b", "a"] for the example above.

    """

    G = nx.DiGraph()
    G.add_nodes_from(dep_dict.keys())
    for v, deps in dep_dict.items():
        for u in deps:
            #  new node may be automatically created here if u is not in dep_dict.keys().
            G.add_edge(u, v)
    return list(nx.topological_sort(G))
