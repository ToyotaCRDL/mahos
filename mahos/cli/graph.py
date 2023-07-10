#!/usr/bin/env python3

import typing as T
import argparse
import enum

import networkx as nx
import matplotlib.pyplot as plt

from ..node.node import join_name, local_conf, hosts, host_nodes, threaded_nodes
from ..util.conv import invert_mapping
from .util import check_load_gconf


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog="mahos graph", description="Visualize a config file as graph."
    )
    parser.add_argument(
        "-c", "--conf", type=str, default="conf.toml", help="config file name (default: conf.toml)"
    )
    parser.add_argument("-o", "--output", type=str, help="output file name")
    parser.add_argument(
        "-x",
        "--networkx",
        action="store_true",
        help="use networkx for rendering instead of pygraphviz",
    )
    parser.add_argument(
        "-n", "--no-show", dest="show", action="store_false", help="don't show graph in window"
    )
    parser.add_argument(
        "-C", "--class-name", action="store_true", help="include module.class in node label"
    )
    parser.add_argument(
        "-e",
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="full name (hostname::nodename) to exclude",
    )

    args = parser.parse_args(args)

    return args


class LabelStyle(enum.Enum):
    Node = 0
    HostNode = 1
    NodeClass = 2
    HostNodeClass = 3


def style(use_nx: bool, class_name: bool):
    if use_nx:
        if class_name:
            return LabelStyle.HostNodeClass
        else:
            return LabelStyle.HostNode
    else:
        if class_name:
            return LabelStyle.NodeClass
        else:
            return LabelStyle.Node


def add_edge(G, joined_name, target, exclude):
    for key, value in target.items():
        if isinstance(value, str):
            if value in exclude:
                continue
            G.add_edge(joined_name, join_name(value), label=key)
        elif isinstance(value, (list, tuple)):
            for n in value:
                if n in exclude:
                    continue
                G.add_edge(joined_name, join_name(n), label=key)
        elif isinstance(value, dict):
            for n, target_key in invert_mapping(value).items():
                if n in exclude:
                    continue
                G.add_edge(
                    joined_name, join_name(n), label="{}({})".format(key, ", ".join(target_key))
                )


def make_graph(gconf: dict, style: LabelStyle, exclude: T.List[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    for host, node in host_nodes(gconf):
        joined_name = join_name((host, node))
        if joined_name in exclude:
            continue
        lconf = local_conf(gconf, joined_name)
        m = lconf.get("module", "<module unknown>")
        c = lconf.get("class", "<class unknown>")
        if style == LabelStyle.Node:
            label = node
        elif style == LabelStyle.HostNode:
            label = joined_name
        elif style == LabelStyle.NodeClass:
            label = f"{node}\n({m}.{c})"
        elif style == LabelStyle.HostNodeClass:
            label = f"{joined_name}\n({m}.{c})"
        G.add_node(joined_name, label=label)

    for host, node in host_nodes(gconf):
        joined_name = join_name((host, node))
        lconf = local_conf(gconf, joined_name)
        if "target" in lconf:
            add_edge(G, joined_name, lconf["target"], exclude)

    return G


def draw_nx(G: nx.DiGraph, fn: str, show: bool):
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, labels={n: l for n, l in G.nodes(data="label")})
    nx.draw_networkx_edges(G, pos)
    edge_labels = {(u, v): l for u, v, l in G.edges(data="label")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if fn:
        plt.savefig(fn)
    if show:
        plt.show()


def draw_pgv(gconf: dict, G: nx.DiGraph, fn: str, show: bool):
    AG = nx.nx_agraph.to_agraph(G)
    for host in hosts(gconf):
        names = [join_name((host, node)) for node in gconf[host]]
        H = AG.add_subgraph(nbunch=names, name="cluster_" + host, label=host)
        for tname, nodes in threaded_nodes(gconf, host).items():
            names = [join_name((host, node)) for node in nodes]
            H.add_subgraph(nbunch=names, name=f"cluster_{host}_{tname}", label=f"thread({tname})")

    AG.layout(prog="dot")  # AG.layout(prog="neato", args="-Goverlap=prism")
    if fn:
        AG.draw(fn)

    if not show:
        return
    try:
        AG.draw(format="xlib")
    except OSError:
        pass


def main(args=None):
    args = parse_args(args)
    gconf = check_load_gconf(args.conf)
    G = make_graph(gconf, style(args.networkx, args.class_name), args.exclude)
    if args.networkx:
        draw_nx(G, args.output, args.show)
    else:
        draw_pgv(gconf, G, args.output, args.show)
