#!/usr/bin/env python3

import argparse

import IPython

from ..node.node import join_name
from ..node.client import get_client
from .util import init_gconf_host_nodes


class Clients(object):
    """Simple object for storing attributes.

    Taken from Namespace from argparse.

    """

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append("%s=%r" % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append("**%s" % repr(star_args))
        return "%s(%s)" % (type_name, ", ".join(arg_strings))

    def _get_kwargs(self):
        return sorted(self.__dict__.items())

    def _get_args(self):
        return []


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog="mahos shell", description="Start shell with a client for mahos node."
    )
    parser.add_argument(
        "-c", "--conf", type=str, default="conf.toml", help="config file name (default: conf.toml)"
    )
    parser.add_argument("-H", "--host", type=str, default="", help="host name")
    parser.add_argument(
        "-C",
        "--colors",
        type=str,
        default="neutral",
        help="IPython shell color. One of ('nocolor', 'neutral', 'linux', 'lightbg')"
        + " (default: neutral)",
    )
    parser.add_argument(
        "nodes",
        type=str,
        nargs="+",
        help="list of node name or full name ({})".format(join_name(("host", "node"))),
    )

    args = parser.parse_args(args)

    return args


def main(args=None):
    """Start IPython shell with NodeClient instances in a variable `cli`.

    Refs:
    - IPython.embed()
      https://ipython.readthedocs.io/en/stable/api/generated/IPython.terminal.embed.html

    - IPython Terminal Color
      https://ipython.readthedocs.io/en/stable/config/details.html#terminal-colors

    """

    args = parse_args(args)
    gconf, host, nodes = init_gconf_host_nodes(args.conf, args.host, args.nodes)
    joined_names = [join_name((host, n)) for n in nodes]

    print("Starting shell with clients for {} (config file: {}).".format(joined_names, args.conf))

    _clients = []
    for n in nodes:
        _clients.append(get_client(n, conf_path=args.conf, host=host))
    if len(nodes) == 1:
        cli = _clients[0]
    else:
        cli = {n: cli for n, cli in zip(nodes, _clients)}
        if all([n.isidentifier() for n in nodes]):
            cli = Clients(**cli)

    header = 'variable "cli" holds the client(s)'
    IPython.embed(header=header, colors=args.colors)

    print("Closing clients.")
    for cli in _clients:
        cli.close()
