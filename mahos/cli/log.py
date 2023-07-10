#!/usr/bin/env python3

import time
import argparse
from functools import partial

from ..node.node import join_name
from ..node.log_broker import LogClient, parse_log, format_log, format_log_term_color, should_show
from .util import init_gconf_host_node


def parse_args(args):
    parser = argparse.ArgumentParser(prog="mahos log", description="Print logs from LogBroker.")
    parser.add_argument(
        "-c", "--conf", type=str, default="conf.toml", help="config file name (default: conf.toml)"
    )
    parser.add_argument("-H", "--host", type=str, default="", help="host name")
    parser.add_argument("-C", "--color", action="store_true", help="colorize logs")
    parser.add_argument(
        "-l",
        "--level",
        type=str,
        default="DEBUG",
        help="log level (NOTSET|DEBUG|INFO|WARNING|ERROR|CRITICAL)",
    )
    parser.add_argument(
        "node",
        type=str,
        nargs="?",
        default="log_broker",
        help="node name or full name ({}) (default: log_broker)".format(
            join_name(("host", "node"))
        ),
    )

    args = parser.parse_args(args)

    return args


def _handler(level: str, fmt, msg):
    msg = parse_log(msg)
    if not should_show(level, msg):
        return
    print(fmt(msg), end="")


def main(args=None):
    args = parse_args(args)
    gconf, host, node = init_gconf_host_node(args.conf, args.host, args.node)
    joined_name = join_name((host, node))
    print("Opening LogClient for {} (config file: {}).".format(joined_name, args.conf))

    handler = partial(
        _handler, args.level.upper(), format_log_term_color if args.color else format_log
    )
    cli = LogClient(gconf, joined_name, log_num=1, log_handler=handler)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exitting.")
    finally:
        cli.close()
