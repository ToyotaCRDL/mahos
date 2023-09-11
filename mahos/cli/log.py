#!/usr/bin/env python3

import time
import argparse
from functools import partial

from ..node.node import join_name, is_threaded
from ..node.log_broker import (
    LogClient,
    parse_log,
    format_log,
    format_log_term_color,
    should_show,
    log_broker_is_up,
)
from .util import init_gconf_host_node, host_is_local
from .launch import Launcher


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
        default="log",
        help="node name or full name ({}) (default: log)".format(join_name(("host", "node"))),
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

    if (
        host_is_local(gconf, host)
        and not is_threaded(gconf, joined_name)
        and not log_broker_is_up(gconf, joined_name)
    ):
        print(f"Automatically starting {joined_name}.")
        launcher = Launcher(args.conf, args.host, [node], [], shutdown_delay_sec=0.2)
        launcher.start_all_nodes()
    else:
        launcher = None

    print("Opening LogClient for {} (config file: {}).".format(joined_name, args.conf))

    handler = partial(
        _handler, args.level.upper(), format_log_term_color if args.color else format_log
    )
    cli = LogClient(gconf, joined_name, log_num=1, log_handler=handler)

    if launcher is not None:
        try:
            launcher.check_loop()
        except KeyboardInterrupt:
            launcher.shutdown_nodes()
        finally:
            launcher.terminate_procs()
            cli.close()
    else:
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exitting.")
        finally:
            cli.close()
