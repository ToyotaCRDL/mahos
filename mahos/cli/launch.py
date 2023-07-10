#!/usr/bin/env python3

import typing as T
import time
import importlib
import argparse
import multiprocessing as mp

from ..node.node import Node, start_node_proc, join_name, local_conf, threaded_nodes
from ..gui.gui_node import GUINode, start_gui_node_proc
from .util import init_gconf_host
from .threaded_nodes import start_threaded_nodes_proc


def parse_args(args):
    parser = argparse.ArgumentParser(prog="mahos launch", description="Launch mahos nodes.")
    parser.add_argument(
        "-c", "--conf", type=str, default="conf.toml", help="config file name (default: conf.toml)"
    )
    parser.add_argument("-H", "--host", type=str, default="", help="host name")
    parser.add_argument(
        "-e",
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="node names (or threaded nodes names) to exclude",
    )
    parser.add_argument(
        "include",
        type=str,
        nargs="*",
        default=[],
        help="node names (or threaded nodes names) to include. leave empty to launch all.",
    )

    args = parser.parse_args(args)

    if args.include and args.exclude:
        parser.error("Only one of include or -e (--exclude) can be used.")

    return args


class Launcher(object):
    def __init__(
        self,
        gconf_fn: str,
        host: str,
        include: T.Optional[T.List[str]],
        exclude: T.Optional[T.List[str]],
    ):
        self.gconf, self.host = init_gconf_host(gconf_fn, host)
        self.include = include or []
        self.exclude = exclude or []

        self.ctx = mp.get_context()
        self.procs = {}
        self.shutdown_events = {}
        self.class_names = {}

        self.check_interval_sec = 0.5
        self.shutdown_delay_sec = 1.0
        self.shutdown_order = ["InstrumentServer", "LogBroker"]  # ParamServer?
        self._threaded_node_names = []

    def start_node(self, name: str):
        conf = local_conf(self.gconf, name)
        module = importlib.import_module(conf["module"])
        NodeClass = getattr(module, conf["class"])

        print(f"Starting {name}.")

        if issubclass(NodeClass, Node):
            proc, ev = start_node_proc(self.ctx, NodeClass, self.gconf, name)
            return proc, ev, conf["class"]
        elif issubclass(NodeClass, GUINode):
            proc = start_gui_node_proc(self.ctx, NodeClass, self.gconf, name)
            return proc, None, conf["class"]
        else:
            raise ValueError("{} isn't a valid Node class: {}".format(name, NodeClass.__name__))

    def is_excluded(self, name: str):
        return (self.include and name not in self.include) or name in self.exclude

    def start_threaded_nodes(self):
        for name, node_names in threaded_nodes(self.gconf, self.host).items():
            if self.is_excluded(name):
                continue
            proc, ev = start_threaded_nodes_proc(self.ctx, self.gconf, self.host, name)

            # Try to avoid name conflict with raw node names. just in case.
            n = f"thread::{self.host}::{name}"

            self.procs[n] = proc
            if ev is not None:
                self.shutdown_events[n] = ev
            self.class_names[n] = "ThreadedNodes"

            self._threaded_node_names.extend(node_names)

    def start_raw_nodes(self):
        for node_name in self.gconf[self.host]:
            if self.is_excluded(node_name) or node_name in self._threaded_node_names:
                continue
            n = join_name((self.host, node_name))
            proc, ev, class_name = self.start_node(n)

            self.procs[n] = proc
            if ev is not None:
                self.shutdown_events[n] = ev
            self.class_names[n] = class_name

    def start_all_nodes(self):
        self.start_threaded_nodes()
        self.start_raw_nodes()
        print("Started all nodes.")

    def check_alive(self):
        terminated = []
        for name, proc in self.procs.items():
            if not proc.is_alive():
                print("{} has been terminated.".format(name))
                terminated.append(name)
        for name in terminated:
            del self.procs[name]

    def check_loop(self):
        while True:
            self.check_alive()
            time.sleep(self.check_interval_sec)

    def shutdown_nodes(self):
        """Shutdown nodes gracefully by sending shutdown events."""

        # Nodes not in shutdown_order
        for name, ev in self.shutdown_events.items():
            if self.class_names[name] not in self.shutdown_order:
                print("Sending shutdown signal to {}.".format(name))
                ev.set()

        time.sleep(self.shutdown_delay_sec)

        # Nodes in shutdown_order
        for class_name in self.shutdown_order:
            for name, ev in self.shutdown_events.items():
                if self.class_names[name] == class_name:
                    print("Sending shutdown signal to {}.".format(name))
                    ev.set()
            time.sleep(self.shutdown_delay_sec)

    def terminate_procs(self):
        """Force shutdown the processes by sending term signal."""

        for name, proc in self.procs.items():
            if proc.is_alive():
                print("Sending term signal to {}.".format(name))
                proc.terminate()


def main(args=None):
    args = parse_args(args)

    launcher = Launcher(args.conf, args.host, args.include, args.exclude)
    print("Launching at {} (config file: {}).".format(launcher.host, args.conf))

    if launcher.include:
        print("Including nodes: {}".format(launcher.include))
    if launcher.exclude:
        print("Excluding nodes: {}".format(launcher.exclude))

    try:
        launcher.start_all_nodes()
        launcher.check_loop()
    except KeyboardInterrupt:
        launcher.shutdown_nodes()
    finally:
        launcher.terminate_procs()
