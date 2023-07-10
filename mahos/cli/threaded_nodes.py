#!/usr/bin/env python3

import importlib
import time
import multiprocessing as mp

from ..node.node import Node, join_name, local_conf, start_node_thread, threaded_nodes
from ..node.comm import Context
from ..gui.gui_node import GUINode, start_gui_node_thread


class ThreadedNodes(object):
    """Runner for threaded nodes."""

    def __init__(self, gconf, host, name):
        self.gconf = gconf
        self.host = host
        self.name = name
        self.threads = {}
        self.class_names = {}
        self.shutdown_events = {}
        self.check_interval_sec = 0.5
        self.shutdown_delay_sec = 0.5
        self.shutdown_order = ["InstrumentServer", "LogBroker"]  # ParamServer?

    def start(self):
        ctx = Context()

        tnodes = threaded_nodes(self.gconf, self.host)
        if self.name not in tnodes:
            raise KeyError(f"Threaded nodes {self.name} at {self.host} is not defined.")
        print(
            "Starting thread::{}::{} ({}).".format(
                self.host, self.name, ", ".join(tnodes[self.name])
            )
        )

        for node_name in tnodes[self.name]:
            name = join_name((self.host, node_name))
            conf = local_conf(self.gconf, name)
            module = importlib.import_module(conf["module"])
            NodeClass = getattr(module, conf["class"])

            if issubclass(NodeClass, Node):
                thread, ev = start_node_thread(ctx, NodeClass, self.gconf, name)
            elif issubclass(NodeClass, GUINode):
                thread = start_gui_node_thread(ctx, NodeClass, self.gconf, name)
                ev = None

            self.threads[name] = thread
            if ev is not None:
                self.shutdown_events[name] = ev
            self.class_names[name] = conf["class"]

    def check_alive(self):
        terminated = []
        for name, thread in self.threads.items():
            if not thread.is_alive():
                print("{} has been terminated.".format(name))
                terminated.append(name)
        for name in terminated:
            del self.threads[name]

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

    def main_interrupt(self):
        try:
            self.start()
            while True:
                self.check_alive()
                time.sleep(self.check_interval_sec)
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt: Exitting ThreadedNodes({self.name}).")
        finally:
            self.shutdown_nodes()

    def main_event(self, shutdown_ev):
        try:
            self.start()
            while not shutdown_ev.is_set():
                self.check_alive()
                time.sleep(self.check_interval_sec)
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt: Exitting ThreadedNodes({self.name}).")
        finally:
            self.shutdown_nodes()


def run_threaded_nodes_proc(gconf: dict, host: str, name: str, shutdown_ev: mp.Event):
    tn = ThreadedNodes(gconf, host, name)
    tn.main_event(shutdown_ev)


def start_threaded_nodes_proc(
    ctx: mp.context.BaseContext, gconf: dict, host: str, name: str
) -> (mp.Process, mp.Event):
    shutdown_ev = mp.Event()
    proc = ctx.Process(target=run_threaded_nodes_proc, args=(gconf, host, name, shutdown_ev))
    proc.start()
    return proc, shutdown_ev
