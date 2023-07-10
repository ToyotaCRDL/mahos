#!/usr/bin/env python3

import time

from mahos.node.node import load_gconf
from nodes import ClockClient


def main():
    gconf = load_gconf("conf.toml")
    cli = ClockClient(gconf, "localhost::clock")

    for i in range(10):
        print(cli.get_time())
        time.sleep(1.0)


if __name__ == "__main__":
    main()
