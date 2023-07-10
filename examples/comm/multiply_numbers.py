#!/usr/bin/env python3

from mahos.node.node import load_gconf
from nodes import MultiplierClient


def main():
    gconf = load_gconf("conf.toml")
    cli = MultiplierClient(gconf, "localhost::multiplier")

    numbers = [(1, 1), (2, 3), (100, 0)]
    for a, b in numbers:
        ans = cli.multiply(a, b)
        print(f"{a} * {b} = {ans}")


if __name__ == "__main__":
    main()
