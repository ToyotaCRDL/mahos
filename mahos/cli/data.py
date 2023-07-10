#!/usr/bin/env python3

from . import data_ls
from . import data_print
from . import plot

data_usage = """usage: mahos data COMMAND args

COMMAND (l[s] | p[lot] | pr[int]) :
    the command to execute.

"""


def main(args):
    if len(args) < 2:
        print(data_usage)
        return 1

    pkg = args[0].lower()

    if "ls".startswith(pkg):
        return data_ls.main(args[1:])
    elif "plot".startswith(pkg):
        return plot.main(args[1:])
    elif "print".startswith(pkg):
        return data_print.main(args[1:])
    else:
        print(data_usage)
        return 1
