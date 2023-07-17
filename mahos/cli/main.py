#!/usr/bin/env python3

"""
main entrypoint of mahos cli.

.. This file is a part of MAHOS project.

"""

import sys
import os

from . import run, launch, log, ls, graph, echo, shell, data, plot

main_usage = """usage: mahos COMMAND args

COMMAND (r[un] | l[aunch] | lo[g] | ls | g[raph] | e[cho] | s[hell] | d[ata] | p[lot]) :
    the command to execute.

"""


def main():
    if len(sys.argv) < 2:
        print(main_usage)
        return 1

    # Let us import the modules in cwd
    sys.path.append(os.getcwd())

    pkg = sys.argv[1].lower()
    if "run".startswith(pkg):
        return run.main(sys.argv[2:])
    elif "launch".startswith(pkg):
        return launch.main(sys.argv[2:])
    elif "log".startswith(pkg):
        return log.main(sys.argv[2:])
    elif "ls" == pkg:
        return ls.main(sys.argv[2:])
    elif "graph".startswith(pkg):
        return graph.main(sys.argv[2:])
    elif "echo".startswith(pkg):
        return echo.main(sys.argv[2:])
    elif "shell".startswith(pkg):
        return shell.main(sys.argv[2:])
    elif "data".startswith(pkg):
        return data.main(sys.argv[2:])
    elif "plot".startswith(pkg):
        return plot.main(sys.argv[2:])
    else:
        print(main_usage)
        return 1


if __name__ == "__main__":
    sys.exit(main())
