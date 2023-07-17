#!/usr/bin/env python3

"""
Character User Interface module

.. This file is a part of MAHOS project.

This file contains some common functions for CUI.

"""

import os


def prompt(msg="?", default=None):
    """Prompt user for yes or no answer."""

    if default == "y":
        message = msg + "([y]/n):"
    elif default == "n":
        message = msg + "(y/[n]):"
    else:
        message = msg + "(y/n):"

    while True:
        ans = input(message)
        if (ans == "" and default == "y") or ans.lower() in ("y", "yes"):
            return True
        elif (ans == "" and default == "n") or ans.lower() in ("n", "no"):
            return False


def check_existance(fpath):
    if os.path.exists(fpath):
        return prompt(msg="%s already exists. Overwrite it?" % (fpath), default="y")
    else:
        return True
