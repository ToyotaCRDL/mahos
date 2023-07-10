#!/usr/bin/env python3

COLOURS = {
    "k": 0,
    "black": 0,
    "r": 1,
    "red": 1,
    "g": 2,
    "green": 2,
    "y": 3,
    "yellow": 3,
    "b": 4,
    "blue": 4,
    "p": 5,
    "purple": 5,
    "c": 6,
    "cyan": 6,
    "w": 7,
    "white": 7,
    "K": 8,
    "BLACK": 8,
    "R": 9,
    "RED": 9,
    "G": 10,
    "GREEN": 10,
    "Y": 11,
    "YELLOW": 11,
    "B": 12,
    "BLUE": 12,
    "P": 13,
    "PURPLE": 13,
    "C": 14,
    "CYAN": 14,
    "W": 15,
    "WHITE": 15,
}


def col(text, fg=None, bg=None, bold=False, uline=False):
    def parse(c):
        if type(c) == str:
            return COLOURS[c]
        elif type(c) == int and 0 <= c <= 255:
            return c
        else:
            raise ValueError("Invalid color code {}".format(c))

    bold = "1;" if bold else ""
    uline = "4;" if uline else ""
    fg = "38;5;{};".format(parse(fg)) if fg is not None else ""
    bg = "48;5;{};".format(parse(bg)) if bg is not None else ""

    code = bold + uline + fg + bg
    if not code:  # no modification made.
        return text
    return "\033[{}m{}\033[0m".format(code[:-1], text)


def printc(text, fg=None, bg=None, bold=False, uline=False, **keyw):
    print(col(text, fg, bg, bold, uline), **keyw)
