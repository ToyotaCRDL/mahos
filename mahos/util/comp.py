#!/usr/bin/env python3

"""
Comparison module.

.. This file is a part of MAHOS project.

Defines generic comparison functions.

"""

import numpy as np
from collections import OrderedDict


def has_compatible_types(val, val_other) -> bool:
    """Check types of two values are compatible.

    builtin list and tuple are considered compatible.
    builtin dict and collections.OrderedDict are considered compatible.
    builtin float and np.floating are considered compatible.
    builtin int and np.integer are considered compatible.
    builtin bool and np.bool_ are considered compatible.

    For other types, equality of type(val) and type(val_other) is returned.

    """

    if isinstance(val, (list, tuple)):
        return isinstance(val_other, (list, tuple))
    if isinstance(val, (dict, OrderedDict)):
        return isinstance(val_other, (dict, OrderedDict))
    if isinstance(val, (float, np.floating)):
        return isinstance(val_other, (float, np.floating))
    if isinstance(val, (int, np.integer)):
        return isinstance(val_other, (int, np.integer))
    if isinstance(val, (bool, np.bool_)):
        return isinstance(val_other, (bool, np.bool_))
    return type(val) == type(val_other)


def dict_equal(dic: dict, other: dict) -> bool:
    """Check if two dicts are equal.

    Recursively checks the equality of the contents (values) of two dicts.

    A difference of default impl. (params == other) is that
    this function can compare the dict containing np.ndarray.
    Builtin containers dict, list, or tuple can be included in the values.
    Values of other types are just checked by equality comparison (val == val_other).

    Also, the types are checked by has_compatible_types();
    if any values with incompatible types are contained,
    the result will be False.

    """

    def equal_val(val, val_other) -> bool:
        if not has_compatible_types(val, val_other):
            return False

        if isinstance(val, np.ndarray):
            if val.shape != val_other.shape or val.dtype != val_other.dtype:
                return False
            return bool(np.all(val == val_other))
        elif isinstance(val, dict):
            return dict_equal(val, val_other)
        elif isinstance(val, (list, tuple)):
            if len(val) != len(val_other):
                return False
            return all((equal_val(elem, elem_other) for elem, elem_other in zip(val, val_other)))
        else:
            return val == val_other

    if len(dic) != len(other):
        return False
    return all((k in other and equal_val(v, other[k]) for k, v in dic.items()))


def dict_equal_inspect(dic: dict, other: dict) -> bool:
    """Same as dict_equal() but report the encountered difference."""

    def equal_val(val, val_other) -> bool:
        if not has_compatible_types(val, val_other):
            print(f"type mismatch: {type(val)} != {type(val_other)}")
            return False

        if isinstance(val, np.ndarray):
            if val.shape != val_other.shape or val.dtype != val_other.dtype:
                print(
                    f"array shape / dtype mismatch: {val.shape} != {val_other.shape} or"
                    + f" {val.dtype} != {val_other.dtype}"
                )
                return False
            all_eq = bool(np.all(val == val_other))
            if not all_eq:
                print(f"array contents mismatch: {val} != {val_other}")
            return all_eq
        elif isinstance(val, dict):
            return dict_equal_inspect(val, val_other)
        elif isinstance(val, (list, tuple)):
            if len(val) != len(val_other):
                print(f"list or tuple size mismatch: {len(val)} != {len(val_other)}")
                return False
            return all((equal_val(elem, elem_other) for elem, elem_other in zip(val, val_other)))
        else:
            eq = val == val_other
            if not eq:
                print(f"{val} != {val_other}")
            return eq

    if len(dic) != len(other):
        print(f"dict size mismatch: {len(dic)} != {len(other)}")
        return False
    for k, v in dic.items():
        if k not in other:
            print(f"key {k} not in other dict")
            return False
        if not equal_val(v, other[k]):
            print(f"at key {k}")
            return False
    return True


def dict_isclose(dic: dict, other: dict, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if two dicts are close.

    This function does almost same comparison with dict_equal(); however,
    it uses np.isclose for comparison of floats (builtin float, np.floating, or float np.ndarray).

    """

    def isclose_val(val, val_other) -> bool:
        if not has_compatible_types(val, val_other):
            return False

        if isinstance(val, np.ndarray):
            if val.shape != val_other.shape or val.dtype != val_other.dtype:
                return False
            if np.issubdtype(val.dtype, np.floating):
                return bool(np.allclose(val, val_other, rtol=rtol, atol=atol))
            else:
                return bool(np.all(val == val_other))
        elif isinstance(val, (float, np.floating)):
            return bool(np.isclose(val, val_other, rtol=rtol, atol=atol))
        elif isinstance(val, dict):
            return dict_isclose(val, val_other)
        elif isinstance(val, (list, tuple)):
            if len(val) != len(val_other):
                return False
            return all((isclose_val(elem, elem_other) for elem, elem_other in zip(val, val_other)))
        else:
            return val == val_other

    if len(dic) != len(other):
        return False
    return all((k in other and isclose_val(v, other[k]) for k, v in dic.items()))


def dict_isclose_inspect(dic: dict, other: dict, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Same as dict_isclose() but report the encountered difference."""

    def isclose_val(val, val_other) -> bool:
        if not has_compatible_types(val, val_other):
            print(f"type mismatch: {type(val)} != {type(val_other)}")
            return False

        if isinstance(val, np.ndarray):
            if val.shape != val_other.shape or val.dtype != val_other.dtype:
                print(
                    f"array shape / dtype mismatch: {val.shape} != {val_other.shape} or"
                    + f" {val.dtype} != {val_other.dtype}"
                )
                return False
            if np.issubdtype(val.dtype, np.floating):
                eq = bool(np.allclose(val, val_other, rtol=rtol, atol=atol))
                if not eq:
                    print(f"float arrays are not allclose: {val} != {val_other}")
                return eq
            else:
                eq = bool(np.all(val == val_other))
                if not eq:
                    print(f"non-float arrays are not all equal: {val} != {val_other}")
                return eq
        elif isinstance(val, (float, np.floating)):
            eq = bool(np.isclose(val, val_other, rtol=rtol, atol=atol))
            if not eq:
                print(f"float values are not close: {val} != {val_other}")
            return eq
        elif isinstance(val, dict):
            return dict_isclose_inspect(val, val_other)
        elif isinstance(val, (list, tuple)):
            if len(val) != len(val_other):
                print(f"list or tuple size mismatch: {len(val)} != {len(val_other)}")
                return False
            return all((isclose_val(elem, elem_other) for elem, elem_other in zip(val, val_other)))
        else:
            eq = val == val_other
            if not eq:
                print(f"{val} != {val_other}")
            return eq

    if len(dic) != len(other):
        print(f"dict size mismatch: {len(dic)} != {len(other)}")
        return False
    for k, v in dic.items():
        if k not in other:
            print(f"key {k} not in other dict")
            return False
        if not isclose_val(v, other[k]):
            print(f"at key {k}")
            return False
    return True
