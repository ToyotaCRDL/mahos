#!/usr/bin/env python3

"""
Message Types for Generic Parameters.

.. This file is a part of MAHOS project.

Param is a wrapper around a value of primitive (mostly Python builtin) type, with a default.
NumberParam has bounds (minimum and maximum).
ChoiceParam has its available options.
Collection of parameters are expressed as a ParamDict: nestable, str-keyed dict of Params.

GetParamDictReq (get_param_dict()) is usually served by meas nodes,
to tell the client about required parameters for a measurement.
The client can use param_dict to update the parameter values and
pass it in StateReq (change_state()).
At the meas node side, it is usually required to unwrap the Params using unwrap().

"""

from __future__ import annotations
import typing as T
import enum
import uuid
from collections import UserDict
import re

import numpy as np

from .common_msgs import Request
from ..util.unit import SI_scale
from ..util.comp import dict_isclose, has_compatible_types


class Param(object):
    """Base class for all Param types."""

    def __init__(self, typ, default, optional: bool, enable: bool, doc: str):
        self._type = typ
        self._default = default
        self._optional = optional
        self._enabled = enable
        self._doc = doc

        if not self.restore_default():
            raise ValueError("Default value is invalid.")

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.value())

    def default(self):
        return self._default

    def restore_default(self) -> bool:
        return self.set(self._default)

    def value(self):
        """Get raw value of this Param."""

        if self._optional and not self._enabled:
            return None
        return self._value

    def some_value(self):
        """Get raw value of this Param ignoring optional state."""

        return self._value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.value() == other.value()

    def validate(self, value) -> bool:
        return True

    def set(self, value) -> bool:
        """Set value of this Param. Return True on success."""

        if not isinstance(value, self._type):
            try:
                value = self._type(value)
            except ValueError:
                return False

        if not self.validate(value):
            return False

        self._value = value
        return True

    def set_enable(self, enable: bool = True) -> bool:
        """Set enable or disable of this Param. Return True on success."""

        if not self._optional:
            return False

        self._enabled = enable

    def set_disable(self) -> bool:
        """Alias for set_enable(False). Return True on success."""

        return self.set_enable(False)

    def optional(self) -> bool:
        """Return True if this Param is optional."""

        return self._optional

    def enabled(self) -> bool:
        """Return True if this Param is enabled."""

        return self._enabled

    def doc(self) -> str:
        """Get help documentation of this Param."""

        return self._doc

    def help(self) -> str:
        """Alias for doc()."""

        return self.doc()


class NumberParam(Param):
    """Base class for Param types represents a number."""

    def __init__(
        self, typ, default, minimum, maximum, unit, SI_prefix, digit, optional, enable, doc
    ):
        self._minimum = minimum
        self._maximum = maximum
        self._unit = unit
        self._SI_prefix = SI_prefix
        self._digit = digit

        Param.__init__(self, typ, default, optional, enable, doc)

    def value_to_str(self, val=None, digits: int = 3) -> str:
        if val is None:
            val = self.value()
        if val is None:
            return "None"

        if self.SI_prefix():
            factor, prefix = SI_scale(val)
            val *= factor
        else:
            prefix = ""
        unit = self.unit()

        if isinstance(self, IntParam):
            vs = f"{int(val):d}"
        else:  # FloatParam
            vs = f"{val:.{self._digit}f}"
        if prefix or unit:
            vs += f" {prefix}{unit}"
        return vs

    def __repr__(self):
        cn = self.__class__.__name__
        if self._minimum is None and self._maximum is None:
            return "<{}: {}>".format(cn, self.value_to_str(self.value()))
        if self._minimum is None:
            return "<{}: {} [<={}]>".format(
                cn, self.value_to_str(), self.value_to_str(self.maximum())
            )
        if self._maximum is None:
            return "<{}: {} [>={}]>".format(
                cn, self.value_to_str(), self.value_to_str(self.minimum())
            )
        return "<{}: {} [{}, {}]>".format(
            cn,
            self.value_to_str(),
            self.value_to_str(self.minimum()),
            self.value_to_str(self.maximum()),
        )

    def validate(self, value) -> bool:
        if self._minimum is not None and value < self._minimum:
            return False
        if self._maximum is not None and value > self._maximum:
            return False

        return True

    def unit(self) -> str:
        return self._unit

    def SI_prefix(self) -> bool:
        return self._SI_prefix

    def digit(self) -> int:
        return self._digit

    def minimum(self):
        """Get minimum value (lower bound) of this NumberParam."""

        return self._minimum

    def maximum(self):
        """Get maximum value (upper bound) of this NumberParam."""

        return self._maximum

    def bounds(self):
        """Get bounds (minimum, maximum) of this NumberParam."""

        return (self.minimum(), self.maximum())


class ChoiceParam(Param):
    """Base class for Param types expressing a choice from finite set `options`."""

    def __init__(self, typ, default, options, optional, enable, doc):
        self._options = tuple(options)

        Param.__init__(self, typ, default, optional, enable, doc)

    def __repr__(self):
        options = ", ".join([str(v) for v in self.options()])
        return "<{}: {} [{}]>".format(self.__class__.__name__, self.value(), options)

    def validate(self, value) -> bool:
        return value in self._options

    def options(self) -> tuple:
        """Get available options for this Param."""

        return self._options


class StrParam(Param):
    """Param type of builtin str."""

    def __init__(
        self, default: str = "", optional: bool = False, enable: bool = True, doc: str = ""
    ):
        Param.__init__(self, str, default, optional, enable, doc)


class UUIDParam(Param):
    """Param type of uuid.UUID."""

    def __init__(self, optional: bool = False, enable: bool = True, doc: str = ""):
        Param.__init__(self, uuid.UUID, uuid.uuid4(), optional, enable, doc)


class IntParam(NumberParam):
    """Param type of builtin int."""

    def __init__(
        self,
        default: int = 0,
        minimum=None,
        maximum=None,
        unit: str = "",
        SI_prefix: bool = False,
        digit: int = 6,
        optional: bool = False,
        enable: bool = True,
        doc: str = "",
    ):
        NumberParam.__init__(
            self, int, default, minimum, maximum, unit, SI_prefix, digit, optional, enable, doc
        )


class FloatParam(NumberParam):
    """Param type of builtin float."""

    def __init__(
        self,
        default: float = 0.0,
        minimum=None,
        maximum=None,
        unit: str = "",
        SI_prefix: bool = False,
        digit: int = 6,
        optional: bool = False,
        enable: bool = True,
        doc: str = "",
    ):
        NumberParam.__init__(
            self, float, default, minimum, maximum, unit, SI_prefix, digit, optional, enable, doc
        )


class BoolParam(ChoiceParam):
    """Param type of builtin bool."""

    def __init__(self, default: bool, optional: bool = False, enable: bool = True, doc: str = ""):
        ChoiceParam.__init__(self, bool, default, (True, False), optional, enable, doc)


class EnumParam(ChoiceParam):
    """Param type of Enum."""

    def __init__(
        self,
        typ: enum.Enum,
        default,
        options=None,
        optional: bool = False,
        enable: bool = True,
        doc: str = "",
    ):
        if options is None:
            options = tuple(typ)
        ChoiceParam.__init__(self, typ, default, options, optional, enable, doc)


class StrChoiceParam(ChoiceParam):
    """Param type of selection type of builtin strs."""

    def __init__(
        self,
        default,
        options: tuple[str],
        optional: bool = False,
        enable: bool = True,
        doc: str = "",
    ):
        ChoiceParam.__init__(self, str, default, options, optional, enable, doc)


class IntChoiceParam(ChoiceParam):
    """Param type of selection type of builtin ints."""

    def __init__(
        self,
        default,
        options: tuple[int],
        optional: bool = False,
        enable: bool = True,
        doc: str = "",
    ):
        ChoiceParam.__init__(self, int, default, options, optional, enable, doc)


_key_pattern = re.compile(r"\.?(?P<word>[\w-]+)|\[(?P<num>\d+)\]")


class ParamDict(UserDict):
    def _split_key(self, k: str) -> list[str | int]:
        keys = []
        for match in re.finditer(_key_pattern, k):
            if match.lastgroup == "word":
                keys.append(match.groups()[0])
            else:  # "num"
                keys.append(int(match.groups()[1]))
        return keys

    def getf(self, key: str, default=None) -> PDValue | None:
        """Search for a value in ParamDict `as if it is flattened`.

        :param key: dict key for flattened dict.
            Expression like 'level0.level1.level2' can be used to express nested dict access.
            Expression like 'key_of_list[2]' can be used to express list indexing.

        """

        d = self.data
        keys = self._split_key(key)
        if not keys:
            return default

        try:
            for k in keys:
                d = d[k]
            return d
        except KeyError:
            return default

    def flatten(self) -> ParamDict[str, Param]:
        """Get `flattened view` of the nested-dict / list structure in this ParamDict.

        Key like 'level0.level1.level2' can be used to express nested dict access.
        Key like 'key_of_list[2]' can be used to express list indexing.

        """

        def add(d, key, val):
            if isinstance(val, Param):
                d[key] = val
            elif isinstance(val, (dict, ParamDict)):
                for k, v in val.items():
                    add(d, ".".join((key, k)), v)
            elif isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    add(d, f"{key}[{i}]", v)
            else:
                raise ValueError("Invalid ParamDict structure.")

        d = ParamDict()
        for key, val in self.data.items():
            add(d, key, val)
        return d

    def validate(self) -> bool:
        """Check if this is valid ParamDict."""

        def validate_key(key: str) -> bool:
            return isinstance(key, str) and not any([s in key for s in (".", "[", "]")])

        def validate_val(val) -> bool:
            if isinstance(val, Param):
                return True
            elif isinstance(val, (dict, ParamDict)):
                return ParamDict(val).validate()
            elif isinstance(val, (list, tuple)):
                return all((validate_val(elem) for elem in val))
            else:
                return False

        return all((validate_key(k) and validate_val(v) for k, v in self.data.items()))

    def _unwrap(self, process_unknown) -> dict[str, RawPDValue]:
        """Unwrap values of given ParamDict to get RawParamDict."""

        def unwrap_val(val):
            if isinstance(val, Param):
                return val.value()
            elif isinstance(val, (dict, ParamDict)):
                return ParamDict(val)._unwrap(process_unknown)
            elif isinstance(val, (list, tuple)):
                return [unwrap_val(elem) for elem in val]
            else:
                return process_unknown(val)

        return {k: unwrap_val(v) for k, v in self.data.items()}

    def unwrap(self) -> dict[str, RawPDValue]:
        """Unwrap this ParamDict to get RawParamDict.

        This function (compared to unwrap_exn) is not strict;
        if the value has unknown type (typically raw value is incorporated), pass through it.

        """

        return self._unwrap(lambda v: v)

    def unwrap_exn(self) -> dict[str, RawPDValue]:
        """Unwrap values of given ParamDict to get RawParamDict.

        This function (compared to unwrap) is strict;
        if the value has unknown type (typically raw value is incorporated), raises TypeError.

        :raises TypeError: unknown type encountered.

        """

        def _raise_exn(val):
            raise TypeError(f"Encountered value {val} of type {type(val)}")

        return self._unwrap(_raise_exn)

    def isclose(
        self, other: ParamDict[str, PDValue], rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """Check if this ParamDict has same structure and values as `other`.

        Difference between self == other is that this method uses np.isclose to compare floats.

        """

        def isclose_val(val, val_other) -> bool:
            if not has_compatible_types(val, val_other):
                return False

            if isinstance(val, FloatParam):
                return np.isclose(val.value(), val_other.value(), rtol=rtol, atol=atol)
            elif isinstance(val, Param):
                return val.value() == val_other.value()
            elif isinstance(val, (dict, ParamDict)):
                return ParamDict(val).isclose(ParamDict(val_other), rtol=rtol, atol=atol)
            elif isinstance(val, (list, tuple)):
                if len(val) != len(val_other):
                    return False
                return all((isclose_val(elem, elem_) for elem, elem_ in zip(val, val_other)))
            else:  # will not reach here if self.valid()
                return False

        if len(self.data) != len(other):
            return False
        return all((k in other and isclose_val(v, other[k]) for k, v in self.data.items()))


#: Allowed value types for ParamDict
PDValue = T.Union[Param, ParamDict, T.List["PDValue"]]

#: Allowed value types for Raw (unwrapped) ParamDict
RawPDValue = T.Union[bool, str, int, float, enum.Enum, uuid.UUID, dict, T.List["RawPDValue"]]

#: This is just for reference. ParamDict has str key.
ParamDict_T = T.Dict[str, PDValue]
# we want to do this but not allowed in Python 3.8
# ParamDict_T = ParamDict[str, PDValue]

#: This is just for reference. RawParamDict has str key
RawParamDict_T = T.Dict[str, RawPDValue]
# this but not allowed in Python 3.8 too
# RawParamDict_T = dict[str, RawPDValue]


def unwrap(params: ParamDict[str, PDValue] | dict[str, RawPDValue]) -> dict[str, RawPDValue]:
    """Unwrap the ParamDict to RawParamDict. If params is not a ParamDict, do nothing."""

    if isinstance(params, ParamDict):
        return params.unwrap()
    return params


def isclose(
    params: ParamDict[str, PDValue] | dict[str, RawPDValue],
    other: ParamDict[str, PDValue] | dict[str, RawPDValue],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check if two ParamDicts or RawParamDicts are close."""

    if isinstance(params, ParamDict) and isinstance(other, ParamDict):
        return params.isclose(other, rtol=rtol, atol=atol)
    elif isinstance(params, dict) and isinstance(other, dict):
        return dict_isclose(params, other, rtol=rtol, atol=atol)
    else:
        return False


class GetParamDictReq(Request):
    """Request to get ParamDict."""

    def __init__(self, name: str = "", group: str = ""):
        self.name = name
        self.group = group


class GetParamDictNamesReq(Request):
    """Request to get list names of available ParamDicts."""

    def __init__(self, group: str = ""):
        self.group = group
