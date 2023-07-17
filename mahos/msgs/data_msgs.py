#!/usr/bin/env python3

"""
Base Data type definitions for mahos measurements.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import uuid
import datetime

import numpy as np
import h5py
import msgpack

from .common_msgs import Message


_H5_RESERVED_ATTRS = ("_description", "_type", "_save_time", "_version_h5_base")


class Data(Message):
    """Base class for measurement data."""

    # Common meta data: note, version, version_h5

    def set_note(self, note: str):
        self._note = note

    def note(self) -> str:
        if hasattr(self, "_note"):
            return self._note
        else:
            return ""

    def set_version(self, version: int):
        self._version = version

    def version(self) -> int:
        """The version of this data.

        The version should be updated when data schema is changed,
        e.g., when an attribute is added, deleted, or renamed.

        """

        if hasattr(self, "_version"):
            return self._version
        else:
            return 0

    def set_version_h5(self, version: int):
        self._version_h5 = version

    def version_h5(self) -> int:
        """The version of HDF5 IO (h5_writers / h5_readers) for this data.

        The version should be updated when behavour of writer or reader is changed.

        """

        if hasattr(self, "_version_h5"):
            return self._version_h5
        else:
            return 0

    # params

    def init_params(self, params: dict | None):
        if params is None:
            self.params = None
            self.ident = uuid.uuid4()
            return

        self.params = {k: v for k, v in params.items() if k != "ident"}
        if params.get("ident") is not None:
            self.ident = params["ident"]
        else:
            self.ident = uuid.uuid4()

    def has_params(self) -> bool:
        """return True if this has meaningful params, i.e., this is not dummy."""

        return self.params is not None

    def update_params(self, params: dict | None):
        """update the params except `ident`. if None is given, do nothing."""

        if params is None:
            return
        self.params.update({k: v for k, v in params.items() if k != "ident"})

    # h5

    def _h5_attr_writers(self) -> dict:
        """Get dict of key: writer, where writer writes self.key as an attribute.

        The write is actually a filter.
        It should convert self.key to an h5 (numpy) value.

        The returned value is assigined as group.attrs[key] = value.

        """

        return {}

    def _h5_dataset_writers(self) -> dict:
        """Get dict of key: writer, where writer writes self.key as a dataset.

        The write is actually a filter.
        It should convert self.key to an h5 (numpy) value.

        """

        return {}

    def _h5_readers(self) -> dict:
        """Get dict of key: reader, where reader reads self.key stored in hdf5 format.

        The reader is actually a filter.
        It should take an h5 (numpy) value and return the converted value.

        """

        return {}

    def to_h5(self, group: h5py.File | h5py.Group, compression=None, compression_opts=None):
        """Write this Data to hdf5 file (root group) or group.

        :raises ValueError: the name exists in `group`, data type not understood, etc.

        """

        # Set the h5-only attributes for metadata,
        # which won't be appeared in Data's attribute in Python.
        # 1. _description is a comment for data viewer who cares little about mahos.
        #    Since it's only for human, the contents may be changed without versioning.
        # 2. _save_time is a string to store the date/time when this date is saved
        #    (when this function is called).
        # 3. _type is a string to store the type information in format 'module.class'
        #    We won't rely on _type for loading because module layout can be changed.
        # 4. _version_h5_base is the version of base HDF5 IO (Data.to_h5() and Data.of_h5()).
        #    In other words, the version of how the Data attribute is converted to / from h5
        #    by default, not through custom h5_writers / h5_readers.

        if any((hasattr(self, s) for s in _H5_RESERVED_ATTRS)):
            raise RuntimeError("Attributes are not allowed: " + ", ".join(_H5_RESERVED_ATTRS))
        d = [
            "MAHOS Data in HDF5",
            "See attribute _type for type information (module.class) for this Data",
            "Binary blobs (Opaque values) are in MessagePack format",
        ]
        group.attrs["_description"] = "\n".join(d)
        group.attrs["_save_time"] = datetime.datetime.now().astimezone().isoformat()
        group.attrs["_type"] = ".".join((self.__class__.__module__, self.__class__.__qualname__))
        group.attrs["_version_h5_base"] = 0

        attr_writers = self._h5_attr_writers()
        dataset_writers = self._h5_dataset_writers()
        for key, val in self.__dict__.items():
            if val is None:
                # None val is skipped because there is no dedicated way to express None (null)
                # in HDF5 (or numpy).
                # We could use empty array for example, but we don't need to do it.
                pass
            elif key in attr_writers:
                group.attrs[key] = attr_writers[key](val)
            elif key in dataset_writers:
                group.create_dataset(
                    key,
                    data=dataset_writers[key](val),
                    compression=compression,
                    compression_opts=compression_opts,
                )
            elif isinstance(val, np.ndarray):
                group.create_dataset(
                    key, data=val, compression=compression, compression_opts=compression_opts
                )
            elif isinstance(val, dict):
                # for backward compatibility: in case ident (uuid.UUID) is contained in params.
                d = val.copy()
                if "ident" in d and isinstance(d["ident"], uuid.UUID):
                    d["ident"] = d["ident"].hex
                group.attrs[key] = np.void(msgpack.dumps(d))
            elif key == "ident" and isinstance(val, uuid.UUID):
                group.attrs[key] = val.hex
            elif isinstance(val, (bool, int, float, str)):
                group.attrs[key] = val
            else:
                raise ValueError(
                    f"Data type not understood: key: {key} type: {type(val)} val: {val}"
                )

    @classmethod
    def _h5_read_group_attr(cls, key, val, readers):
        if key in readers:
            return readers[key](val)
        elif isinstance(val, np.void):
            d = msgpack.loads(val.tobytes())
            if "ident" in d and isinstance(d["ident"], str):
                d["ident"] = uuid.UUID(hex=d["ident"])
            return d
        elif key == "ident" and isinstance(key, str):
            return uuid.UUID(hex=val)
        elif isinstance(val, str):
            return val
        elif isinstance(val, np.bool_):
            return bool(val)
        elif isinstance(val, np.integer):
            return int(val)
        elif isinstance(val, np.floating):
            return float(val)
        else:
            raise ValueError(f"Data type not understood: key: {key} type: {type(val)} val: {val}")

    @classmethod
    def of_h5(cls, group: h5py.File | h5py.Group) -> Data:
        """Create a Data from hdf5 file or group.

        :raises ValueError: data type not understood, etc.

        """

        data = cls()
        readers = data._h5_readers()

        # read the h5 formatting versions first.
        # _version_h5_base will be used in this function, but not for now.
        # version_h5_base = group.attrs.get("_version_h5_base", 0)
        # _version_h5 may be used by the readers
        if "_version_h5" in group.attrs:
            data.set_version_h5(group.attrs["_version_h5"])
        # if data is unversioned, set version 0 explicitly.
        if "_version" not in group.attrs:
            data.set_version(0)

        for key, val in group.items():
            # if key not in d.__dict__:
            #     continue
            # We should skip checking key existence like this
            # because key name may be changed by version update of Data.

            if isinstance(val, h5py.Group):
                # unlikely (unintended usage), but this group may not be a leaf.
                pass
            elif key in readers:
                setattr(data, key, readers[key](val))
            else:
                setattr(data, key, np.array(val))

        for key, val in group.attrs.items():
            if key not in _H5_RESERVED_ATTRS + ("_version_h5",):
                setattr(data, key, cls._h5_read_group_attr(key, val, readers))

        return data

    @classmethod
    def _get_attr_h5(cls, group: h5py.File | h5py.Group, key: str):
        data = cls()
        readers = data._h5_readers()

        # read the h5 formatting versions first.
        # _version_h5_base will be used in this function, but not for now.
        # version_h5_base = group.attrs.get("_version_h5_base", 0)
        # _version_h5 may be used by the readers
        if "_version_h5" in group.attrs:
            data.set_version_h5(group.attrs["_version_h5"])

        if key in group:
            if key in readers:
                return True, readers[key](group[key])
            else:
                return True, np.array(group[key])

        if key in group.attrs:
            return True, cls._h5_read_group_attr(key, group.attrs[key], readers)

        return False, None

    @classmethod
    def get_attr_h5(cls, group: h5py.File | h5py.Group, key: str, default=None):
        """Get an attribute from hdf5 file or group.

        If key is not found, `default` is returned.

        :raises ValueError: data type not understood, etc.

        """

        found, val = cls._get_attr_h5(group, key)
        if found:
            return val
        return default

    @classmethod
    def get_attr_h5_exn(cls, group: h5py.File | h5py.Group, key: str):
        """Get an attribute from hdf5 file or group.

        If key is not found, raises KeyError.

        :raises ValueError: data type not understood, etc.
        :raises KeyError: key not found.

        """

        found, val = cls._get_attr_h5(group, key)
        if found:
            return val
        raise KeyError(key)

    @classmethod
    def list_attr_h5(cls, group: h5py.File | h5py.Group) -> list[tuple[str, str]]:
        """Get list of the attributes and type names from hdf5 file or group.

        :raises ValueError: data type not understood, etc.

        """

        res = []
        for key, val in group.items():
            if isinstance(val, h5py.Group):
                # unlikely (unintended usage), but this group may not be a leaf.
                pass
            res.append((key, "{:s} {:s}".format(val.dtype.name, str(val.shape))))

        for key, val in group.attrs.items():
            res.append((key, type(val).__name__))
        return res
