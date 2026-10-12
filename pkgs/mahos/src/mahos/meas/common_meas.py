#!/usr/bin/env python3

"""
Common implementations for meas nodes.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import os

from mahos.msgs.common_msgs import BinaryStatus, BinaryState, StateReq, Request, Reply
from mahos.msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from mahos.msgs.common_msgs import (
    SaveArtifactReq,
    ExportArtifactReq,
    LoadArtifactReq,
    CleanupArtifactReq,
)
from mahos.msgs.common_meas_msgs import PopBufferReq, ClearBufferReq, FitReq, ClearFitReq
from mahos.msgs.common_meas_msgs import Buffer, BasicMeasData
from mahos.msgs import param_msgs as P
from mahos.inst.server import MultiInstrumentClient
from mahos.node.node import Node, NodeName
from mahos.node.client import NodeClient, StateClientMixin
from mahos.node.comm import Context
from mahos.meas.file_transport import FileTransportClientMixin, FileTransportNodeMixin
from mahos.meas.tweaker import TweakSaver


class BasicMeasClientBase(FileTransportClientMixin, NodeClient):
    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        prefix: str | None = None,
        status_handler=None,
        data_handler=None,
        buffer_handler=None,
        file_transport_dir: str | None = None,
    ):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self._get_status, self._get_data, self._get_buffer = self.add_sub(
            [(b"status", status_handler), (b"data", data_handler), (b"buffer", buffer_handler)]
        )

        self.req = self.add_req(gconf)
        self.init_file_transport(file_transport_dir)

    def get_status(self) -> BinaryStatus | None:
        return self._get_status()

    def get_data(self) -> BasicMeasData:
        return self._get_data()

    def get_buffer(self) -> Buffer:
        return self._get_buffer()


class ParamDictReqMixin(object):
    """Implements get_param_dict_labels() and get_param_dict()."""

    def get_param_dict_labels(self) -> list[str]:
        """Get list of available ParamDict labels."""

        rep = self.req.request(P.GetParamDictLabelsReq())
        if rep.success:
            return rep.ret
        else:
            return []

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for a measurement with `label`.

        :param label: param dict label (measurement method etc.).
                      can be empty if target node provides only one label.

        """

        rep = self.req.request(P.GetParamDictReq(label))
        if rep.success:
            return rep.ret
        else:
            return None


class BaseMeasClientMixin(StateClientMixin, ParamDictReqMixin):
    """Implements change_state(), get_state(), get_param_dict_labels(), and get_param_dict()."""

    pass


class BasicMeasReqMixin(FileTransportClientMixin):
    """implements start(), stop(), {save,export,load}_data(), and {pop,clear}_buffer()."""

    def start(self, params=None, label: str = "") -> bool:
        """Start the measurement, i.e., change state to ACTIVE."""

        return self.change_state(BinaryState.ACTIVE, params=params, label=label)

    def stop(self, params=None, label: str = "") -> bool:
        """Stop the measurement, i.e., change state to IDLE."""

        return self.change_state(BinaryState.IDLE, params=params, label=label)

    def save_data(self, file_name: str, params: dict | None = None, note: str = "") -> bool:
        """Save data to `file_name`."""

        return self.request_output(
            SaveDataReq(file_name, params=params, note=note),
            SaveArtifactReq(os.path.basename(file_name), params=params, note=note),
            file_name,
        )

    def export_data(
        self,
        file_name: str,
        data: BasicMeasData | list[BasicMeasData] | None = None,
        params: dict | None = None,
    ) -> bool:
        """Export data to `file_name`.

        :param data: If None, data holded by worker is used.

        """

        return self.request_output(
            ExportDataReq(file_name, data=data, params=params),
            ExportArtifactReq(os.path.basename(file_name), data=data, params=params),
            file_name,
        )

    def load_data(self, file_name: str, to_buffer: bool = False) -> BasicMeasData | None:
        """Load data from `file_name`.

        :param to_buffer: True: load data to buffer. False: overwrite current data with loaded one.

        """

        rep = self.request_input(
            LoadDataReq(file_name, to_buffer=to_buffer),
            lambda artifact: LoadArtifactReq(
                artifact, os.path.basename(file_name), to_buffer=to_buffer
            ),
            file_name,
        )
        return rep.ret if rep.success else None

    def pop_buffer(self, index: int = -1) -> BasicMeasData | None:
        rep = self.req.request(PopBufferReq(index))
        if rep.success:
            return rep.ret
        else:
            return None

    def clear_buffer(self) -> bool:
        rep = self.req.request(ClearBufferReq())
        return rep.success

    def fit(self, params: dict, label: str = "", data_index=-1) -> Reply:
        """Fit data."""

        return self.req.request(FitReq(params, label, data_index))

    def clear_fit(self, data_index=-1) -> bool:
        """Clear fit data."""

        rep = self.req.request(ClearFitReq(data_index))
        return rep.success


class BasicMeasClient(BasicMeasClientBase, BaseMeasClientMixin, BasicMeasReqMixin):
    """Client for Basic Measurement Nodes (Node with the BinaryState, Data, and Buffer).

    Client with a subscriber to status (with state inside) & data, and a requester.

    """

    # overrides for annotations

    def get_state(self) -> BinaryState | None:
        s = self.get_status()
        return s.state if s is not None else None


class BasicMeasNode(FileTransportNodeMixin, Node):
    """Base implementation for basic measurement nodes (a node with BinaryState, Data, and Buffer).

    Implements initialization (clients and communication), ``change_state()``,
    and ``get_param_dict()``.

    :param target.servers: InstrumentServer targets (instrument name -> server full name).
    :type target.servers: dict[str, str]
    :param target.tweakers: Optional Tweaker node names saved alongside measurement data.
    :type target.tweakers: list[str]
    :param target.log: LogBroker target full name.
    :type target.log: str
    :param inst_remap: Optional logical-to-physical instrument name remapping.
    :type inst_remap: dict[str, str]
    :param file_transport_dir: Optional measurement-side directory for client-measurement file
        transport.
    :type file_transport_dir: str

    """

    #: Data type for this measurement.
    DATA = BasicMeasData

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.state = BinaryState.IDLE

        self.init_file_transport()

        self.cli = MultiInstrumentClient(
            gconf,
            self.conf["target"]["servers"],
            inst_remap=self.conf.get("inst_remap"),
            context=self.ctx,
            prefix=self.joined_name(),
        )
        self.add_clients(self.cli)

        self.tweaker_clis: dict[str, TweakSaver] = {}
        for tweaker in self.conf["target"].get("tweakers", []):
            cli = TweakSaver(gconf, tweaker, context=self.ctx, prefix=self.joined_name())
            self.tweaker_clis[tweaker] = cli
            self.add_clients(cli)

        self.add_rep()
        self.status_pub = self.add_pub(b"status")
        self.data_pub = self.add_pub(b"data")
        self.buffer_pub = self.add_pub(b"buffer")

    def change_state(self, msg: StateReq) -> Reply:
        """Change state to msg.state. Inherited class must implement this."""

        return Reply(False, "change_state() is not implemented.")

    def get_param_dict_labels(self, msg: P.GetParamDictLabelsReq) -> Reply:
        """Get parameter dict labels. Inherited class must implement this."""

        return Reply(False, "get_param_dict_labels() is not implemented.")

    def get_param_dict(self, msg: P.GetParamDictReq) -> Reply:
        """Get parameter dict. Inherited class must implement this."""

        return Reply(False, "get_param_dict() is not implemented.")

    def save_data(self, msg: SaveDataReq) -> Reply:
        """Save data. Inherited class must implement this."""

        return Reply(False, "save_data() is not implemented.")

    def export_data(self, msg: ExportDataReq) -> Reply:
        """Export data. Inherited class must implement this."""

        return Reply(False, "export_data() is not implemented.")

    def load_data(self, msg: LoadDataReq) -> Reply:
        """Load data. Inherited class must implement this."""

        return Reply(False, "load_data() is not implemented.")

    def _save_data_for_artifact(self, msg: SaveDataReq) -> Reply:
        """Save synchronously for artifact publication; override asynchronous implementations."""

        return self.save_data(msg)

    def save_artifact(self, msg: SaveArtifactReq) -> Reply:
        """Atomically create a saved-data transport artifact."""

        return self.publish_artifact(
            msg.file_name,
            "save",
            lambda path: self._save_data_for_artifact(
                SaveDataReq(path, params=msg.params, note=msg.note)
            ),
            "Failed to save artifact",
        )

    def export_artifact(self, msg: ExportArtifactReq) -> Reply:
        """Atomically create an exported-data transport artifact."""

        return self.publish_artifact(
            msg.file_name,
            "export",
            lambda path: self.export_data(ExportDataReq(path, data=msg.data, params=msg.params)),
            "Failed to export artifact",
        )

    def load_artifact(self, msg: LoadArtifactReq) -> Reply:
        """Load data from a validated shared artifact."""

        return self.consume_artifact(
            msg.artifact,
            lambda path: self.load_data(
                LoadDataReq(
                    path,
                    to_buffer=msg.to_buffer,
                    buffer_name=os.path.basename(msg.file_name),
                )
            ),
            "Failed to load artifact",
        )

    def pop_buffer(self, msg: PopBufferReq) -> Reply:
        """Pop data from the buffer. Inherited class should have attribute buffer: Buffer."""

        if not hasattr(self, "buffer"):
            return Reply(False, "Buffer is not supported.")
        try:
            return Reply(True, ret=self.buffer.pop(msg.index))
        except IndexError:
            return Reply(False, f"Failed to pop buffer (i={msg.index})")

    def clear_buffer(self, msg: ClearBufferReq) -> Reply:
        """Clear the data buffer. Inherited class should have attribute buffer: Buffer."""

        if not hasattr(self, "buffer"):
            return Reply(False, "Buffer is not supported.")
        self.buffer.clear()
        return Reply(True)

    def fit(self, msg: FitReq) -> Reply:
        """Fit data. Inherited class should have attribute worker: Worker."""

        if msg.data_index == -1:
            data = self.worker.data_msg()
        else:
            data = self.buffer.get_data(msg.data_index)
        if not isinstance(data, self.DATA) or not data.has_data():
            return Reply(False, message="Data is invalid / not ready")

        try:
            ret = self.fitter.fitd(data, msg.params, msg.label)
        except Exception:
            self.logger.exception("Failed to fit")
            return Reply(False, message="Failed to fit")
        return Reply(True, ret=ret)

    def clear_fit(self, msg: ClearFitReq) -> Reply:
        """Clear fit data. Inherited class should have attribute worker: Worker."""

        if msg.data_index == -1:
            data = self.worker.data_msg()
        else:
            data = self.buffer.get_data(msg.data_index)
        if not isinstance(data, self.DATA):
            return Reply(False, message="Data is invalid")

        data.remove_fit_data()
        return Reply(True)

    def _handle_req(self, msg: Request) -> Reply:
        """Handle basic Requests.

        Handles following:
        - change_state()
        - get_param_dict(), get_param_dict_labels()
        - save_data() ,export_data(), load_data()
        - pop_buffer(), clear_buffer()
        - fit(), clear_fit()

        """

        try:
            if isinstance(msg, StateReq):
                return self.change_state(msg)
            elif isinstance(msg, P.GetParamDictReq):
                return self.get_param_dict(msg)
            elif isinstance(msg, P.GetParamDictLabelsReq):
                return self.get_param_dict_labels(msg)
            elif isinstance(msg, SaveDataReq):
                return self.save_data(msg)
            elif isinstance(msg, ExportDataReq):
                return self.export_data(msg)
            elif isinstance(msg, LoadDataReq):
                return self.load_data(msg)
            elif isinstance(msg, SaveArtifactReq):
                return self.save_artifact(msg)
            elif isinstance(msg, ExportArtifactReq):
                return self.export_artifact(msg)
            elif isinstance(msg, LoadArtifactReq):
                return self.load_artifact(msg)
            elif isinstance(msg, CleanupArtifactReq):
                return self.cleanup_artifact(msg)
            elif isinstance(msg, PopBufferReq):
                return self.pop_buffer(msg)
            elif isinstance(msg, ClearBufferReq):
                return self.clear_buffer(msg)
            elif isinstance(msg, FitReq):
                return self.fit(msg)
            elif isinstance(msg, ClearFitReq):
                return self.clear_fit(msg)
            else:
                return self.handle_req(msg)
        except Exception:
            msg = "Exception raised while handling request."
            self.logger.exception(msg)
            return Reply(False, msg)

    def handle_req(self, msg: Request) -> Reply:
        """Handle Request other than basic requests.

        basic requests are: change_state(), get_param_dict()
        {save,export,load}_data(), and {pop,clear}_buffer().

        """

        return Reply(False, "Invalid message type")
