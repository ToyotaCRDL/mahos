#!/usr/bin/env python3

"""
Common implementations for meas nodes.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from ..msgs.common_msgs import BinaryStatus, BinaryState, StateReq, Request, Resp
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.common_meas_msgs import PopBufferReq, ClearBufferReq, FitReq, ClearFitReq
from ..msgs.common_meas_msgs import Buffer, BasicMeasData
from ..msgs import param_msgs as P
from ..inst.server import MultiInstrumentClient
from ..node.node import Node, NodeName
from ..node.client import NodeClient, StateClientMixin
from ..node.comm import Context


class BasicMeasClientBase(NodeClient):
    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        prefix: str | None = None,
        status_handler=None,
        data_handler=None,
        buffer_handler=None,
    ):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self._get_status, self._get_data, self._get_buffer = self.add_sub(
            [(b"status", status_handler), (b"data", data_handler), (b"buffer", buffer_handler)]
        )

        self.req = self.add_req(gconf)

    def get_status(self) -> BinaryStatus | None:
        return self._get_status()

    def get_data(self) -> BasicMeasData:
        return self._get_data()

    def get_buffer(self) -> Buffer:
        return self._get_buffer()


class ParamDictReqMixin(object):
    """Implements get_param_dict_names() and get_param_dict()."""

    def get_param_dict_names(self, group: str = "") -> list[str]:
        """Get list of names of available ParamDicts pertaining to `group`.

        :param group: measurement method group name.
                      can be empty if target node provides only one group.

        """

        resp = self.req.request(P.GetParamDictNamesReq(group))
        if resp.success:
            return resp.ret
        else:
            return []

    def get_param_dict(
        self, name: str = "", group: str = ""
    ) -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for a measurement method `name`.

        :param name: measurement method name.
                     can be empty if target node provides only one method.
        :param group: measurement method group name.
                      can be empty if target node provides only one group.

        """

        resp = self.req.request(P.GetParamDictReq(name, group))
        if resp.success:
            return resp.ret
        else:
            return None


class BaseMeasClientMixin(StateClientMixin, ParamDictReqMixin):
    """Implements change_state(), get_state(), get_param_dict_names(), and get_param_dict()."""

    pass


class BasicMeasReqMixin(object):
    """implements start(), stop(), {save,export,load}_data(), and {pop,clear}_buffer()."""

    def start(self, params=None) -> bool:
        """Start the measurement, i.e., change state to ACTIVE."""

        return self.change_state(BinaryState.ACTIVE, params=params)

    def stop(self, params=None) -> bool:
        """Stop the measurement, i.e., change state to IDLE."""

        return self.change_state(BinaryState.IDLE, params=params)

    def save_data(self, file_name: str, params: dict | None = None, note: str = "") -> bool:
        """Save data to `file_name`."""

        resp = self.req.request(SaveDataReq(file_name, params=params, note=note))
        return resp.success

    def export_data(
        self,
        file_name: str,
        data: BasicMeasData | list[BasicMeasData] | None = None,
        params: dict | None = None,
    ) -> bool:
        """Export data to `file_name`.

        :param data: If None, data holded by worker is used.

        """

        resp = self.req.request(ExportDataReq(file_name, data=data, params=params))
        return resp.success

    def load_data(self, file_name: str, to_buffer: bool = False) -> BasicMeasData | None:
        """Load data from `file_name`.

        :param to_buffer: True: load data to buffer. False: overwrite current data with loaded one.

        """

        resp = self.req.request(LoadDataReq(file_name, to_buffer=to_buffer))
        if resp.success:
            return resp.ret
        else:
            return None

    def pop_buffer(self, index: int = -1) -> BasicMeasData | None:
        resp = self.req.request(PopBufferReq(index))
        if resp.success:
            return resp.ret
        else:
            return None

    def clear_buffer(self) -> bool:
        resp = self.req.request(ClearBufferReq())
        return resp.success

    def fit(self, params: dict, data_index=-1) -> Resp:
        """Fit data."""

        return self.req.request(FitReq(params, data_index))

    def clear_fit(self, data_index=-1) -> bool:
        """Clear fit data."""

        resp = self.req.request(ClearFitReq(data_index))
        return resp.success


class BasicMeasClient(BasicMeasClientBase, BaseMeasClientMixin, BasicMeasReqMixin):
    """Client for Basic Measurement Nodes (Node with the BinaryState, Data, and Buffer).

    Client with a subscriber to status (with state inside) & data, and a requester.

    """

    # overrides for annotations

    def get_state(self) -> BinaryState | None:
        s = self.get_status()
        return s.state if s is not None else None


class BasicMeasNode(Node):
    """Base Implementation for Basic Meas Nodes (Node with the BinaryState, Data, and Buffer).

    implements initialization (client and communication), change_state() and get_param_dict().

    """

    #: Data type for this measurement.
    DATA = BasicMeasData

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.state = BinaryState.IDLE

        self.cli = MultiInstrumentClient(
            gconf, self.conf["target"]["servers"], context=self.ctx, prefix=self.joined_name()
        )
        self.add_client(self.cli)
        self.add_rep()
        self.status_pub = self.add_pub(b"status")
        self.data_pub = self.add_pub(b"data")
        self.buffer_pub = self.add_pub(b"buffer")

    def change_state(self, msg: StateReq) -> Resp:
        """Change state to msg.state. Inherited class must implement this."""

        return Resp(False, "change_state() is not implemented.")

    def get_param_dict_names(self, msg: P.GetParamDictNamesReq) -> Resp:
        """Get parameter dict names. Inherited class must implement this."""

        return Resp(False, "get_param_dict_names() is not implemented.")

    def get_param_dict(self, msg: P.GetParamDictReq) -> Resp:
        """Get parameter dict. Inherited class must implement this."""

        return Resp(False, "get_param_dict() is not implemented.")

    def save_data(self, msg: SaveDataReq) -> Resp:
        """Save data. Inherited class must implement this."""

        return Resp(False, "save_data() is not implemented.")

    def export_data(self, msg: ExportDataReq) -> Resp:
        """Export data. Inherited class must implement this."""

        return Resp(False, "export_data() is not implemented.")

    def load_data(self, msg: LoadDataReq) -> Resp:
        """Load data. Inherited class must implement this."""

        return Resp(False, "load_data() is not implemented.")

    def pop_buffer(self, msg: PopBufferReq) -> Resp:
        """Pop data from the buffer. Inherited class should have attribute buffer: Buffer."""

        if not hasattr(self, "buffer"):
            return Resp(False, "Buffer is not supported.")
        try:
            return Resp(True, ret=self.buffer.pop(msg.index))
        except IndexError:
            return Resp(False, f"Failed to pop buffer (i={msg.index})")

    def clear_buffer(self, msg: ClearBufferReq) -> Resp:
        """Clear the data buffer. Inherited class should have attribute buffer: Buffer."""

        if not hasattr(self, "buffer"):
            return Resp(False, "Buffer is not supported.")
        self.buffer.clear()
        return Resp(True)

    def fit(self, msg: FitReq) -> Resp:
        """Fit data. Inherited class should have attribute worker: Worker."""

        if msg.data_index == -1:
            data = self.worker.data_msg()
        else:
            data = self.buffer.get_data(msg.data_index)
        if not isinstance(data, self.DATA) or not data.has_data():
            return Resp(False, message="Data is invalid / not ready")

        try:
            ret = self.fitter.fitd(data, msg.params)
        except Exception:
            self.logger.exception("Failed to fit")
            return Resp(False, message="Failed to fit")
        return Resp(True, ret=ret)

    def clear_fit(self, msg: ClearFitReq) -> Resp:
        """Clear fit data. Inherited class should have attribute worker: Worker."""

        if msg.data_index == -1:
            data = self.worker.data_msg()
        else:
            data = self.buffer.get_data(msg.data_index)
        if not isinstance(data, self.DATA):
            return Resp(False, message="Data is invalid")

        data.clear_fit_data()
        return Resp(True)

    def _handle_req(self, msg: Request) -> Resp:
        """Handle basic Requests.

        Handles following:
        - change_state()
        - get_param_dict(), get_param_dict_names()
        - save_data() ,export_data(), load_data()
        - pop_buffer(), clear_buffer()
        - fit(), clear_fit()

        """

        try:
            if isinstance(msg, StateReq):
                return self.change_state(msg)
            elif isinstance(msg, P.GetParamDictReq):
                return self.get_param_dict(msg)
            elif isinstance(msg, P.GetParamDictNamesReq):
                return self.get_param_dict_names(msg)
            elif isinstance(msg, SaveDataReq):
                return self.save_data(msg)
            elif isinstance(msg, ExportDataReq):
                return self.export_data(msg)
            elif isinstance(msg, LoadDataReq):
                return self.load_data(msg)
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
            return Resp(False, msg)

    def handle_req(self, msg: Request) -> Resp:
        """Handle Request other than basic requests.

        basic requests are: change_state(), get_param_dict()
        {save,export,load}_data(), and {pop,clear}_buffer().

        """

        return Resp(False, "Invalid message type")
