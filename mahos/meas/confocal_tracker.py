#!/usr/bin/env python3

"""
The ConfocalTracker logic for position-tracking over confocal.Confocal.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from os import path
import uuid
import copy
import sys
import pickle
import time
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    print("mahos.meas.confocal_tracker: failed to import cv2")

from .. import CONFIG_DIR
from .confocal import ConfocalClient
from ..msgs.common_msgs import Reply, StateReq, BinaryState
from ..msgs import confocal_msgs
from ..msgs.confocal_msgs import Image, ScanDirection, ScanMode, PiezoPos
from ..msgs.confocal_msgs import ConfocalState
from ..msgs.confocal_msgs import direction_to_labels, direction_to_axes
from ..msgs.confocal_tracker_msgs import OptMode, ConfocalTrackerStatus
from ..msgs.confocal_tracker_msgs import SaveParamsReq, LoadParamsReq, TrackNowReq
from ..msgs import confocal_tracker_msgs
from ..node.node import Node
from ..node.client import StateClient
from ..util.timer import OneshotTimer
from ..util import conv, fit
from .state_manager import StateManagerClient


def optimize_pos(mode: OptMode, data, prev_data, print_fn=print):
    """Find optimized position of scan data (2D array).

    If optimization is successful, return optimized position (x, y, diff).
    If either of optimized values (x or y) is invalid, substituted with None.

    If optimized position is the differential one, diff is set True.
    """

    if (mode == OptMode.POC) and ("cv2" in sys.modules):
        if prev_data is None or prev_data.shape != data.shape:
            return (None, None, True)

        (x, y), response = cv2.phaseCorrelate(data, prev_data)

        print_fn("Phase Only Correlation x: {} y: {} response: {}".format(x, y, response))

        # Optimization fail if translation is too large or correlation is too low.
        if abs(x) > data.shape[0] * 0.4 or abs(y) > data.shape[1] * 0.4:  # or response < 0.1:
            return (None, None, True)

        return (x, y, True)

    elif mode == OptMode.Gauss2D:
        # Background subtraction. Background is now 20th percentile.
        _sorted = np.sort(data, axis=None)
        bg = _sorted[int(round(len(_sorted) * 0.2))]
        # bg = np.mean(_sorted[:int(round(len(_sorted) * 0.2))])
        print_fn("2D Gaussian fit. Estimated BG: {}".format(bg))

        p, ier = fit.fit_gaussian2d(data - bg)
        height, x, y, width_x, width_y = p

        xl, yl = data.shape

        if ier in [1, 2, 3, 4] and 0 <= x <= xl and 0 <= y <= yl:
            return (x, y, False)
        else:
            return (None, None, False)

    elif (
        mode == OptMode.Gauss1D_0
    ):  # infact, calculate expectation value when data is regarded as probability distribution
        data = np.mean(data, axis=1)

        X = np.arange(data.size)
        x = sum(X * data) / sum(data)

        return (x, None, False)

    elif (
        mode == OptMode.Gauss1D_1
    ):  # infact, calculate expectation value when data is regarded as probability distribution
        data = np.mean(data, axis=0)

        X = np.arange(data.size)
        x = sum(X * data) / sum(data)

        return (None, x, False)

    elif mode == OptMode.Max2D:
        x, y = np.unravel_index(np.argmax(data), data.shape)

        return x, y, False

    elif mode == OptMode.Max1D_0:
        data = np.mean(data, axis=1)

        return (np.argmax(data), None, False)

    elif mode == OptMode.Max1D_1:
        data = np.mean(data, axis=0)

        return (None, np.argmax(data), False)

    else:
        raise ValueError("Invalid mode " + str(mode))


class ImageBuffer(object):
    def __init__(self):
        self.xy_params = None
        self.xy_img = None
        self.xz_params = None
        self.xz_img = None
        self.yz_params = None
        self.yz_img = None
        self.pos = None

    def push_pos(self, pos: PiezoPos):
        self.pos = pos

    def push(self, scan_params: dict, image: Image, pos: PiezoPos):
        if scan_params["direction"] == ScanDirection.XY:
            self.xy_params = scan_params
            self.xy_img = image
        elif scan_params["direction"] == ScanDirection.XZ:
            self.xz_params = scan_params
            self.xz_img = image
        elif scan_params["direction"] == ScanDirection.YZ:
            self.yz_params = scan_params
            self.yz_img = image
        self.pos = pos

    def get(self, scan_params: dict):
        if scan_params["direction"] == ScanDirection.XY:
            return (
                self.xy_params,
                self.xy_img,
                (None, None) if self.pos is None else (self.pos.x_tgt, self.pos.y_tgt),
            )
        elif scan_params["direction"] == ScanDirection.XZ:
            return (
                self.xz_params,
                self.xz_img,
                (None, None) if self.pos is None else (self.pos.x_tgt, self.pos.z_tgt),
            )
        elif scan_params["direction"] == ScanDirection.YZ:
            return (
                self.yz_params,
                self.yz_img,
                (None, None) if self.pos is None else (self.pos.y_tgt, self.pos.z_tgt),
            )


class ConfocalTrackerClient(StateClient):
    """Simple ConfocalTracker Client."""

    #: The module that defines message types for target node (ConfocalTracker).
    M = confocal_tracker_msgs

    #: The module that defines message types for Confocal.
    MC = confocal_msgs

    def save_params(self, params: dict, file_name: Optional[str] = None) -> bool:
        rep = self.req.request(SaveParamsReq(params, file_name=file_name))
        return rep.success

    def load_params(self, file_name: Optional[str] = None) -> Optional[dict]:
        rep = self.req.request(LoadParamsReq(file_name))
        if rep.success:
            return rep.ret
        else:
            return None

    def track_now(self) -> bool:
        rep = self.req.request(TrackNowReq())
        return rep.success


class ConfocalTracker(Node):
    CLIENT = ConfocalTrackerClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.state = BinaryState.IDLE

        self.cli = ConfocalClient(gconf, self.conf["target"]["confocal"], context=self.ctx)
        self.sm_cli = StateManagerClient(gconf, self.conf["target"]["manager"], context=self.ctx)
        self.add_clients(self.cli)
        self.add_clients(self.sm_cli)

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

        self.track_params = self.scan_params = self.timer = self.img_buf = None
        self._prev_confocal_state = None
        self.idx = 0
        self.tracking = False

    def wait(self):
        self.logger.info("Waiting for Confocal and StateManager...")
        self.cli.wait()
        self.sm_cli.wait()
        self.logger.info("Confocal and StateManager are up!")

    def change_state(self, msg: StateReq):
        if self.state == msg.state:
            return Reply(True, "Already in that state")

        # TODO: resolve states of other meas modules
        # if condition is not met, Reply(False "Couldn't change state")

        if msg.state == BinaryState.IDLE:
            self.stop()
        elif msg.state == BinaryState.ACTIVE:
            self.start(msg.params)

        self.state = msg.state
        return Reply(True)

    def track_now(self, msg: TrackNowReq) -> Reply:
        if self.state == BinaryState.IDLE:
            return self.fail_with("TrackNow rejected: tracker is IDLE.")
        if self.timer is None:
            return self.fail_with("TrackNow rejected: timer is None.")

        self.timer.force_activation()
        return Reply(True)

    def _default_params_file(self):
        return path.join(CONFIG_DIR, "confocal_tracker_params.pkl")

    def save_params(self, msg: SaveParamsReq):
        fn = self._default_params_file() if msg.file_name is None else msg.file_name
        try:
            with open(fn, "wb") as f:
                pickle.dump(msg.params, f)
            self.logger.info("Saved {}".format(fn))
            return Reply(True)
        except IOError:
            msg = "Cannot save {}".format(fn)
            self.logger.exception(fn)
            return Reply(False, msg)

    def load_params(self, msg: LoadParamsReq):
        fn = self._default_params_file() if msg.file_name is None else msg.file_name
        if not path.exists(fn):
            msg = "Params file doesn't exist: {}".format(fn)
            self.logger.info(msg)
            return Reply(False, msg)

        try:
            with open(fn, "rb") as f:
                params = pickle.load(f)
            self.logger.info("Loaded {}".format(fn))
            return Reply(True, ret=params)
        except IOError:
            msg = "Cannot load {}".format(fn)
            self.logger.exception(msg)
            return Reply(False, msg)

    def handle_req(self, msg):
        if isinstance(msg, StateReq):
            return self.change_state(msg)
        elif isinstance(msg, TrackNowReq):
            return self.track_now(msg)
        elif isinstance(msg, SaveParamsReq):
            return self.save_params(msg)
        elif isinstance(msg, LoadParamsReq):
            return self.load_params(msg)
        else:
            return Reply(False, "Invalid message type")

    def main(self):
        self.poll()

        if self.state == BinaryState.ACTIVE:
            self._work()
        self._publish()

    def _publish(self):
        status = ConfocalTrackerStatus(state=self.state, tracking=self.tracking)
        self.status_pub.publish(status)

    def _work(self):
        if not self.timer.check():
            self.tracking = False
            return
        self.tracking = True

        image = self.cli.get_image()
        pos = self.cli.get_status().pos

        if self.idx == 0:
            self.img_buf.push_pos(pos)
            if self.prepare_states() and self.request_scan(pos):
                self.idx += 1
            else:
                self.logger.error("Failed to start scan. Restarting.")
                self.restart(True)
            return

        # finished scanning
        if (
            isinstance(image, Image)
            and self.scan_params["ident"] == image.ident
            and not image.running
        ):
            self.optimize(image)
            pos = self.get_new_pos()
            self.img_buf.push(self.scan_params, image, pos)

            # all scans are finished
            if self.idx >= len(self.track_params["order"]):
                self.restore_states()
                self.restart(False)
                return

            # start next scan
            if self.request_scan(pos):
                self.idx += 1
            else:
                self.logger.error("Failed to start scan. Restarting.")
                self.restart(True)

    def optimize(self, image: Image):
        def _fmt_float_opt(v):
            return "None" if v is None else f"{v:.4f}"

        def _clip(val, vmin, vmax):
            return vmin if val < vmin else vmax if val > vmax else val

        prev_conf, prev_img, (px, py) = self.img_buf.get(self.scan_params)
        if np.all(image.image == 0.0) or np.all(
            np.isclose(image.image, image.image[0, 0], atol=0.0)
        ):
            # All-equal image can appear when detector power is off.
            self.logger.error("Image is invalid. Skipping optimization.")
            self.request_move(direction_to_axes(image.direction), (px, py))
            return

        if prev_conf is not None:
            prev_x0, prev_y0 = prev_conf["xmin"], prev_conf["ymin"]
        else:
            prev_x0, prev_y0 = None, None
        prev_data = None if prev_img is None else prev_img.image
        xi, yi, diff = optimize_pos(
            self.scan_params["opt_mode"], image.image, prev_data, print_fn=self.logger.debug
        )
        x0, y0 = self.scan_params["xmin"], self.scan_params["ymin"]
        xs = conv.num_to_step(
            self.scan_params["xmin"], self.scan_params["xmax"], self.scan_params["xnum"]
        )
        ys = conv.num_to_step(
            self.scan_params["ymin"], self.scan_params["ymax"], self.scan_params["ynum"]
        )
        _xlabel, _ylabel = direction_to_labels(image.direction)

        xofs = self.scan_params.get("xoffset", 0.0)
        yofs = self.scan_params.get("yoffset", 0.0)
        if xi is None:
            nx = px
        else:
            if diff:
                nx = x0 - xs * xi + (px - prev_x0)
            else:
                nx = x0 + xs * xi
            nx = _clip(nx + xofs, x0, self.scan_params["xmax"])

        if yi is None:
            ny = py
        else:
            if diff:
                ny = y0 - ys * yi + (py - prev_y0)
            else:
                ny = y0 + ys * yi
            ny = _clip(ny + yofs, y0, self.scan_params["ymax"])

        _px, _py, _nx, _ny = [_fmt_float_opt(v) for v in (px, py, nx, ny)]
        _dx, _dy = [_fmt_float_opt(v) for v in (nx - px, ny - py)]
        msg = f"Optimizing {_xlabel}, {_ylabel} from {_px}, {_py} to {_nx}, {_ny}."
        msg += f" Delta {_dx}, {_dy}. Offsets: {xofs:.4f}, {yofs:.4f}."
        self.logger.info(msg)
        self.request_move(direction_to_axes(image.direction), (nx, ny))

    def request_move(self, axes, pos):
        self.cli.change_state(ConfocalState.PIEZO)
        _axes = []
        _pos = []
        for ax, p in zip(axes, pos):
            if p is not None:
                _axes.append(ax)
                _pos.append(p)
        if _axes:
            self.cli.move(_axes, _pos)

    def get_new_pos(self):
        """Get updated pos which reflects the move by previous optimization."""

        pos = self.cli.get_status().pos
        for i in range(300):
            new_pos = self.cli.get_status().pos
            if new_pos is not pos:
                return new_pos
            time.sleep(0.01)
        self.logger.error("Failed to get new pos.")
        return pos

    def get_z(self, d: ScanDirection, pos: PiezoPos):
        if d == ScanDirection.XY:
            return pos.z_tgt
        if d == ScanDirection.XZ:
            return pos.y_tgt
        if d == ScanDirection.YZ:
            return pos.x_tgt

    def get_bounds(self, d: ScanDirection, pos: PiezoPos):
        xl, yl = self.scan_params["xlen"], self.scan_params["ylen"]
        if d == ScanDirection.XY:
            x, y = pos.x_tgt, pos.y_tgt
        if d == ScanDirection.XZ:
            x, y = pos.x_tgt, pos.z_tgt
        if d == ScanDirection.YZ:
            x, y = pos.y_tgt, pos.z_tgt

        xb = x - xl / 2.0, x + xl / 2.0
        yb = y - yl / 2.0, y + yl / 2.0

        if d == ScanDirection.YZ:
            xmn, xmx = pos.y_range
        else:
            xmn, xmx = pos.x_range
        if d == ScanDirection.XY:
            ymn, ymx = pos.y_range
        else:
            ymn, ymx = pos.z_range

        if xb[0] < xmn:
            xb = xmn, xmn + xl
        elif xb[1] > xmx:
            xb = xmx - xl, xmx
        if yb[0] < ymn:
            yb = ymn, ymn + yl
        elif yb[1] > ymx:
            yb = ymx - yl, ymx

        return xb, yb

    def request_scan(self, pos) -> bool:
        d = self.track_params["order"][self.idx]
        self.scan_params = copy.copy(self.track_params[d])

        mode = self.track_params["mode"]
        self.scan_params["mode"] = mode
        self.scan_params["line_mode"] = self.track_params["line_mode"]
        if mode == ScanMode.ANALOG:
            self.scan_params["dummy_samples"] = self.track_params["dummy_samples"]
        if mode == ScanMode.COM_DIPOLL:
            self.scan_params["poll_samples"] = self.track_params["poll_samples"]
        if mode != ScanMode.ANALOG:
            self.scan_params["delay"] = self.track_params["delay"]

        self.scan_params["z"] = self.get_z(d, pos)
        xb, yb = self.get_bounds(d, pos)
        self.scan_params["xmin"], self.scan_params["xmax"] = xb
        self.scan_params["ymin"], self.scan_params["ymax"] = yb

        self.scan_params["direction"] = d
        self.scan_params["time_window"] = self.track_params["time_window"]
        self.scan_params["ident"] = uuid.uuid4()

        # delete opt_mode here because it causes error when saving confocal image
        p = copy.copy(self.scan_params)
        del p["opt_mode"]
        return self.cli.change_state(ConfocalState.SCAN, params=p)

    def prepare_states(self) -> bool:
        if not self.sm_cli.command("prepare_scan"):
            self.logger.error("Failed to execute prepare_scan command.")
            return False
        s = self.cli.get_status().state
        if s == ConfocalState.SCAN:
            self.logger.error("Cannot start tracking because another scan is running.")
            return False

        self._prev_confocal_state = s
        self.logger.info("Store confocal state {}".format(s))
        return True

    def restore_states(self):
        if (
            isinstance(self._prev_confocal_state, ConfocalState)
            and self._prev_confocal_state != ConfocalState.SCAN
        ):
            self.cli.change_state(self._prev_confocal_state)
        self.sm_cli.restore("prepare_scan")

    def finalize_states(self):
        s = self.cli.get_status().state
        if (
            isinstance(self._prev_confocal_state, ConfocalState)
            and self._prev_confocal_state != ConfocalState.SCAN
        ):
            self.cli.change_state(self._prev_confocal_state)

        # invoke restore of manager only when scan is ongoing.
        # (the other nodes are interrupted.)
        if s == ConfocalState.SCAN:
            self.sm_cli.restore("prepare_scan")

    def restart(self, clear_buffer: bool):
        interval = self.track_params["interval_sec"]
        if interval == 0.0:
            # Setting 0.0 interval means infinite interval.
            # In this case, tracking is invoked only by TrackNowReq.
            interval = float("inf")
            self.logger.info("Resetting timer. Infinite interval (manual start only).")
        else:
            t = time.strftime("%H:%M:%S", time.localtime(time.time() + interval))
            self.logger.info(f"Resetting timer. Next track starts at {t}.")

        self.timer = OneshotTimer(interval)
        self.scan_params = None
        if clear_buffer:
            self.img_buf = ImageBuffer()
        self.idx = 0
        self.tracking = False

    def start(self, track_params):
        self.track_params = track_params
        self.restart(True)
        self.logger.info("Started waiting for track.")

    def stop(self):
        self.track_params = self.scan_params = self.timer = self.img_buf = None
        self.idx = 0
        self.tracking = False
        self.finalize_states()
        self.logger.info("Stopped waiting for track.")
