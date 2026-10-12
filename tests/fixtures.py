#!/usr/bin/env python3

"""
Test fixtures.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from os import path
import multiprocessing as mp
import time

import pytest

from mahos.node.node import split_name, load_gconf, local_conf, start_node_proc, Node
from mahos.node.log_broker import LogBroker, LogClient
from mahos.node.global_params import GlobalParams, GlobalParamsClient
from mahos.inst.server import InstrumentServer, InstrumentClient
from mahos_dq.meas.confocal import Confocal, ConfocalClient
from mahos_dq.meas.confocal_tracker import ConfocalTracker, ConfocalTrackerClient
from mahos_dq.meas.odmr import ODMR, ODMRClient
from mahos_dq.meas.podmr import PODMR, PODMRClient
from mahos_dq.meas.spodmr import SPODMR, SPODMRClient
from mahos_dq.meas.apodmr import APODMR, APODMRClient
from mahos_dq.meas.iodmr import IODMR, IODMRClient
from mahos_dq.meas.qdyne import Qdyne, QdyneClient
from mahos_dq.meas.hbt import HBT, HBTClient
from mahos_dq.meas.spectroscopy import Spectroscopy, SpectroscopyClient
from mahos.meas.camera import Camera, CameraClient
from mahos.meas.state_manager import StateManager, StateManagerClient
from mahos.meas.tweaker import Tweaker, TweakerClient
from mahos.meas.pos_tweaker import PosTweaker, PosTweakerClient
from mahos.meas.recorder import Recorder, RecorderClient
from mahos.meas.grid_sweeper import GridSweeper, GridSweeperClient
from mahos.meas.sweeper import Sweeper, SweeperClient


script_dir = path.dirname(path.realpath(__file__))

server_name = "localhost::server"
gparams_name = "localhost::gparams"
log_name = "localhost::log"
manager_name = "localhost::manager"
tweaker_name = "localhost::tweaker"
pos_tweaker_name = "localhost::pos_tweaker"
recorder_name = "localhost::recorder"
sweeper_name = "localhost::sweeper"
grid_sweeper_name = "localhost::grid_sweeper"
confocal_name = "localhost::confocal"
tracker_name = "localhost::tracker"
odmr_name = "localhost::odmr"
podmr_name = "localhost::podmr"
spodmr_name = "localhost::spodmr"
apodmr_name = "localhost::apodmr"
iodmr_name = "localhost::iodmr"
qdyne_name = "localhost::qdyne"
hbt_name = "localhost::hbt"
spectroscopy_name = "localhost::spectroscopy"
camera_name = "localhost::camera"
dummy_name = "localhost::dummy"
dummy_interval_sec = 0.01


def stop_proc(proc, shutdown_ev):
    shutdown_ev.set()
    proc.join(5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
    if proc.is_alive():
        proc.kill()
        proc.join()


class DummyLoggingNode(Node):
    """Dummy Node putting log messages for testing LogBroker."""

    def __init__(self, gconf, name, context=None):
        Node.__init__(self, gconf, name, context=context)
        self.i = 0

    def main(self):
        self.logger.info(str(self.i))
        self.i += 1
        time.sleep(dummy_interval_sec)


class ServerRestarter(object):
    def __init__(self, ctx, gconf):
        self.ctx = ctx
        self.gconf = gconf
        self.start()

    def stop(self):
        stop_proc(self.proc, self.shutdown_ev)
        del self.proc, self.shutdown_ev

    def start(self):
        self.proc, self.shutdown_ev = start_node_proc(
            self.ctx, InstrumentServer, self.gconf, server_name
        )


@pytest.fixture
def ctx():
    return mp.get_context()


@pytest.fixture
def gconf():
    gconf = load_gconf(path.join(script_dir, "conf.toml"))
    # fix timeout and interval to test quickly
    gconf["global"]["req_timeout_ms"] = 3000
    local_conf(gconf, server_name)["poll_timeout_ms"] = 10
    local_conf(gconf, server_name)["instrument"]["pg"]["conf"]["local_dir"] = script_dir
    local_conf(gconf, gparams_name)["poll_timeout_ms"] = 10
    local_conf(gconf, log_name)["poll_timeout_ms"] = 10
    confocal_conf = local_conf(gconf, confocal_name)
    confocal_conf["poll_timeout_ms"] = 50
    confocal_conf["piezo"]["interval_sec"] = 0.01
    confocal_conf["tracer"]["interval_sec"] = 0.01
    local_conf(gconf, tracker_name)["poll_timeout_ms"] = 50
    local_conf(gconf, odmr_name)["poll_timeout_ms"] = 50
    local_conf(gconf, podmr_name)["poll_timeout_ms"] = 50
    local_conf(gconf, spodmr_name)["poll_timeout_ms"] = 50
    local_conf(gconf, apodmr_name)["poll_timeout_ms"] = 50
    local_conf(gconf, iodmr_name)["poll_timeout_ms"] = 50
    local_conf(gconf, qdyne_name)["poll_timeout_ms"] = 50
    local_conf(gconf, hbt_name)["poll_timeout_ms"] = 50
    local_conf(gconf, spectroscopy_name)["poll_timeout_ms"] = 50
    local_conf(gconf, camera_name)["poll_timeout_ms"] = 50
    local_conf(gconf, manager_name)["poll_timeout_ms"] = 50
    local_conf(gconf, tweaker_name)["poll_timeout_ms"] = 50
    local_conf(gconf, pos_tweaker_name)["poll_timeout_ms"] = 50
    local_conf(gconf, recorder_name)["poll_timeout_ms"] = 50
    local_conf(gconf, recorder_name)["collector"]["interval_sec"] = 0.05
    local_conf(gconf, sweeper_name)["poll_timeout_ms"] = 50
    local_conf(gconf, grid_sweeper_name)["poll_timeout_ms"] = 50

    # add conf for dummy
    n = split_name(dummy_name)
    gconf[n[0]][n[1]] = {"target": {"log": log_name}}

    return gconf


@pytest.fixture
def server_conf(gconf):
    return local_conf(gconf, server_name)


@pytest.fixture
def gparams_conf(gconf):
    return local_conf(gconf, gparams_name)


@pytest.fixture
def log_conf(gconf):
    return local_conf(gconf, log_name)


@pytest.fixture
def manager_conf(gconf):
    return local_conf(gconf, manager_name)


@pytest.fixture
def tweaker_conf(gconf):
    return local_conf(gconf, tweaker_name)


@pytest.fixture
def pos_tweaker_conf(gconf):
    return local_conf(gconf, pos_tweaker_name)


@pytest.fixture
def recorder_conf(gconf):
    return local_conf(gconf, recorder_name)


@pytest.fixture
def sweeper_conf(gconf):
    return local_conf(gconf, sweeper_name)


@pytest.fixture
def grid_sweeper_conf(gconf):
    return local_conf(gconf, grid_sweeper_name)


@pytest.fixture
def confocal_conf(gconf):
    return local_conf(gconf, confocal_name)


@pytest.fixture
def tracker_conf(gconf):
    return local_conf(gconf, tracker_name)


@pytest.fixture
def odmr_conf(gconf):
    return local_conf(gconf, odmr_name)


@pytest.fixture
def podmr_conf(gconf):
    return local_conf(gconf, podmr_name)


@pytest.fixture
def spodmr_conf(gconf):
    return local_conf(gconf, spodmr_name)


@pytest.fixture
def apodmr_conf(gconf):
    return local_conf(gconf, apodmr_name)


@pytest.fixture
def iodmr_conf(gconf):
    return local_conf(gconf, iodmr_name)


@pytest.fixture
def qdyne_conf(gconf):
    return local_conf(gconf, qdyne_name)


@pytest.fixture
def hbt_conf(gconf):
    return local_conf(gconf, hbt_name)


@pytest.fixture
def spectroscopy_conf(gconf):
    return local_conf(gconf, spectroscopy_name)


@pytest.fixture
def camera_conf(gconf):
    return local_conf(gconf, camera_name)


@pytest.fixture
def server(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, InstrumentServer, gconf, server_name)
    client = InstrumentClient(gconf, server_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def server_restart(ctx, gconf):
    srv = ServerRestarter(ctx, gconf)
    client = InstrumentClient(gconf, server_name)
    yield srv, client
    client.close()
    srv.stop()


@pytest.fixture
def server_2clients(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, InstrumentServer, gconf, server_name)
    client0 = InstrumentClient(gconf, server_name)
    client1 = InstrumentClient(gconf, server_name)
    yield client0, client1
    client0.close()
    client1.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def global_params(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, GlobalParams, gconf, gparams_name)
    client = GlobalParamsClient(gconf, gparams_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def global_params_2clients(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, GlobalParams, gconf, gparams_name)
    client0 = GlobalParamsClient(gconf, gparams_name)
    client1 = GlobalParamsClient(gconf, gparams_name)
    yield client0, client1
    client0.close()
    client1.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def log_broker(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, LogBroker, gconf, log_name)
    client = LogClient(gconf, log_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def dummy_logging(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, DummyLoggingNode, gconf, dummy_name)
    yield
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def manager(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, StateManager, gconf, manager_name)
    client = StateManagerClient(gconf, manager_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def tweaker(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, Tweaker, gconf, tweaker_name)
    client = TweakerClient(gconf, tweaker_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def pos_tweaker(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, PosTweaker, gconf, pos_tweaker_name)
    client = PosTweakerClient(gconf, pos_tweaker_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def recorder(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, Recorder, gconf, recorder_name)
    client = RecorderClient(gconf, recorder_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def sweeper(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, Sweeper, gconf, sweeper_name)
    client = SweeperClient(gconf, sweeper_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def grid_sweeper(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, GridSweeper, gconf, grid_sweeper_name)
    client = GridSweeperClient(gconf, grid_sweeper_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def confocal(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, Confocal, gconf, confocal_name)
    client = ConfocalClient(gconf, confocal_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def confocal_ev(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, Confocal, gconf, confocal_name)
    client = ConfocalClient(gconf, confocal_name)
    yield client, shutdown_ev
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def tracker(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, ConfocalTracker, gconf, tracker_name)
    client = ConfocalTrackerClient(gconf, tracker_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def odmr(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, ODMR, gconf, odmr_name)
    client = ODMRClient(gconf, odmr_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def odmr_ev(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, ODMR, gconf, odmr_name)
    client = ODMRClient(gconf, odmr_name)
    yield client, shutdown_ev
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def podmr(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, PODMR, gconf, podmr_name)
    client = PODMRClient(gconf, podmr_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def spodmr(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, SPODMR, gconf, spodmr_name)
    client = SPODMRClient(gconf, spodmr_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def apodmr(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, APODMR, gconf, apodmr_name)
    client = APODMRClient(gconf, apodmr_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def iodmr(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, IODMR, gconf, iodmr_name)
    client = IODMRClient(gconf, iodmr_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def qdyne(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, Qdyne, gconf, qdyne_name)
    client = QdyneClient(gconf, qdyne_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def hbt(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, HBT, gconf, hbt_name)
    client = HBTClient(gconf, hbt_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def spectroscopy(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, Spectroscopy, gconf, spectroscopy_name)
    client = SpectroscopyClient(gconf, spectroscopy_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)


@pytest.fixture
def camera(ctx, gconf):
    proc, shutdown_ev = start_node_proc(ctx, Camera, gconf, camera_name)
    client = CameraClient(gconf, camera_name)
    yield client
    client.close()
    stop_proc(proc, shutdown_ev)
