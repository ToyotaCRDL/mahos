#!/usr/bin/env python3

"""
GUI frontend for Tweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from mahos.gui.ui.tweaker import Ui_TweakerWidget

from mahos.msgs.tweaker_msgs import TweakerStatus
from mahos.node.global_params import GlobalParamsClient
from mahos.gui.common_widget import ClientTopWidget
from mahos.gui.tweaker_client import QTweakerClient
from mahos.gui.param import ParamTable
from mahos.node.node import local_conf, join_name
from mahos.gui.gui_node import GUINode, get_gui_logger
from mahos.gui.dialog import save_dialog, load_dialog
from mahos.gui.Qt import QtWidgets


class TweakerWidget(ClientTopWidget, Ui_TweakerWidget):
    """Top widget for TweakerGUI."""

    def __init__(
        self,
        gconf: dict,
        name,
        gparams_name,
        verbose,
        context,
        parent=None,
        logger=None,
        file_transport_dir=None,
    ):
        ClientTopWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)
        self.setupUi(self)
        self.setWindowTitle(f"MAHOS.TweakerGUI ({join_name(name)})")

        self.cli = QTweakerClient(
            gconf, name, context=context, parent=self, file_transport_dir=file_transport_dir
        )
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli, self.gparams_cli)

        self._verbose = verbose
        self._group_name = "__" + name + "__"

        self.setEnabled(False)

    def init_with_status(self, status: TweakerStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.tabWidget.clear()
        self.param_dict_ids = status.param_dict_ids

        for param_dict_id in status.param_dict_ids:
            pt = ParamTable(parent=self)
            self.tabWidget.addTab(pt, param_dict_id)

            d = self.cli.read(param_dict_id)
            if d is not None:
                pt.update_contents(d)

        self.init_connections()

        self.setEnabled(True)

    def init_connections(self):
        self.readallButton.clicked.connect(self.request_read_all)
        self.readButton.clicked.connect(self.request_read)
        self.writeallButton.clicked.connect(self.request_write_all)
        self.writeButton.clicked.connect(self.request_write)
        self.startButton.clicked.connect(self.request_start)
        self.stopButton.clicked.connect(self.request_stop)
        self.resetButton.clicked.connect(self.request_reset)
        self.saveButton.clicked.connect(self.request_save)
        self.loadButton.clicked.connect(self.request_load)

    def update_all(self, param_dicts):
        if param_dicts is None:
            return
        for i, pid in enumerate(self.param_dict_ids):
            if pid in param_dicts and param_dicts[pid] is not None:
                self.tabWidget.widget(i).update_contents(param_dicts[pid])

    def request_read_all(self):
        success, param_dicts = self.cli.read_all()
        if self._verbose and not success:
            QtWidgets.QMessageBox.warning(
                self, "Failed to read.", "Failed to some of the parameters."
            )
        self.update_all(param_dicts)

    def request_read(self):
        i = self.tabWidget.currentIndex()
        param_dict_id = self.param_dict_ids[i]
        d = self.cli.read(param_dict_id)
        if d is not None:
            self.tabWidget.currentWidget().update_contents(d)
        elif self._verbose:
            QtWidgets.QMessageBox.warning(
                self, "Failed to read.", f"Failed to read {param_dict_id}."
            )

    def request_write_all(self):
        param_dict = {
            pid: self.tabWidget.widget(i).params() for i, pid in enumerate(self.param_dict_ids)
        }
        success = self.cli.write_all(param_dict)
        if self._verbose and not success:
            QtWidgets.QMessageBox.warning(
                self, "Failed to write.", "Failed to write some of the parameters."
            )

    def request_write(self):
        i = self.tabWidget.currentIndex()
        param_dict_id = self.param_dict_ids[i]
        params = self.tabWidget.currentWidget().params()
        success = self.cli.write(param_dict_id, params)
        if self._verbose and not success:
            QtWidgets.QMessageBox.warning(
                self, "Failed to write.", f"Failed to write {param_dict_id}."
            )

    def request_start(self):
        i = self.tabWidget.currentIndex()
        param_dict_id = self.param_dict_ids[i]
        success = self.cli.start(param_dict_id)
        if not success and self._verbose:
            QtWidgets.QMessageBox.warning(
                self, "Failed to start.", f"Failed to start {param_dict_id}."
            )

    def request_stop(self):
        i = self.tabWidget.currentIndex()
        param_dict_id = self.param_dict_ids[i]
        success = self.cli.stop(param_dict_id)
        if not success and self._verbose:
            QtWidgets.QMessageBox.warning(
                self, "Failed to stop.", f"Failed to stop {param_dict_id}."
            )

    def request_reset(self):
        i = self.tabWidget.currentIndex()
        param_dict_id = self.param_dict_ids[i]
        success = self.cli.reset(param_dict_id)
        if not success and self._verbose:
            QtWidgets.QMessageBox.warning(
                self, "Failed to reset.", f"Failed to reset {param_dict_id}."
            )

    def request_save(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Tweaker", ".tweak")
        if not fn:
            return

        self.cli.save(fn)

    def request_load(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "Tweaker or measurement", "")
        if not fn:
            return
        if fn.endswith(".tweak.h5"):
            # ParamDicts in individual file for Tweaker
            group = ""
        else:
            # ParamDicts written within measurement Data
            group = self._group_name

        self.update_all(self.cli.load(fn, group))


class TweakerGUI(GUINode):
    """GUINode for TweakerWidget.

    :param target.log: Optional LogBroker target for formal GUI logging.
    :type target.log: tuple[str, str] | str
    :param log_level: (default: ``"INFO"``) Optional logging level.
    :type log_level: str
    :param target.tweaker: Target Tweaker node full name.
    :type target.tweaker: tuple[str, str] | str
    :param target.gparams: Target GlobalParams node full name.
    :type target.gparams: tuple[str, str] | str
    :param verbose: Show warning dialogs when RPC requests fail.
    :type verbose: bool
    :param file_transport_dir: Optional GUI-side directory for shared file transport.
    :type file_transport_dir: str

    """

    def init_widget(self, gconf: dict, name, context):
        lconf = local_conf(gconf, name)
        target = lconf["target"]
        verbose = lconf.get("verbose", True)
        return TweakerWidget(
            gconf,
            target["tweaker"],
            target["gparams"],
            verbose,
            context,
            logger=self.logger,
            file_transport_dir=lconf.get("file_transport_dir"),
        )
