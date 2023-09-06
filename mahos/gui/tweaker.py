#!/usr/bin/env python3

"""
GUI frontend for Tweaker.

"""

from __future__ import annotations

from .ui.tweaker import Ui_TweakerWidget

from ..msgs.tweaker_msgs import TweakerStatus
from ..node.global_params import GlobalParamsClient
from .common_widget import ClientTopWidget
from .tweaker_client import QTweakerClient
from .param import ParamTable
from ..node.node import local_conf, join_name
from .gui_node import GUINode
from .dialog import save_dialog, load_dialog


class TweakerWidget(ClientTopWidget, Ui_TweakerWidget):
    """Top widget for TweakerGUI."""

    def __init__(self, gconf: dict, name, gparams_name, context, parent=None):
        ClientTopWidget.__init__(self, parent)
        self.setupUi(self)
        self.setWindowTitle(f"MAHOS.TweakerGUI ({join_name(name)})")

        self.cli = QTweakerClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli, self.gparams_cli)

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
        self.saveButton.clicked.connect(self.request_save)
        self.loadButton.clicked.connect(self.request_load)

    def update_all(self, param_dicts):
        if param_dicts is None:
            return
        for i, pid in enumerate(self.param_dict_ids):
            if pid in param_dicts and param_dicts[pid] is not None:
                self.tabWidget.widget(i).update_contents(param_dicts[pid])

    def request_read_all(self):
        self.update_all(self.cli.read_all())

    def request_read(self):
        i = self.tabWidget.currentIndex()
        d = self.cli.read(self.param_dict_ids[i])
        if d is not None:
            self.tabWidget.currentWidget().update_contents(d)

    def request_write_all(self):
        param_dict = {
            pid: self.tabWidget.widget(i).params() for i, pid in enumerate(self.param_dict_ids)
        }
        self.cli.write_all(param_dict)

    def request_write(self):
        i = self.tabWidget.currentIndex()
        param_dict_id = self.param_dict_ids[i]
        params = self.tabWidget.currentWidget().params()
        self.cli.write(param_dict_id, params)

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
            group = "_inst_params"

        self.update_all(self.cli.load(fn, group))


class TweakerGUI(GUINode):
    """GUINode for BasicMeasNode using BasicMeasWidget."""

    def init_widget(self, gconf: dict, name, context):
        target = local_conf(gconf, name)["target"]
        return TweakerWidget(gconf, target["tweaker"], target["gparams"], context)
