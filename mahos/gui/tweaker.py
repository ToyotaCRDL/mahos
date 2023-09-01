#!/usr/bin/env python3

"""
GUI frontend for Tweaker.

"""

from __future__ import annotations

from .ui.tweaker import Ui_TweakerWidget

from ..msgs.tweaker_msgs import TweakerStatus
from ..node.param_server import ParamClient
from .common_widget import ClientTopWidget
from .tweaker_client import QTweakerClient
from .param import ParamTable
from ..node.node import local_conf, join_name
from .gui_node import GUINode
from .dialog import save_dialog, load_dialog


class TweakerWidget(ClientTopWidget, Ui_TweakerWidget):
    """Top widget for TweakerGUI."""

    def __init__(self, gconf: dict, name, param_server_name, context, parent=None):
        ClientTopWidget.__init__(self, parent)
        self.setupUi(self)
        self.setWindowTitle(f"MAHOS.TweakerGUI ({join_name(name)})")

        self.cli = QTweakerClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.param_cli = ParamClient(gconf, param_server_name, context=context)

        self.add_clients(self.cli, self.param_cli)

        self.setEnabled(False)

    def init_with_status(self, status: TweakerStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.tabWidget.clear()
        self.param_dict_names = status.param_dict_names

        for pd_name in status.param_dict_names:
            pt = ParamTable(parent=self)
            self.tabWidget.addTab(widget=pt, label=pd_name)

            d = self.cli.read(pd_name)
            if d is not None:
                pt.update_contents(d)

        self.init_connections(self)

        self.setEnabled(True)

    def init_connections(self):
        self.readallButton.clicked.connect(self.request_read_all)
        self.readButton.clicked.connect(self.request_read)
        self.writeButton.clicked.connect(self.request_write)
        self.saveButton.clicked.connect(self.request_save)
        self.loadButton.clicked.connect(self.request_load)

    def update_all(self, param_dicts):
        if param_dicts is None:
            return
        for i, name in enumerate(self.param_dict_names):
            if name in param_dicts and param_dicts[name] is not None:
                self.tabWidget.widget(i).update_contents(param_dicts[name])

    def request_read_all(self):
        self.update_all(self.cli.read_all())

    def request_read(self):
        i = self.tabWidget.currentIndex()
        d = self.cli.read(self.param_dict_names[i])
        if d is not None:
            self.tabWidget.currentWidget().update_contents(d)

    def request_write(self):
        i = self.tabWidget.currentIndex()
        pd_name = self.param_dict_names[i]
        params = self.tabWidget.currentWidget().params()
        self.cli.write(pd_name, params)

    def request_save(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Tweaker", ".tweak")
        if not fn:
            return

        self.cli.save(fn)

    def request_load(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "Tweaker", ".tweak")
        if not fn:
            return

        self.update_all(self.cli.load(fn))


class TweakerGUI(GUINode):
    """GUINode for BasicMeasNode using BasicMeasWidget."""

    def init_widget(self, gconf: dict, name, context):
        target = local_conf(gconf, name)["target"]
        return TweakerWidget(gconf, target["tweaker"], target["param_server"], context)
