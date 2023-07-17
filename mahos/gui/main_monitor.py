#!/usr/bin/env python3

"""
Main monitor for MAHOS system.

.. This file is a part of MAHOS project.

"""

import os

from .Qt import QtCore, QtWidgets
from .ui.mainMonitor import Ui_MainMonitor

from .gui_node import GUINode
from .common_widget import ClientTopWidget
from ..node.node import local_conf, join_name
from ..node.log_broker import format_log_html_color, should_show, LogEntry
from .param_client import QParamClient
from .log_client import QLogSubscriber
from .inst_client import QInstrumentSubscriber
from .manager_client import QManagerSubscriber
from ..msgs.inst_server_msgs import ServerStatus, Ident
from ..msgs.param_server_msgs import ParamServerStatus


class MainMonitorWidget(ClientTopWidget, Ui_MainMonitor):
    def __init__(self, gconf: dict, name, context):
        ClientTopWidget.__init__(self)
        self.setupUi(self)

        for table in (self.lockTable, self.stateTable):
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.setWindowTitle(f"MAHOS.MainMonitor ({join_name(name)})")

        self.conf = local_conf(gconf, name)
        target = self.conf["target"]

        self.init_log()

        if "param_server" in target:
            self.param_cli = QParamClient(
                gconf, target["param_server"], context=context, parent=self
            )
            self.param_cli.statusUpdated.connect(self.init_param)
            self.add_client(self.param_cli)
        else:
            print("[WARN] param_server is not defined for {}.target".format(name))

        if "log" in target:
            self.log_cli = QLogSubscriber(gconf, target["log"], context=context, parent=self)
            self.log_cli.logArrived.connect(self.write_log)
            self.add_client(self.log_cli)
        else:
            print("[WARN] log is not defined for {}.target".format(name))

        self.inst_clis = {}
        self.locks = {}
        if "servers" in target:
            for name in target["servers"]:
                n = join_name(name)
                cli = QInstrumentSubscriber(gconf, name, context=context, parent=self)
                cli.statusUpdated.connect(self.update_locks)
                self.inst_clis[n] = cli
                self.locks[n] = None
                self.add_client(cli)
        else:
            print("[WARN] servers is not defined for {}.target".format(name))

        if "manager" in target:
            self.manager_cli = QManagerSubscriber(
                gconf, target["manager"], context=context, parent=self
            )
            self.manager_cli.statesUpdated.connect(self.update_states)
            self.add_client(self.manager_cli)
        else:
            print("[WARN] manager is not defined for {}.target".format(name))

    def init_param(self, msg: ParamServerStatus):
        # only once.
        self.param_cli.statusUpdated.disconnect(self.init_param)

        if "work_dir" in self.conf:
            d = os.path.expanduser(self.conf["work_dir"])
        else:
            d = os.path.abspath(".")

        self.pathEdit.setText(d)
        self.param_cli.set_param("work_dir", d)

        self.pathButton.clicked.connect(self.pathDialog)
        self.param_cli.statusUpdated.connect(self.check_param)

        self.init_note()

    def init_log(self):
        self.clearlogButton.clicked.connect(self.logEdit.clear)

    def init_note(self):
        self.noteEdit.commit.connect(self.commit_note)
        self.commitnoteButton.clicked.connect(self.commit_note)
        self.loadnoteButton.clicked.connect(self.load_note)

    def pathDialog(self):
        current = str(self.pathEdit.text())
        dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Change working directory", current)
        if dn and current != dn:
            self.pathEdit.setText(dn)
            self.param_cli.set_param("work_dir", dn)

    def write_log(self, log: LogEntry):
        if not should_show(self.loglevelBox.currentText(), log):
            return
        m = format_log_html_color(log)
        self.logEdit.appendHtml(m)

    def check_param(self, msg: ParamServerStatus):
        d = msg.params.get("work_dir")
        if d is not None:
            self.pathEdit.setText(str(d))

        n = msg.params.get("loaded_note")
        if n is not None and str(n):
            self.loadnoteEdit.setPlainText(str(n))

    def commit_note(self):
        note = self.noteEdit.toPlainText()
        self.commitnoteEdit.setPlainText(note)
        self.param_cli.set_param("note", note)

    def load_note(self):
        self.noteEdit.setPlainText(self.loadnoteEdit.toPlainText())

    def update_locks(self, status: ServerStatus):
        name = join_name((status.host, status.name))
        self.locks[name] = status.locks
        if any([l is None for l in self.locks.values()]):
            return
        if not self.lockTable.rowCount():
            self.create_lock_table()
        else:
            self.update_lock_table(name)

    def update_lock_table(self, name: str):
        row = 0
        for n, locks in self.locks.items():
            if n == name:
                break
            row += len(locks)

        for i, ident in enumerate(self.locks[name].values()):
            if isinstance(ident, Ident):
                self.lockTable.item(row + i, 2).setText(ident.name)
                self.lockTable.item(row + i, 3).setText(str(ident.uuid))
            else:
                self.lockTable.item(row + i, 2).setText("")
                self.lockTable.item(row + i, 3).setText("")

    def _set_table_item_flags(self, item):
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)

    def create_lock_table(self):
        table = []
        for srv_name, locks in self.locks.items():
            if locks is None:
                continue
            for inst_name, ident in locks.items():
                if isinstance(ident, Ident):
                    table.append([srv_name, inst_name, ident.name, ident.uuid])
                else:
                    table.append([srv_name, inst_name, "", ""])
        self.lockTable.clear()
        self.lockTable.setColumnCount(4)
        self.lockTable.setHorizontalHeaderLabels(
            ["Server", "Instrument", "Lock Name", "Lock UUID"]
        )
        self.lockTable.setRowCount(len(table))
        for row, li in enumerate(table):
            for col, s in enumerate(li):
                item = QtWidgets.QTableWidgetItem(str(s))
                self._set_table_item_flags(item)
                self.lockTable.setItem(row, col, item)

    def update_states(self, states: dict):
        if not self.stateTable.rowCount():
            self.create_state_table(states)
        else:
            self.update_state_table(states)

    def create_state_table(self, states: dict):
        self.stateTable.clear()
        self.stateTable.setColumnCount(2)
        self.stateTable.setHorizontalHeaderLabels(["Node", "State"])
        self.stateTable.setRowCount(len(states))
        for row, (name, state) in enumerate(states.items()):
            s = "" if state is None else state.name
            item = QtWidgets.QTableWidgetItem(name)
            self._set_table_item_flags(item)
            self.stateTable.setItem(row, 0, item)
            item = QtWidgets.QTableWidgetItem(s)
            self._set_table_item_flags(item)
            self.stateTable.setItem(row, 1, item)

    def update_state_table(self, states: dict):
        for row, (name, state) in enumerate(states.items()):
            s = "" if state is None else state.name
            self.stateTable.item(row, 1).setText(s)


class MainMonitor(GUINode):
    def init_widget(self, gconf: dict, name, context):
        return MainMonitorWidget(gconf, name, context)
