# Form implementation generated from reading ui file 'chrono.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Chrono(object):
    def setupUi(self, Chrono):
        Chrono.setObjectName("Chrono")
        Chrono.resize(957, 707)
        self.verticalLayout = QtWidgets.QVBoxLayout(Chrono)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.startButton = QtWidgets.QPushButton(parent=Chrono)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout.addWidget(self.startButton)
        self.stopButton = QtWidgets.QPushButton(parent=Chrono)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout.addWidget(self.stopButton)
        self.saveButton = QtWidgets.QPushButton(parent=Chrono)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout.addWidget(self.saveButton)
        self.exportButton = QtWidgets.QPushButton(parent=Chrono)
        self.exportButton.setObjectName("exportButton")
        self.horizontalLayout.addWidget(self.exportButton)
        self.loadButton = QtWidgets.QPushButton(parent=Chrono)
        self.loadButton.setObjectName("loadButton")
        self.horizontalLayout.addWidget(self.loadButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.labelBox = QtWidgets.QComboBox(parent=Chrono)
        self.labelBox.setObjectName("labelBox")
        self.horizontalLayout_2.addWidget(self.labelBox)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.paramTable = ParamTable(parent=Chrono)
        self.paramTable.setObjectName("paramTable")
        self.paramTable.setColumnCount(0)
        self.paramTable.setRowCount(0)
        self.verticalLayout.addWidget(self.paramTable)

        self.retranslateUi(Chrono)
        QtCore.QMetaObject.connectSlotsByName(Chrono)

    def retranslateUi(self, Chrono):
        _translate = QtCore.QCoreApplication.translate
        Chrono.setWindowTitle(_translate("Chrono", "Form"))
        self.startButton.setText(_translate("Chrono", "Start"))
        self.stopButton.setText(_translate("Chrono", "Stop"))
        self.saveButton.setText(_translate("Chrono", "Save"))
        self.exportButton.setText(_translate("Chrono", "Export"))
        self.loadButton.setText(_translate("Chrono", "Load"))
from mahos.gui.param import ParamTable