# Form implementation generated from reading ui file 'oe200.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_OE200Widget(object):
    def setupUi(self, OE200Widget):
        OE200Widget.setObjectName("OE200Widget")
        OE200Widget.resize(359, 124)
        self.gridLayout = QtWidgets.QGridLayout(OE200Widget)
        self.gridLayout.setObjectName("gridLayout")
        self.lowButton = QtWidgets.QRadioButton(parent=OE200Widget)
        self.lowButton.setChecked(True)
        self.lowButton.setObjectName("lowButton")
        self.gainGroup = QtWidgets.QButtonGroup(OE200Widget)
        self.gainGroup.setObjectName("gainGroup")
        self.gainGroup.addButton(self.lowButton)
        self.gridLayout.addWidget(self.lowButton, 0, 0, 1, 1)
        self.highButton = QtWidgets.QRadioButton(parent=OE200Widget)
        self.highButton.setObjectName("highButton")
        self.gainGroup.addButton(self.highButton)
        self.gridLayout.addWidget(self.highButton, 1, 0, 1, 1)
        self.acButton = QtWidgets.QRadioButton(parent=OE200Widget)
        self.acButton.setChecked(True)
        self.acButton.setObjectName("acButton")
        self.couplingGroup = QtWidgets.QButtonGroup(OE200Widget)
        self.couplingGroup.setObjectName("couplingGroup")
        self.couplingGroup.addButton(self.acButton)
        self.gridLayout.addWidget(self.acButton, 2, 0, 1, 1)
        self.dcButton = QtWidgets.QRadioButton(parent=OE200Widget)
        self.dcButton.setObjectName("dcButton")
        self.couplingGroup.addButton(self.dcButton)
        self.gridLayout.addWidget(self.dcButton, 3, 0, 1, 1)
        self.setButton = QtWidgets.QPushButton(parent=OE200Widget)
        self.setButton.setObjectName("setButton")
        self.gridLayout.addWidget(self.setButton, 3, 1, 1, 1)
        self.lowBox = QtWidgets.QComboBox(parent=OE200Widget)
        self.lowBox.setObjectName("lowBox")
        self.lowBox.addItem("")
        self.lowBox.addItem("")
        self.lowBox.addItem("")
        self.lowBox.addItem("")
        self.lowBox.addItem("")
        self.lowBox.addItem("")
        self.lowBox.addItem("")
        self.gridLayout.addWidget(self.lowBox, 0, 1, 1, 1)
        self.highBox = QtWidgets.QComboBox(parent=OE200Widget)
        self.highBox.setObjectName("highBox")
        self.highBox.addItem("")
        self.highBox.addItem("")
        self.highBox.addItem("")
        self.highBox.addItem("")
        self.highBox.addItem("")
        self.highBox.addItem("")
        self.highBox.addItem("")
        self.gridLayout.addWidget(self.highBox, 1, 1, 1, 1)

        self.retranslateUi(OE200Widget)
        QtCore.QMetaObject.connectSlotsByName(OE200Widget)
        OE200Widget.setTabOrder(self.lowButton, self.highButton)
        OE200Widget.setTabOrder(self.highButton, self.lowBox)
        OE200Widget.setTabOrder(self.lowBox, self.highBox)
        OE200Widget.setTabOrder(self.highBox, self.acButton)
        OE200Widget.setTabOrder(self.acButton, self.dcButton)
        OE200Widget.setTabOrder(self.dcButton, self.setButton)

    def retranslateUi(self, OE200Widget):
        _translate = QtCore.QCoreApplication.translate
        OE200Widget.setWindowTitle(_translate("OE200Widget", "Form"))
        self.lowButton.setText(_translate("OE200Widget", "Low Noise"))
        self.highButton.setText(_translate("OE200Widget", "High Speed"))
        self.acButton.setText(_translate("OE200Widget", "AC"))
        self.dcButton.setText(_translate("OE200Widget", "DC"))
        self.setButton.setText(_translate("OE200Widget", "Set"))
        self.lowBox.setItemText(0, _translate("OE200Widget", "3"))
        self.lowBox.setItemText(1, _translate("OE200Widget", "4"))
        self.lowBox.setItemText(2, _translate("OE200Widget", "5"))
        self.lowBox.setItemText(3, _translate("OE200Widget", "6"))
        self.lowBox.setItemText(4, _translate("OE200Widget", "7"))
        self.lowBox.setItemText(5, _translate("OE200Widget", "8"))
        self.lowBox.setItemText(6, _translate("OE200Widget", "9"))
        self.highBox.setItemText(0, _translate("OE200Widget", "5"))
        self.highBox.setItemText(1, _translate("OE200Widget", "6"))
        self.highBox.setItemText(2, _translate("OE200Widget", "7"))
        self.highBox.setItemText(3, _translate("OE200Widget", "8"))
        self.highBox.setItemText(4, _translate("OE200Widget", "9"))
        self.highBox.setItemText(5, _translate("OE200Widget", "10"))
        self.highBox.setItemText(6, _translate("OE200Widget", "11"))