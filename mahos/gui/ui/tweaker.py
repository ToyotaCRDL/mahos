# Form implementation generated from reading ui file 'tweaker.ui'
#
# Created by: PyQt6 UI code generator 6.7.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_TweakerWidget(object):
    def setupUi(self, TweakerWidget):
        TweakerWidget.setObjectName("TweakerWidget")
        TweakerWidget.resize(970, 476)
        self.verticalLayout = QtWidgets.QVBoxLayout(TweakerWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.readallButton = QtWidgets.QPushButton(parent=TweakerWidget)
        self.readallButton.setObjectName("readallButton")
        self.horizontalLayout.addWidget(self.readallButton)
        self.readButton = QtWidgets.QPushButton(parent=TweakerWidget)
        self.readButton.setObjectName("readButton")
        self.horizontalLayout.addWidget(self.readButton)
        self.writeallButton = QtWidgets.QPushButton(parent=TweakerWidget)
        self.writeallButton.setObjectName("writeallButton")
        self.horizontalLayout.addWidget(self.writeallButton)
        self.writeButton = QtWidgets.QPushButton(parent=TweakerWidget)
        self.writeButton.setObjectName("writeButton")
        self.horizontalLayout.addWidget(self.writeButton)
        self.startButton = QtWidgets.QPushButton(parent=TweakerWidget)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout.addWidget(self.startButton)
        self.stopButton = QtWidgets.QPushButton(parent=TweakerWidget)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout.addWidget(self.stopButton)
        self.saveButton = QtWidgets.QPushButton(parent=TweakerWidget)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout.addWidget(self.saveButton)
        self.loadButton = QtWidgets.QPushButton(parent=TweakerWidget)
        self.loadButton.setObjectName("loadButton")
        self.horizontalLayout.addWidget(self.loadButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.tabWidget = QtWidgets.QTabWidget(parent=TweakerWidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(TweakerWidget)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(TweakerWidget)
        TweakerWidget.setTabOrder(self.readallButton, self.readButton)
        TweakerWidget.setTabOrder(self.readButton, self.writeallButton)
        TweakerWidget.setTabOrder(self.writeallButton, self.writeButton)
        TweakerWidget.setTabOrder(self.writeButton, self.startButton)
        TweakerWidget.setTabOrder(self.startButton, self.stopButton)
        TweakerWidget.setTabOrder(self.stopButton, self.saveButton)
        TweakerWidget.setTabOrder(self.saveButton, self.loadButton)
        TweakerWidget.setTabOrder(self.loadButton, self.tabWidget)

    def retranslateUi(self, TweakerWidget):
        _translate = QtCore.QCoreApplication.translate
        TweakerWidget.setWindowTitle(_translate("TweakerWidget", "Form"))
        self.readallButton.setText(_translate("TweakerWidget", "Read All"))
        self.readButton.setText(_translate("TweakerWidget", "Read"))
        self.writeallButton.setText(_translate("TweakerWidget", "Write All"))
        self.writeButton.setText(_translate("TweakerWidget", "Write"))
        self.startButton.setText(_translate("TweakerWidget", "Start"))
        self.stopButton.setText(_translate("TweakerWidget", "Stop"))
        self.saveButton.setText(_translate("TweakerWidget", "Save"))
        self.loadButton.setText(_translate("TweakerWidget", "Load"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("TweakerWidget", "Tab 1"))
