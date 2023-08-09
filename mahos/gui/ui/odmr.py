# Form implementation generated from reading ui file 'odmr.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_ODMR(object):
    def setupUi(self, ODMR):
        ODMR.setObjectName("ODMR")
        ODMR.resize(1371, 497)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(ODMR)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(parent=ODMR)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.measTab = QtWidgets.QWidget()
        self.measTab.setObjectName("measTab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.measTab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.startButton = QtWidgets.QPushButton(parent=self.measTab)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout.addWidget(self.startButton)
        self.stopButton = QtWidgets.QPushButton(parent=self.measTab)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout.addWidget(self.stopButton)
        self.saveButton = QtWidgets.QPushButton(parent=self.measTab)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout.addWidget(self.saveButton)
        self.exportButton = QtWidgets.QPushButton(parent=self.measTab)
        self.exportButton.setObjectName("exportButton")
        self.horizontalLayout.addWidget(self.exportButton)
        self.loadButton = QtWidgets.QPushButton(parent=self.measTab)
        self.loadButton.setObjectName("loadButton")
        self.horizontalLayout.addWidget(self.loadButton)
        self.saveconfocalBox = QtWidgets.QCheckBox(parent=self.measTab)
        self.saveconfocalBox.setObjectName("saveconfocalBox")
        self.horizontalLayout.addWidget(self.saveconfocalBox)
        spacerItem = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.startBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.startBox.setDecimals(2)
        self.startBox.setMinimum(10.0)
        self.startBox.setMaximum(3000.0)
        self.startBox.setSingleStep(100.0)
        self.startBox.setProperty("value", 2740.0)
        self.startBox.setObjectName("startBox")
        self.gridLayout_2.addWidget(self.startBox, 0, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(parent=self.measTab)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 0, 0, 1, 1)
        self.bnumBox = QtWidgets.QSpinBox(parent=self.measTab)
        self.bnumBox.setSuffix("")
        self.bnumBox.setMinimum(1)
        self.bnumBox.setMaximum(999999)
        self.bnumBox.setSingleStep(100)
        self.bnumBox.setProperty("value", 1000)
        self.bnumBox.setObjectName("bnumBox")
        self.gridLayout_2.addWidget(self.bnumBox, 5, 1, 1, 1)
        self.mwdelayBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.mwdelayBox.setDecimals(1)
        self.mwdelayBox.setSingleStep(10.0)
        self.mwdelayBox.setObjectName("mwdelayBox")
        self.gridLayout_2.addWidget(self.mwdelayBox, 4, 3, 1, 1)
        self.powerBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.powerBox.setMinimum(-65.0)
        self.powerBox.setMaximum(54.0)
        self.powerBox.setProperty("value", 0.0)
        self.powerBox.setObjectName("powerBox")
        self.gridLayout_2.addWidget(self.powerBox, 0, 1, 1, 1)
        self.cwButton = QtWidgets.QRadioButton(parent=self.measTab)
        self.cwButton.setObjectName("cwButton")
        self.gridLayout_2.addWidget(self.cwButton, 3, 0, 1, 1)
        self.laserdelayBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.laserdelayBox.setEnabled(True)
        self.laserdelayBox.setDecimals(1)
        self.laserdelayBox.setSingleStep(10.0)
        self.laserdelayBox.setObjectName("laserdelayBox")
        self.gridLayout_2.addWidget(self.laserdelayBox, 4, 1, 1, 1)
        self.numBox = QtWidgets.QSpinBox(parent=self.measTab)
        self.numBox.setSuffix("")
        self.numBox.setMinimum(2)
        self.numBox.setMaximum(99999)
        self.numBox.setProperty("value", 101)
        self.numBox.setObjectName("numBox")
        self.gridLayout_2.addWidget(self.numBox, 0, 4, 1, 1)
        self.mwcontBox = QtWidgets.QCheckBox(parent=self.measTab)
        self.mwcontBox.setObjectName("mwcontBox")
        self.gridLayout_2.addWidget(self.mwcontBox, 1, 4, 1, 1)
        self.sweepsBox = QtWidgets.QSpinBox(parent=self.measTab)
        self.sweepsBox.setPrefix("")
        self.sweepsBox.setMinimum(0)
        self.sweepsBox.setMaximum(999999)
        self.sweepsBox.setSingleStep(1)
        self.sweepsBox.setProperty("value", 0)
        self.sweepsBox.setObjectName("sweepsBox")
        self.gridLayout_2.addWidget(self.sweepsBox, 1, 1, 1, 1)
        self.triggerwidthBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.triggerwidthBox.setDecimals(1)
        self.triggerwidthBox.setSingleStep(10.0)
        self.triggerwidthBox.setObjectName("triggerwidthBox")
        self.gridLayout_2.addWidget(self.triggerwidthBox, 5, 2, 1, 1)
        self.laserwidthBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.laserwidthBox.setDecimals(1)
        self.laserwidthBox.setSingleStep(10.0)
        self.laserwidthBox.setObjectName("laserwidthBox")
        self.gridLayout_2.addWidget(self.laserwidthBox, 4, 2, 1, 1)
        self.stopBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.stopBox.setDecimals(2)
        self.stopBox.setMinimum(10.0)
        self.stopBox.setMaximum(3000.0)
        self.stopBox.setSingleStep(100.0)
        self.stopBox.setProperty("value", 3000.0)
        self.stopBox.setObjectName("stopBox")
        self.gridLayout_2.addWidget(self.stopBox, 0, 3, 1, 1)
        self.stepLabel = QtWidgets.QLabel(parent=self.measTab)
        self.stepLabel.setObjectName("stepLabel")
        self.gridLayout_2.addWidget(self.stepLabel, 1, 2, 1, 1)
        self.pulseButton = QtWidgets.QRadioButton(parent=self.measTab)
        self.pulseButton.setObjectName("pulseButton")
        self.gridLayout_2.addWidget(self.pulseButton, 4, 0, 1, 1)
        self.mwwidthBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.mwwidthBox.setDecimals(1)
        self.mwwidthBox.setSingleStep(10.0)
        self.mwwidthBox.setObjectName("mwwidthBox")
        self.gridLayout_2.addWidget(self.mwwidthBox, 4, 4, 1, 1)
        self.windowBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.windowBox.setDecimals(4)
        self.windowBox.setMinimum(0.1)
        self.windowBox.setMaximum(1000.0)
        self.windowBox.setProperty("value", 10.0)
        self.windowBox.setObjectName("windowBox")
        self.gridLayout_2.addWidget(self.windowBox, 3, 1, 1, 1)
        self.backgroundBox = QtWidgets.QCheckBox(parent=self.measTab)
        self.backgroundBox.setObjectName("backgroundBox")
        self.gridLayout_2.addWidget(self.backgroundBox, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(parent=self.measTab)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 2, 0, 1, 1)
        self.pdrateBox = QtWidgets.QSpinBox(parent=self.measTab)
        self.pdrateBox.setMinimum(1)
        self.pdrateBox.setMaximum(10000)
        self.pdrateBox.setObjectName("pdrateBox")
        self.gridLayout_2.addWidget(self.pdrateBox, 2, 3, 1, 1)
        self.bgdelayBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        self.bgdelayBox.setDecimals(4)
        self.bgdelayBox.setMaximum(1000.0)
        self.bgdelayBox.setObjectName("bgdelayBox")
        self.gridLayout_2.addWidget(self.bgdelayBox, 2, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.pulseLabel = QtWidgets.QLabel(parent=self.measTab)
        self.pulseLabel.setObjectName("pulseLabel")
        self.verticalLayout.addWidget(self.pulseLabel)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.tabWidget.addTab(self.measTab, "")
        self.fiTab = QtWidgets.QWidget()
        self.fiTab.setObjectName("fiTab")
        self.tabWidget.addTab(self.fiTab, "")
        self.peaksTab = QtWidgets.QWidget()
        self.peaksTab.setMinimumSize(QtCore.QSize(0, 0))
        self.peaksTab.setObjectName("peaksTab")
        self.tabWidget.addTab(self.peaksTab, "")
        self.verticalLayout_2.addWidget(self.tabWidget)

        self.retranslateUi(ODMR)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(ODMR)
        ODMR.setTabOrder(self.tabWidget, self.startButton)
        ODMR.setTabOrder(self.startButton, self.stopButton)
        ODMR.setTabOrder(self.stopButton, self.saveButton)
        ODMR.setTabOrder(self.saveButton, self.exportButton)
        ODMR.setTabOrder(self.exportButton, self.loadButton)
        ODMR.setTabOrder(self.loadButton, self.saveconfocalBox)
        ODMR.setTabOrder(self.saveconfocalBox, self.powerBox)
        ODMR.setTabOrder(self.powerBox, self.startBox)
        ODMR.setTabOrder(self.startBox, self.stopBox)
        ODMR.setTabOrder(self.stopBox, self.numBox)
        ODMR.setTabOrder(self.numBox, self.sweepsBox)
        ODMR.setTabOrder(self.sweepsBox, self.mwcontBox)
        ODMR.setTabOrder(self.mwcontBox, self.backgroundBox)
        ODMR.setTabOrder(self.backgroundBox, self.bgdelayBox)
        ODMR.setTabOrder(self.bgdelayBox, self.pdrateBox)
        ODMR.setTabOrder(self.pdrateBox, self.cwButton)
        ODMR.setTabOrder(self.cwButton, self.pulseButton)
        ODMR.setTabOrder(self.pulseButton, self.windowBox)
        ODMR.setTabOrder(self.windowBox, self.laserdelayBox)
        ODMR.setTabOrder(self.laserdelayBox, self.laserwidthBox)
        ODMR.setTabOrder(self.laserwidthBox, self.mwdelayBox)
        ODMR.setTabOrder(self.mwdelayBox, self.mwwidthBox)
        ODMR.setTabOrder(self.mwwidthBox, self.bnumBox)
        ODMR.setTabOrder(self.bnumBox, self.triggerwidthBox)

    def retranslateUi(self, ODMR):
        _translate = QtCore.QCoreApplication.translate
        ODMR.setWindowTitle(_translate("ODMR", "ODMR"))
        self.startButton.setText(_translate("ODMR", "Start"))
        self.stopButton.setText(_translate("ODMR", "Stop"))
        self.saveButton.setText(_translate("ODMR", "Save"))
        self.exportButton.setText(_translate("ODMR", "Export"))
        self.loadButton.setText(_translate("ODMR", "Load"))
        self.saveconfocalBox.setText(_translate("ODMR", "Save confocal position"))
        self.startBox.setPrefix(_translate("ODMR", "start: "))
        self.startBox.setSuffix(_translate("ODMR", " MHz"))
        self.label_6.setText(_translate("ODMR", "Micro Wave"))
        self.bnumBox.setPrefix(_translate("ODMR", "burst num: "))
        self.mwdelayBox.setPrefix(_translate("ODMR", "mw delay: "))
        self.mwdelayBox.setSuffix(_translate("ODMR", " ns"))
        self.powerBox.setPrefix(_translate("ODMR", "power: "))
        self.powerBox.setSuffix(_translate("ODMR", " dBm"))
        self.cwButton.setText(_translate("ODMR", "CW"))
        self.laserdelayBox.setPrefix(_translate("ODMR", "laser delay: "))
        self.laserdelayBox.setSuffix(_translate("ODMR", " ns"))
        self.numBox.setPrefix(_translate("ODMR", "num: "))
        self.mwcontBox.setText(_translate("ODMR", "Continue MW irradiation after stop"))
        self.sweepsBox.setSuffix(_translate("ODMR", " sweeps (0 for inf)"))
        self.triggerwidthBox.setPrefix(_translate("ODMR", "trigger width: "))
        self.triggerwidthBox.setSuffix(_translate("ODMR", " ns"))
        self.laserwidthBox.setPrefix(_translate("ODMR", "laser width: "))
        self.laserwidthBox.setSuffix(_translate("ODMR", " ns"))
        self.stopBox.setPrefix(_translate("ODMR", "stop: "))
        self.stopBox.setSuffix(_translate("ODMR", " MHz"))
        self.stepLabel.setText(_translate("ODMR", "step: "))
        self.pulseButton.setText(_translate("ODMR", "Pulse"))
        self.mwwidthBox.setPrefix(_translate("ODMR", "mw width: "))
        self.mwwidthBox.setSuffix(_translate("ODMR", " ns"))
        self.windowBox.setPrefix(_translate("ODMR", "time window: "))
        self.windowBox.setSuffix(_translate("ODMR", " ms"))
        self.backgroundBox.setText(_translate("ODMR", "Measure Background Data"))
        self.label.setText(_translate("ODMR", "Detection"))
        self.pdrateBox.setSuffix(_translate("ODMR", " kHz"))
        self.pdrateBox.setPrefix(_translate("ODMR", "pd rate: "))
        self.bgdelayBox.setPrefix(_translate("ODMR", "bg delay: "))
        self.bgdelayBox.setSuffix(_translate("ODMR", " ms"))
        self.pulseLabel.setText(_translate("ODMR", "total window: 0.00 single window: 0.00 (rate: 0.00)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.measTab), _translate("ODMR", "Main"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.fiTab), _translate("ODMR", "Fit"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.peaksTab), _translate("ODMR", "Peaks"))
