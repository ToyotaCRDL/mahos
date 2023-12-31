# Form implementation generated from reading ui file 'hbt.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_HBT(object):
    def setupUi(self, HBT):
        HBT.setObjectName("HBT")
        HBT.resize(1064, 374)
        self.verticalLayout = QtWidgets.QVBoxLayout(HBT)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(parent=HBT)
        self.tabWidget.setObjectName("tabWidget")
        self.measTab = QtWidgets.QWidget()
        self.measTab.setObjectName("measTab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.measTab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.startButton = QtWidgets.QPushButton(parent=self.measTab)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout_4.addWidget(self.startButton)
        self.stopButton = QtWidgets.QPushButton(parent=self.measTab)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout_4.addWidget(self.stopButton)
        self.saveButton = QtWidgets.QPushButton(parent=self.measTab)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout_4.addWidget(self.saveButton)
        self.exportButton = QtWidgets.QPushButton(parent=self.measTab)
        self.exportButton.setObjectName("exportButton")
        self.horizontalLayout_4.addWidget(self.exportButton)
        self.loadButton = QtWidgets.QPushButton(parent=self.measTab)
        self.loadButton.setObjectName("loadButton")
        self.horizontalLayout_4.addWidget(self.loadButton)
        self.saveconfocalBox = QtWidgets.QCheckBox(parent=self.measTab)
        self.saveconfocalBox.setObjectName("saveconfocalBox")
        self.horizontalLayout_4.addWidget(self.saveconfocalBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.binBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.binBox.sizePolicy().hasHeightForWidth())
        self.binBox.setSizePolicy(sizePolicy)
        self.binBox.setDecimals(1)
        self.binBox.setMinimum(0.1)
        self.binBox.setMaximum(100.0)
        self.binBox.setProperty("value", 0.2)
        self.binBox.setObjectName("binBox")
        self.gridLayout.addWidget(self.binBox, 0, 1, 1, 1)
        self.refstartBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.refstartBox.sizePolicy().hasHeightForWidth())
        self.refstartBox.setSizePolicy(sizePolicy)
        self.refstartBox.setMinimum(-10000.0)
        self.refstartBox.setMaximum(10000.0)
        self.refstartBox.setSingleStep(10.0)
        self.refstartBox.setProperty("value", -200.0)
        self.refstartBox.setObjectName("refstartBox")
        self.gridLayout.addWidget(self.refstartBox, 1, 2, 1, 1)
        self.label = QtWidgets.QLabel(parent=self.measTab)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.plotenableBox = QtWidgets.QCheckBox(parent=self.measTab)
        self.plotenableBox.setObjectName("plotenableBox")
        self.gridLayout.addWidget(self.plotenableBox, 1, 5, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=self.measTab)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.t0Box = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.t0Box.sizePolicy().hasHeightForWidth())
        self.t0Box.setSizePolicy(sizePolicy)
        self.t0Box.setMaximum(500.0)
        self.t0Box.setProperty("value", 100.0)
        self.t0Box.setObjectName("t0Box")
        self.gridLayout.addWidget(self.t0Box, 1, 1, 1, 1)
        self.rangeLabel = QtWidgets.QLabel(parent=self.measTab)
        self.rangeLabel.setObjectName("rangeLabel")
        self.gridLayout.addWidget(self.rangeLabel, 0, 3, 1, 1)
        self.bgratioBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bgratioBox.sizePolicy().hasHeightForWidth())
        self.bgratioBox.setSizePolicy(sizePolicy)
        self.bgratioBox.setDecimals(2)
        self.bgratioBox.setMaximum(99.99)
        self.bgratioBox.setSingleStep(0.01)
        self.bgratioBox.setObjectName("bgratioBox")
        self.gridLayout.addWidget(self.bgratioBox, 1, 4, 1, 1)
        self.windowBox = QtWidgets.QSpinBox(parent=self.measTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.windowBox.sizePolicy().hasHeightForWidth())
        self.windowBox.setSizePolicy(sizePolicy)
        self.windowBox.setMinimumSize(QtCore.QSize(150, 0))
        self.windowBox.setMaximum(999999999)
        self.windowBox.setProperty("value", 1001)
        self.windowBox.setObjectName("windowBox")
        self.gridLayout.addWidget(self.windowBox, 0, 2, 1, 1)
        self.refstopBox = QtWidgets.QDoubleSpinBox(parent=self.measTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.refstopBox.sizePolicy().hasHeightForWidth())
        self.refstopBox.setSizePolicy(sizePolicy)
        self.refstopBox.setMinimum(-10000.0)
        self.refstopBox.setMaximum(10000.0)
        self.refstopBox.setSingleStep(10.0)
        self.refstopBox.setObjectName("refstopBox")
        self.gridLayout.addWidget(self.refstopBox, 1, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 6, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem2)
        self.tabWidget.addTab(self.measTab, "")
        self.fiTab = QtWidgets.QWidget()
        self.fiTab.setObjectName("fiTab")
        self.tabWidget.addTab(self.fiTab, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(HBT)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(HBT)
        HBT.setTabOrder(self.tabWidget, self.startButton)
        HBT.setTabOrder(self.startButton, self.stopButton)
        HBT.setTabOrder(self.stopButton, self.saveButton)
        HBT.setTabOrder(self.saveButton, self.exportButton)
        HBT.setTabOrder(self.exportButton, self.loadButton)
        HBT.setTabOrder(self.loadButton, self.saveconfocalBox)
        HBT.setTabOrder(self.saveconfocalBox, self.binBox)
        HBT.setTabOrder(self.binBox, self.windowBox)
        HBT.setTabOrder(self.windowBox, self.t0Box)
        HBT.setTabOrder(self.t0Box, self.refstartBox)
        HBT.setTabOrder(self.refstartBox, self.refstopBox)
        HBT.setTabOrder(self.refstopBox, self.bgratioBox)
        HBT.setTabOrder(self.bgratioBox, self.plotenableBox)

    def retranslateUi(self, HBT):
        _translate = QtCore.QCoreApplication.translate
        HBT.setWindowTitle(_translate("HBT", "HBT"))
        self.startButton.setText(_translate("HBT", "Start"))
        self.stopButton.setText(_translate("HBT", "Stop"))
        self.saveButton.setText(_translate("HBT", "Save"))
        self.exportButton.setText(_translate("HBT", "Export"))
        self.loadButton.setText(_translate("HBT", "Load"))
        self.saveconfocalBox.setText(_translate("HBT", "Save confocal position"))
        self.binBox.setPrefix(_translate("HBT", "time bin: "))
        self.binBox.setSuffix(_translate("HBT", " ns"))
        self.refstartBox.setPrefix(_translate("HBT", "ref_start: "))
        self.refstartBox.setSuffix(_translate("HBT", " ns"))
        self.label.setText(_translate("HBT", "TDC"))
        self.plotenableBox.setText(_translate("HBT", "enable edit"))
        self.label_2.setText(_translate("HBT", "Plot"))
        self.t0Box.setPrefix(_translate("HBT", "t0: "))
        self.t0Box.setSuffix(_translate("HBT", " ns"))
        self.rangeLabel.setText(_translate("HBT", "range: "))
        self.bgratioBox.setPrefix(_translate("HBT", "bg_ratio: "))
        self.bgratioBox.setSuffix(_translate("HBT", " %"))
        self.windowBox.setSuffix(_translate("HBT", " ns"))
        self.windowBox.setPrefix(_translate("HBT", "time window: "))
        self.refstopBox.setPrefix(_translate("HBT", "ref_stop: "))
        self.refstopBox.setSuffix(_translate("HBT", " ns"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.measTab), _translate("HBT", "Main"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.fiTab), _translate("HBT", "Fit"))
