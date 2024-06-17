# Form implementation generated from reading ui file 'trackDialog.ui'
#
# Created by: PyQt6 UI code generator 6.7.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_trackDialog(object):
    def setupUi(self, trackDialog):
        trackDialog.setObjectName("trackDialog")
        trackDialog.resize(850, 612)
        self.verticalLayout = QtWidgets.QVBoxLayout(trackDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.xyyoffsetBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.xyyoffsetBox.setDecimals(4)
        self.xyyoffsetBox.setMinimum(-10.0)
        self.xyyoffsetBox.setMaximum(10.0)
        self.xyyoffsetBox.setSingleStep(0.01)
        self.xyyoffsetBox.setObjectName("xyyoffsetBox")
        self.gridLayout.addWidget(self.xyyoffsetBox, 9, 4, 1, 1)
        self.yzmodeBox = QtWidgets.QComboBox(parent=trackDialog)
        self.yzmodeBox.setObjectName("yzmodeBox")
        self.yzmodeBox.addItem("")
        self.yzmodeBox.addItem("")
        self.yzmodeBox.addItem("")
        self.yzmodeBox.addItem("")
        self.yzmodeBox.addItem("")
        self.yzmodeBox.addItem("")
        self.yzmodeBox.addItem("")
        self.yzmodeBox.addItem("")
        self.gridLayout.addWidget(self.yzmodeBox, 15, 0, 1, 1)
        self.pd_ubBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.pd_ubBox.setMinimum(-10.0)
        self.pd_ubBox.setMaximum(10.0)
        self.pd_ubBox.setProperty("value", 10.0)
        self.pd_ubBox.setObjectName("pd_ubBox")
        self.gridLayout.addWidget(self.pd_ubBox, 5, 2, 1, 1)
        self.xyystepLabel = QtWidgets.QLabel(parent=trackDialog)
        self.xyystepLabel.setObjectName("xyystepLabel")
        self.gridLayout.addWidget(self.xyystepLabel, 9, 3, 1, 1)
        self.xzxlenBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.xzxlenBox.setDecimals(4)
        self.xzxlenBox.setMinimum(0.1)
        self.xzxlenBox.setProperty("value", 2.0)
        self.xzxlenBox.setObjectName("xzxlenBox")
        self.gridLayout.addWidget(self.xzxlenBox, 12, 1, 1, 1)
        self.xzxstepLabel = QtWidgets.QLabel(parent=trackDialog)
        self.xzxstepLabel.setObjectName("xzxstepLabel")
        self.gridLayout.addWidget(self.xzxstepLabel, 12, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=trackDialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 6, 0, 1, 1)
        self.lenLabel_2 = QtWidgets.QLabel(parent=trackDialog)
        self.lenLabel_2.setObjectName("lenLabel_2")
        self.gridLayout.addWidget(self.lenLabel_2, 11, 1, 1, 1)
        self.xzxoffsetBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.xzxoffsetBox.setDecimals(4)
        self.xzxoffsetBox.setMinimum(-10.0)
        self.xzxoffsetBox.setMaximum(10.0)
        self.xzxoffsetBox.setSingleStep(0.01)
        self.xzxoffsetBox.setObjectName("xzxoffsetBox")
        self.gridLayout.addWidget(self.xzxoffsetBox, 12, 4, 1, 1)
        self.xyylenBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.xyylenBox.setDecimals(4)
        self.xyylenBox.setMinimum(0.1)
        self.xyylenBox.setProperty("value", 2.0)
        self.xyylenBox.setObjectName("xyylenBox")
        self.gridLayout.addWidget(self.xyylenBox, 9, 1, 1, 1)
        self.yzylenBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.yzylenBox.setDecimals(4)
        self.yzylenBox.setMinimum(0.1)
        self.yzylenBox.setProperty("value", 2.0)
        self.yzylenBox.setObjectName("yzylenBox")
        self.gridLayout.addWidget(self.yzylenBox, 16, 1, 1, 1)
        self.lenLabel = QtWidgets.QLabel(parent=trackDialog)
        self.lenLabel.setObjectName("lenLabel")
        self.gridLayout.addWidget(self.lenLabel, 7, 1, 1, 1)
        self.xzmodeBox = QtWidgets.QComboBox(parent=trackDialog)
        self.xzmodeBox.setObjectName("xzmodeBox")
        self.xzmodeBox.addItem("")
        self.xzmodeBox.addItem("")
        self.xzmodeBox.addItem("")
        self.xzmodeBox.addItem("")
        self.xzmodeBox.addItem("")
        self.xzmodeBox.addItem("")
        self.xzmodeBox.addItem("")
        self.xzmodeBox.addItem("")
        self.gridLayout.addWidget(self.xzmodeBox, 11, 0, 1, 1)
        self.xzznumBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.xzznumBox.setMinimum(2)
        self.xzznumBox.setMaximum(9999)
        self.xzznumBox.setProperty("value", 21)
        self.xzznumBox.setObjectName("xzznumBox")
        self.gridLayout.addWidget(self.xzznumBox, 13, 2, 1, 1)
        self.yLabel = QtWidgets.QLabel(parent=trackDialog)
        self.yLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.yLabel.setObjectName("yLabel")
        self.gridLayout.addWidget(self.yLabel, 9, 0, 1, 1)
        self.xyxoffsetBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.xyxoffsetBox.setDecimals(4)
        self.xyxoffsetBox.setMinimum(-10.0)
        self.xyxoffsetBox.setMaximum(10.0)
        self.xyxoffsetBox.setSingleStep(0.01)
        self.xyxoffsetBox.setObjectName("xyxoffsetBox")
        self.gridLayout.addWidget(self.xyxoffsetBox, 8, 4, 1, 1)
        self.label_10 = QtWidgets.QLabel(parent=trackDialog)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 0, 1, 1)
        self.pd_lbBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.pd_lbBox.setMinimum(-10.0)
        self.pd_lbBox.setMaximum(10.0)
        self.pd_lbBox.setProperty("value", -10.0)
        self.pd_lbBox.setObjectName("pd_lbBox")
        self.gridLayout.addWidget(self.pd_lbBox, 5, 1, 1, 1)
        self.xzzlenBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.xzzlenBox.setDecimals(4)
        self.xzzlenBox.setMinimum(0.1)
        self.xzzlenBox.setProperty("value", 2.0)
        self.xzzlenBox.setObjectName("xzzlenBox")
        self.gridLayout.addWidget(self.xzzlenBox, 13, 1, 1, 1)
        self.yzynumBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.yzynumBox.setMinimum(2)
        self.yzynumBox.setMaximum(9999)
        self.yzynumBox.setProperty("value", 21)
        self.yzynumBox.setObjectName("yzynumBox")
        self.gridLayout.addWidget(self.yzynumBox, 16, 2, 1, 1)
        self.yzyoffsetBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.yzyoffsetBox.setDecimals(4)
        self.yzyoffsetBox.setMinimum(-10.0)
        self.yzyoffsetBox.setMaximum(10.0)
        self.yzyoffsetBox.setSingleStep(0.01)
        self.yzyoffsetBox.setObjectName("yzyoffsetBox")
        self.gridLayout.addWidget(self.yzyoffsetBox, 16, 4, 1, 1)
        self.stepLabel_3 = QtWidgets.QLabel(parent=trackDialog)
        self.stepLabel_3.setObjectName("stepLabel_3")
        self.gridLayout.addWidget(self.stepLabel_3, 15, 3, 1, 1)
        self.yzzstepLabel = QtWidgets.QLabel(parent=trackDialog)
        self.yzzstepLabel.setObjectName("yzzstepLabel")
        self.gridLayout.addWidget(self.yzzstepLabel, 17, 3, 1, 1)
        self.label = QtWidgets.QLabel(parent=trackDialog)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 12, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(parent=trackDialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 14, 0, 1, 1)
        self.xyxnumBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.xyxnumBox.setMinimum(2)
        self.xyxnumBox.setMaximum(9999)
        self.xyxnumBox.setProperty("value", 21)
        self.xyxnumBox.setObjectName("xyxnumBox")
        self.gridLayout.addWidget(self.xyxnumBox, 8, 2, 1, 1)
        self.yzzoffsetBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.yzzoffsetBox.setDecimals(4)
        self.yzzoffsetBox.setMinimum(-10.0)
        self.yzzoffsetBox.setMaximum(10.0)
        self.yzzoffsetBox.setSingleStep(0.01)
        self.yzzoffsetBox.setObjectName("yzzoffsetBox")
        self.gridLayout.addWidget(self.yzzoffsetBox, 17, 4, 1, 1)
        self.xymodeBox = QtWidgets.QComboBox(parent=trackDialog)
        self.xymodeBox.setObjectName("xymodeBox")
        self.xymodeBox.addItem("")
        self.xymodeBox.addItem("")
        self.xymodeBox.addItem("")
        self.xymodeBox.addItem("")
        self.xymodeBox.addItem("")
        self.xymodeBox.addItem("")
        self.xymodeBox.addItem("")
        self.xymodeBox.addItem("")
        self.gridLayout.addWidget(self.xymodeBox, 7, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=trackDialog)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 13, 0, 1, 1)
        self.stepLabel_2 = QtWidgets.QLabel(parent=trackDialog)
        self.stepLabel_2.setObjectName("stepLabel_2")
        self.gridLayout.addWidget(self.stepLabel_2, 11, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(parent=trackDialog)
        self.label_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 16, 0, 1, 1)
        self.numLabel_3 = QtWidgets.QLabel(parent=trackDialog)
        self.numLabel_3.setObjectName("numLabel_3")
        self.gridLayout.addWidget(self.numLabel_3, 15, 2, 1, 1)
        self.timeBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.timeBox.setDecimals(4)
        self.timeBox.setMinimum(0.1)
        self.timeBox.setMaximum(1000.0)
        self.timeBox.setProperty("value", 10.0)
        self.timeBox.setObjectName("timeBox")
        self.gridLayout.addWidget(self.timeBox, 3, 1, 1, 1)
        self.numLabel = QtWidgets.QLabel(parent=trackDialog)
        self.numLabel.setObjectName("numLabel")
        self.gridLayout.addWidget(self.numLabel, 7, 2, 1, 1)
        self.xzxnumBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.xzxnumBox.setMinimum(2)
        self.xzxnumBox.setMaximum(9999)
        self.xzxnumBox.setProperty("value", 21)
        self.xzxnumBox.setObjectName("xzxnumBox")
        self.gridLayout.addWidget(self.xzxnumBox, 12, 2, 1, 1)
        self.xzzstepLabel = QtWidgets.QLabel(parent=trackDialog)
        self.xzzstepLabel.setObjectName("xzzstepLabel")
        self.gridLayout.addWidget(self.xzzstepLabel, 13, 3, 1, 1)
        self.saveBox = QtWidgets.QCheckBox(parent=trackDialog)
        self.saveBox.setObjectName("saveBox")
        self.gridLayout.addWidget(self.saveBox, 1, 2, 1, 1)
        self.pollsampleBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.pollsampleBox.setSuffix("")
        self.pollsampleBox.setMinimum(1000)
        self.pollsampleBox.setMaximum(10000)
        self.pollsampleBox.setSingleStep(1000)
        self.pollsampleBox.setObjectName("pollsampleBox")
        self.gridLayout.addWidget(self.pollsampleBox, 4, 1, 1, 1)
        self.linemodeBox = QtWidgets.QComboBox(parent=trackDialog)
        self.linemodeBox.setObjectName("linemodeBox")
        self.linemodeBox.addItem("")
        self.linemodeBox.addItem("")
        self.linemodeBox.addItem("")
        self.gridLayout.addWidget(self.linemodeBox, 2, 3, 1, 1)
        self.intervalBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.intervalBox.setDecimals(1)
        self.intervalBox.setMinimum(-1.0)
        self.intervalBox.setMaximum(10000.0)
        self.intervalBox.setProperty("value", 180.0)
        self.intervalBox.setObjectName("intervalBox")
        self.gridLayout.addWidget(self.intervalBox, 1, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(parent=trackDialog)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 3, 0, 1, 1)
        self.oversampleBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.oversampleBox.setMinimum(1)
        self.oversampleBox.setMaximum(100000)
        self.oversampleBox.setObjectName("oversampleBox")
        self.gridLayout.addWidget(self.oversampleBox, 3, 3, 1, 1)
        self.numLabel_2 = QtWidgets.QLabel(parent=trackDialog)
        self.numLabel_2.setObjectName("numLabel_2")
        self.gridLayout.addWidget(self.numLabel_2, 11, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(parent=trackDialog)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 7, 4, 1, 1)
        self.yzznumBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.yzznumBox.setMinimum(2)
        self.yzznumBox.setMaximum(9999)
        self.yzznumBox.setProperty("value", 21)
        self.yzznumBox.setObjectName("yzznumBox")
        self.gridLayout.addWidget(self.yzznumBox, 17, 2, 1, 1)
        self.xyxstepLabel = QtWidgets.QLabel(parent=trackDialog)
        self.xyxstepLabel.setObjectName("xyxstepLabel")
        self.gridLayout.addWidget(self.xyxstepLabel, 8, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=trackDialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 10, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(parent=trackDialog)
        self.label_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 17, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(parent=trackDialog)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 5, 0, 1, 1)
        self.yzzlenBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.yzzlenBox.setDecimals(4)
        self.yzzlenBox.setMinimum(0.1)
        self.yzzlenBox.setProperty("value", 2.0)
        self.yzzlenBox.setObjectName("yzzlenBox")
        self.gridLayout.addWidget(self.yzzlenBox, 17, 1, 1, 1)
        self.xLabel = QtWidgets.QLabel(parent=trackDialog)
        self.xLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.xLabel.setObjectName("xLabel")
        self.gridLayout.addWidget(self.xLabel, 8, 0, 1, 1)
        self.lenLabel_3 = QtWidgets.QLabel(parent=trackDialog)
        self.lenLabel_3.setObjectName("lenLabel_3")
        self.gridLayout.addWidget(self.lenLabel_3, 15, 1, 1, 1)
        self.xyxlenBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.xyxlenBox.setDecimals(4)
        self.xyxlenBox.setMinimum(0.1)
        self.xyxlenBox.setProperty("value", 2.0)
        self.xyxlenBox.setObjectName("xyxlenBox")
        self.gridLayout.addWidget(self.xyxlenBox, 8, 1, 1, 1)
        self.modeBox = QtWidgets.QComboBox(parent=trackDialog)
        self.modeBox.setObjectName("modeBox")
        self.modeBox.addItem("")
        self.gridLayout.addWidget(self.modeBox, 2, 1, 1, 1)
        self.delayBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.delayBox.setObjectName("delayBox")
        self.gridLayout.addWidget(self.delayBox, 4, 2, 1, 1)
        self.xyynumBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.xyynumBox.setMinimum(2)
        self.xyynumBox.setMaximum(9999)
        self.xyynumBox.setProperty("value", 21)
        self.xyynumBox.setObjectName("xyynumBox")
        self.gridLayout.addWidget(self.xyynumBox, 9, 2, 1, 1)
        self.yzystepLabel = QtWidgets.QLabel(parent=trackDialog)
        self.yzystepLabel.setObjectName("yzystepLabel")
        self.gridLayout.addWidget(self.yzystepLabel, 16, 3, 1, 1)
        self.stepLabel = QtWidgets.QLabel(parent=trackDialog)
        self.stepLabel.setObjectName("stepLabel")
        self.gridLayout.addWidget(self.stepLabel, 7, 3, 1, 1)
        self.dummysampleBox = QtWidgets.QSpinBox(parent=trackDialog)
        self.dummysampleBox.setMinimum(1)
        self.dummysampleBox.setMaximum(999)
        self.dummysampleBox.setProperty("value", 10)
        self.dummysampleBox.setObjectName("dummysampleBox")
        self.gridLayout.addWidget(self.dummysampleBox, 3, 2, 1, 1)
        self.xzzoffsetBox = QtWidgets.QDoubleSpinBox(parent=trackDialog)
        self.xzzoffsetBox.setDecimals(4)
        self.xzzoffsetBox.setMinimum(-10.0)
        self.xzzoffsetBox.setMaximum(10.0)
        self.xzzoffsetBox.setSingleStep(0.01)
        self.xzzoffsetBox.setObjectName("xzzoffsetBox")
        self.gridLayout.addWidget(self.xzzoffsetBox, 13, 4, 1, 1)
        self.label_11 = QtWidgets.QLabel(parent=trackDialog)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 2, 2, 1, 1)
        self.saveButton = QtWidgets.QPushButton(parent=trackDialog)
        self.saveButton.setObjectName("saveButton")
        self.gridLayout.addWidget(self.saveButton, 0, 1, 1, 1)
        self.loadButton = QtWidgets.QPushButton(parent=trackDialog)
        self.loadButton.setObjectName("loadButton")
        self.gridLayout.addWidget(self.loadButton, 0, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(parent=trackDialog)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.defaultButton = QtWidgets.QPushButton(parent=trackDialog)
        self.defaultButton.setObjectName("defaultButton")
        self.gridLayout.addWidget(self.defaultButton, 0, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_13 = QtWidgets.QLabel(parent=trackDialog)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_2.addWidget(self.label_13)
        self.upButton = QtWidgets.QPushButton(parent=trackDialog)
        self.upButton.setObjectName("upButton")
        self.verticalLayout_2.addWidget(self.upButton)
        self.downButton = QtWidgets.QPushButton(parent=trackDialog)
        self.downButton.setObjectName("downButton")
        self.verticalLayout_2.addWidget(self.downButton)
        self.horizontalLayout_9.addLayout(self.verticalLayout_2)
        self.orderList = QtWidgets.QListWidget(parent=trackDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.orderList.sizePolicy().hasHeightForWidth())
        self.orderList.setSizePolicy(sizePolicy)
        self.orderList.setMaximumSize(QtCore.QSize(16777215, 100))
        self.orderList.setObjectName("orderList")
        item = QtWidgets.QListWidgetItem()
        self.orderList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.orderList.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.orderList.addItem(item)
        self.horizontalLayout_9.addWidget(self.orderList)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=trackDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(trackDialog)
        self.buttonBox.accepted.connect(trackDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(trackDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(trackDialog)
        trackDialog.setTabOrder(self.saveButton, self.loadButton)
        trackDialog.setTabOrder(self.loadButton, self.defaultButton)
        trackDialog.setTabOrder(self.defaultButton, self.intervalBox)
        trackDialog.setTabOrder(self.intervalBox, self.saveBox)
        trackDialog.setTabOrder(self.saveBox, self.modeBox)
        trackDialog.setTabOrder(self.modeBox, self.linemodeBox)
        trackDialog.setTabOrder(self.linemodeBox, self.timeBox)
        trackDialog.setTabOrder(self.timeBox, self.dummysampleBox)
        trackDialog.setTabOrder(self.dummysampleBox, self.oversampleBox)
        trackDialog.setTabOrder(self.oversampleBox, self.pollsampleBox)
        trackDialog.setTabOrder(self.pollsampleBox, self.delayBox)
        trackDialog.setTabOrder(self.delayBox, self.pd_lbBox)
        trackDialog.setTabOrder(self.pd_lbBox, self.pd_ubBox)
        trackDialog.setTabOrder(self.pd_ubBox, self.xymodeBox)
        trackDialog.setTabOrder(self.xymodeBox, self.xyxlenBox)
        trackDialog.setTabOrder(self.xyxlenBox, self.xyxnumBox)
        trackDialog.setTabOrder(self.xyxnumBox, self.xyxoffsetBox)
        trackDialog.setTabOrder(self.xyxoffsetBox, self.xyylenBox)
        trackDialog.setTabOrder(self.xyylenBox, self.xyynumBox)
        trackDialog.setTabOrder(self.xyynumBox, self.xyyoffsetBox)
        trackDialog.setTabOrder(self.xyyoffsetBox, self.xzmodeBox)
        trackDialog.setTabOrder(self.xzmodeBox, self.xzxlenBox)
        trackDialog.setTabOrder(self.xzxlenBox, self.xzxnumBox)
        trackDialog.setTabOrder(self.xzxnumBox, self.xzxoffsetBox)
        trackDialog.setTabOrder(self.xzxoffsetBox, self.xzzlenBox)
        trackDialog.setTabOrder(self.xzzlenBox, self.xzznumBox)
        trackDialog.setTabOrder(self.xzznumBox, self.xzzoffsetBox)
        trackDialog.setTabOrder(self.xzzoffsetBox, self.yzmodeBox)
        trackDialog.setTabOrder(self.yzmodeBox, self.yzylenBox)
        trackDialog.setTabOrder(self.yzylenBox, self.yzynumBox)
        trackDialog.setTabOrder(self.yzynumBox, self.yzyoffsetBox)
        trackDialog.setTabOrder(self.yzyoffsetBox, self.yzzlenBox)
        trackDialog.setTabOrder(self.yzzlenBox, self.yzznumBox)
        trackDialog.setTabOrder(self.yzznumBox, self.yzzoffsetBox)
        trackDialog.setTabOrder(self.yzzoffsetBox, self.upButton)
        trackDialog.setTabOrder(self.upButton, self.downButton)
        trackDialog.setTabOrder(self.downButton, self.orderList)

    def retranslateUi(self, trackDialog):
        _translate = QtCore.QCoreApplication.translate
        trackDialog.setWindowTitle(_translate("trackDialog", "Track scan setting"))
        self.yzmodeBox.setItemText(0, _translate("trackDialog", "Disable"))
        self.yzmodeBox.setItemText(1, _translate("trackDialog", "Phase Only Correlation"))
        self.yzmodeBox.setItemText(2, _translate("trackDialog", "2D Gaussian"))
        self.yzmodeBox.setItemText(3, _translate("trackDialog", "1D Gaussian (Y)"))
        self.yzmodeBox.setItemText(4, _translate("trackDialog", "1D Gaussian (Z)"))
        self.yzmodeBox.setItemText(5, _translate("trackDialog", "Maximum pixel"))
        self.yzmodeBox.setItemText(6, _translate("trackDialog", "1D Maximum (Y)"))
        self.yzmodeBox.setItemText(7, _translate("trackDialog", "1D Maximum (Z)"))
        self.pd_ubBox.setPrefix(_translate("trackDialog", "upper: "))
        self.pd_ubBox.setSuffix(_translate("trackDialog", " V"))
        self.xyystepLabel.setText(_translate("trackDialog", "1"))
        self.xzxstepLabel.setText(_translate("trackDialog", "1"))
        self.label_3.setText(_translate("trackDialog", "XY Track Type"))
        self.lenLabel_2.setText(_translate("trackDialog", "length"))
        self.lenLabel.setText(_translate("trackDialog", "length"))
        self.xzmodeBox.setItemText(0, _translate("trackDialog", "Disable"))
        self.xzmodeBox.setItemText(1, _translate("trackDialog", "Phase Only Correlation"))
        self.xzmodeBox.setItemText(2, _translate("trackDialog", "2D Gaussian"))
        self.xzmodeBox.setItemText(3, _translate("trackDialog", "1D Gaussian (X)"))
        self.xzmodeBox.setItemText(4, _translate("trackDialog", "1D Gaussian (Z)"))
        self.xzmodeBox.setItemText(5, _translate("trackDialog", "Maximum pixel"))
        self.xzmodeBox.setItemText(6, _translate("trackDialog", "1D Maximum (X)"))
        self.xzmodeBox.setItemText(7, _translate("trackDialog", "1D Maximum (Z)"))
        self.yLabel.setText(_translate("trackDialog", "Y"))
        self.label_10.setText(_translate("trackDialog", "Scan mode"))
        self.pd_lbBox.setPrefix(_translate("trackDialog", "lower: "))
        self.pd_lbBox.setSuffix(_translate("trackDialog", " V"))
        self.stepLabel_3.setText(_translate("trackDialog", "step"))
        self.yzzstepLabel.setText(_translate("trackDialog", "1"))
        self.label.setText(_translate("trackDialog", "X"))
        self.label_5.setText(_translate("trackDialog", "YZ Track Type"))
        self.xymodeBox.setItemText(0, _translate("trackDialog", "Disable"))
        self.xymodeBox.setItemText(1, _translate("trackDialog", "Phase Only Correlation"))
        self.xymodeBox.setItemText(2, _translate("trackDialog", "2D Gaussian"))
        self.xymodeBox.setItemText(3, _translate("trackDialog", "1D Gaussian (X)"))
        self.xymodeBox.setItemText(4, _translate("trackDialog", "1D Gaussian (Y)"))
        self.xymodeBox.setItemText(5, _translate("trackDialog", "Maximum pixel"))
        self.xymodeBox.setItemText(6, _translate("trackDialog", "1D Maximum (X)"))
        self.xymodeBox.setItemText(7, _translate("trackDialog", "1D Maximum (Y)"))
        self.label_2.setText(_translate("trackDialog", "Z"))
        self.stepLabel_2.setText(_translate("trackDialog", "step"))
        self.label_6.setText(_translate("trackDialog", "Y"))
        self.numLabel_3.setText(_translate("trackDialog", "num"))
        self.timeBox.setPrefix(_translate("trackDialog", "time window: "))
        self.timeBox.setSuffix(_translate("trackDialog", " ms"))
        self.numLabel.setText(_translate("trackDialog", "num"))
        self.xzzstepLabel.setText(_translate("trackDialog", "1"))
        self.saveBox.setText(_translate("trackDialog", "Save data"))
        self.pollsampleBox.setPrefix(_translate("trackDialog", "poll samp: "))
        self.linemodeBox.setItemText(0, _translate("trackDialog", "ASCEND"))
        self.linemodeBox.setItemText(1, _translate("trackDialog", "DESCEND"))
        self.linemodeBox.setItemText(2, _translate("trackDialog", "ZIGZAG"))
        self.intervalBox.setPrefix(_translate("trackDialog", "interval "))
        self.intervalBox.setSuffix(_translate("trackDialog", " s"))
        self.label_9.setText(_translate("trackDialog", "Timing"))
        self.oversampleBox.setPrefix(_translate("trackDialog", "oversamp: "))
        self.numLabel_2.setText(_translate("trackDialog", "num"))
        self.label_12.setText(_translate("trackDialog", "offset"))
        self.xyxstepLabel.setText(_translate("trackDialog", "1"))
        self.label_4.setText(_translate("trackDialog", "XZ Track Type"))
        self.label_7.setText(_translate("trackDialog", "Z"))
        self.label_14.setText(_translate("trackDialog", "Analog PD"))
        self.xLabel.setText(_translate("trackDialog", "X"))
        self.lenLabel_3.setText(_translate("trackDialog", "length"))
        self.modeBox.setItemText(0, _translate("trackDialog", "None"))
        self.delayBox.setPrefix(_translate("trackDialog", "delay: "))
        self.delayBox.setSuffix(_translate("trackDialog", " ms"))
        self.yzystepLabel.setText(_translate("trackDialog", "1"))
        self.stepLabel.setText(_translate("trackDialog", "step"))
        self.dummysampleBox.setPrefix(_translate("trackDialog", "dummy samp: "))
        self.label_11.setText(_translate("trackDialog", "Line mode"))
        self.saveButton.setText(_translate("trackDialog", "Save"))
        self.loadButton.setText(_translate("trackDialog", "Load"))
        self.label_8.setText(_translate("trackDialog", "Set Tracking parameters"))
        self.defaultButton.setText(_translate("trackDialog", "Set default"))
        self.label_13.setText(_translate("trackDialog", "Track order"))
        self.upButton.setText(_translate("trackDialog", "Up"))
        self.downButton.setText(_translate("trackDialog", "Down"))
        __sortingEnabled = self.orderList.isSortingEnabled()
        self.orderList.setSortingEnabled(False)
        item = self.orderList.item(0)
        item.setText(_translate("trackDialog", "XY"))
        item = self.orderList.item(1)
        item.setText(_translate("trackDialog", "XZ"))
        item = self.orderList.item(2)
        item.setText(_translate("trackDialog", "YZ"))
        self.orderList.setSortingEnabled(__sortingEnabled)
