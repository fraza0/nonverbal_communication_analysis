# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nonverbal_communication_analysis/m6_Visualization/UI/feature_comparator.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FeatureComparator(object):
    def setupUi(self, FeatureComparator):
        FeatureComparator.setObjectName("FeatureComparator")
        FeatureComparator.resize(1125, 720)
        self.verticalLayout = QtWidgets.QVBoxLayout(FeatureComparator)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_controls = QtWidgets.QFrame(FeatureComparator)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_controls.sizePolicy().hasHeightForWidth())
        self.frame_controls.setSizePolicy(sizePolicy)
        self.frame_controls.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_controls.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_controls.setObjectName("frame_controls")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_controls)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.cb_group1 = QtWidgets.QComboBox(self.frame_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cb_group1.sizePolicy().hasHeightForWidth())
        self.cb_group1.setSizePolicy(sizePolicy)
        self.cb_group1.setMinimumSize(QtCore.QSize(120, 0))
        self.cb_group1.setObjectName("cb_group1")
        self.cb_group1.addItem("")
        self.horizontalLayout.addWidget(self.cb_group1)
        self.cb_task1 = QtWidgets.QComboBox(self.frame_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cb_task1.sizePolicy().hasHeightForWidth())
        self.cb_task1.setSizePolicy(sizePolicy)
        self.cb_task1.setMinimumSize(QtCore.QSize(90, 0))
        self.cb_task1.setObjectName("cb_task1")
        self.cb_task1.addItem("")
        self.horizontalLayout.addWidget(self.cb_task1)
        self.tb_type1 = QtWidgets.QLineEdit(self.frame_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tb_type1.sizePolicy().hasHeightForWidth())
        self.tb_type1.setSizePolicy(sizePolicy)
        self.tb_type1.setMinimumSize(QtCore.QSize(35, 0))
        self.tb_type1.setMaximumSize(QtCore.QSize(35, 16777215))
        self.tb_type1.setReadOnly(True)
        self.tb_type1.setObjectName("tb_type1")
        self.horizontalLayout.addWidget(self.tb_type1)
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(self.frame_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.cb_group2 = QtWidgets.QComboBox(self.frame_controls)
        self.cb_group2.setMinimumSize(QtCore.QSize(120, 0))
        self.cb_group2.setObjectName("cb_group2")
        self.cb_group2.addItem("")
        self.horizontalLayout.addWidget(self.cb_group2)
        self.cb_task2 = QtWidgets.QComboBox(self.frame_controls)
        self.cb_task2.setMinimumSize(QtCore.QSize(90, 0))
        self.cb_task2.setObjectName("cb_task2")
        self.cb_task2.addItem("")
        self.horizontalLayout.addWidget(self.cb_task2)
        self.tb_type2 = QtWidgets.QLineEdit(self.frame_controls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tb_type2.sizePolicy().hasHeightForWidth())
        self.tb_type2.setSizePolicy(sizePolicy)
        self.tb_type2.setMinimumSize(QtCore.QSize(35, 0))
        self.tb_type2.setMaximumSize(QtCore.QSize(35, 16777215))
        self.tb_type2.setReadOnly(True)
        self.tb_type2.setObjectName("tb_type2")
        self.horizontalLayout.addWidget(self.tb_type2)
        spacerItem1 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.label_3 = QtWidgets.QLabel(self.frame_controls)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.cb_group3 = QtWidgets.QComboBox(self.frame_controls)
        self.cb_group3.setMinimumSize(QtCore.QSize(120, 0))
        self.cb_group3.setObjectName("cb_group3")
        self.cb_group3.addItem("")
        self.horizontalLayout.addWidget(self.cb_group3)
        self.cb_task3 = QtWidgets.QComboBox(self.frame_controls)
        self.cb_task3.setMinimumSize(QtCore.QSize(90, 0))
        self.cb_task3.setObjectName("cb_task3")
        self.cb_task3.addItem("")
        self.horizontalLayout.addWidget(self.cb_task3)
        self.tb_type3 = QtWidgets.QLineEdit(self.frame_controls)
        self.tb_type3.setMinimumSize(QtCore.QSize(35, 0))
        self.tb_type3.setMaximumSize(QtCore.QSize(35, 16777215))
        self.tb_type3.setReadOnly(True)
        self.tb_type3.setObjectName("tb_type3")
        self.horizontalLayout.addWidget(self.tb_type3)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout.addWidget(self.frame_controls)
        self.frame_filters = QtWidgets.QFrame(FeatureComparator)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_filters.sizePolicy().hasHeightForWidth())
        self.frame_filters.setSizePolicy(sizePolicy)
        self.frame_filters.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_filters.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_filters.setObjectName("frame_filters")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_filters)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.frame_filters)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.chb_subject1 = QtWidgets.QCheckBox(self.frame_filters)
        self.chb_subject1.setChecked(True)
        self.chb_subject1.setObjectName("chb_subject1")
        self.horizontalLayout_2.addWidget(self.chb_subject1)
        self.chb_subject2 = QtWidgets.QCheckBox(self.frame_filters)
        self.chb_subject2.setChecked(True)
        self.chb_subject2.setObjectName("chb_subject2")
        self.horizontalLayout_2.addWidget(self.chb_subject2)
        self.chb_subject3 = QtWidgets.QCheckBox(self.frame_filters)
        self.chb_subject3.setChecked(True)
        self.chb_subject3.setObjectName("chb_subject3")
        self.horizontalLayout_2.addWidget(self.chb_subject3)
        self.chb_subject4 = QtWidgets.QCheckBox(self.frame_filters)
        self.chb_subject4.setChecked(True)
        self.chb_subject4.setObjectName("chb_subject4")
        self.horizontalLayout_2.addWidget(self.chb_subject4)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.label_6 = QtWidgets.QLabel(self.frame_filters)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_2.addWidget(self.label_6)
        self.cb_camera = QtWidgets.QComboBox(self.frame_filters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cb_camera.sizePolicy().hasHeightForWidth())
        self.cb_camera.setSizePolicy(sizePolicy)
        self.cb_camera.setMinimumSize(QtCore.QSize(100, 0))
        self.cb_camera.setObjectName("cb_camera")
        self.cb_camera.addItem("")
        self.cb_camera.addItem("")
        self.cb_camera.addItem("")
        self.cb_camera.addItem("")
        self.horizontalLayout_2.addWidget(self.cb_camera)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.label_7 = QtWidgets.QLabel(self.frame_filters)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.cb_metric = QtWidgets.QComboBox(self.frame_filters)
        self.cb_metric.setMinimumSize(QtCore.QSize(200, 0))
        self.cb_metric.setObjectName("cb_metric")
        self.cb_metric.addItem("")
        self.cb_metric.addItem("")
        self.cb_metric.addItem("")
        self.cb_metric.addItem("")
        self.horizontalLayout_2.addWidget(self.cb_metric)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.line = QtWidgets.QFrame(self.frame_filters)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_3.addWidget(self.line)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_8 = QtWidgets.QLabel(self.frame_filters)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_7.addWidget(self.label_8)
        self.rad_raw = QtWidgets.QRadioButton(self.frame_filters)
        self.rad_raw.setObjectName("rad_raw")
        self.horizontalLayout_7.addWidget(self.rad_raw)
        self.rad_poly = QtWidgets.QRadioButton(self.frame_filters)
        self.rad_poly.setObjectName("rad_poly")
        self.horizontalLayout_7.addWidget(self.rad_poly)
        self.rad_moving_avg = QtWidgets.QRadioButton(self.frame_filters)
        self.rad_moving_avg.setObjectName("rad_moving_avg")
        self.horizontalLayout_7.addWidget(self.rad_moving_avg)
        self.rad_spline = QtWidgets.QRadioButton(self.frame_filters)
        self.rad_spline.setChecked(True)
        self.rad_spline.setObjectName("rad_spline")
        self.horizontalLayout_7.addWidget(self.rad_spline)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem4)
        self.btn_compare = QtWidgets.QPushButton(self.frame_filters)
        self.btn_compare.setObjectName("btn_compare")
        self.horizontalLayout_7.addWidget(self.btn_compare)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.verticalLayout.addWidget(self.frame_filters)
        self.frame_plots = QtWidgets.QFrame(FeatureComparator)
        self.frame_plots.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_plots.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_plots.setObjectName("frame_plots")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_plots)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.cvs_plot = QtWidgets.QFrame(self.frame_plots)
        self.cvs_plot.setObjectName("cvs_plot")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.cvs_plot)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_9.addLayout(self.verticalLayout_6)
        self.verticalLayout_4.addWidget(self.cvs_plot)
        self.verticalLayout.addWidget(self.frame_plots)
        self.horizontalFrame = QtWidgets.QFrame(FeatureComparator)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalFrame.sizePolicy().hasHeightForWidth())
        self.horizontalFrame.setSizePolicy(sizePolicy)
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem5 = QtWidgets.QSpacerItem(40, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)
        self.btn_save = QtWidgets.QPushButton(self.horizontalFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_save.sizePolicy().hasHeightForWidth())
        self.btn_save.setSizePolicy(sizePolicy)
        self.btn_save.setObjectName("btn_save")
        self.horizontalLayout_4.addWidget(self.btn_save)
        self.btn_close = QtWidgets.QPushButton(self.horizontalFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_close.sizePolicy().hasHeightForWidth())
        self.btn_close.setSizePolicy(sizePolicy)
        self.btn_close.setObjectName("btn_close")
        self.horizontalLayout_4.addWidget(self.btn_close)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_4)
        self.verticalLayout.addWidget(self.horizontalFrame)

        self.retranslateUi(FeatureComparator)
        QtCore.QMetaObject.connectSlotsByName(FeatureComparator)

    def retranslateUi(self, FeatureComparator):
        _translate = QtCore.QCoreApplication.translate
        FeatureComparator.setWindowTitle(_translate("FeatureComparator", "Group Feature Comparator"))
        self.label.setText(_translate("FeatureComparator", "Group 1"))
        self.cb_group1.setItemText(0, _translate("FeatureComparator", "Select Group ID"))
        self.cb_task1.setItemText(0, _translate("FeatureComparator", "Select Task"))
        self.label_2.setText(_translate("FeatureComparator", "Group 2"))
        self.cb_group2.setItemText(0, _translate("FeatureComparator", "Select Group ID"))
        self.cb_task2.setItemText(0, _translate("FeatureComparator", "Select Task"))
        self.label_3.setText(_translate("FeatureComparator", "Group 3"))
        self.cb_group3.setItemText(0, _translate("FeatureComparator", "Select Group ID"))
        self.cb_task3.setItemText(0, _translate("FeatureComparator", "Select Task"))
        self.label_5.setText(_translate("FeatureComparator", "Subjects:"))
        self.chb_subject1.setText(_translate("FeatureComparator", "Subject 1"))
        self.chb_subject2.setText(_translate("FeatureComparator", "Subject 2"))
        self.chb_subject3.setText(_translate("FeatureComparator", "Subject 3"))
        self.chb_subject4.setText(_translate("FeatureComparator", "Subject 4"))
        self.label_6.setText(_translate("FeatureComparator", "Camera:"))
        self.cb_camera.setItemText(0, _translate("FeatureComparator", "None"))
        self.cb_camera.setItemText(1, _translate("FeatureComparator", "PC1"))
        self.cb_camera.setItemText(2, _translate("FeatureComparator", "PC2"))
        self.cb_camera.setItemText(3, _translate("FeatureComparator", "PC3"))
        self.label_7.setText(_translate("FeatureComparator", "Metric:"))
        self.cb_metric.setItemText(0, _translate("FeatureComparator", "Intragroup Distance"))
        self.cb_metric.setItemText(1, _translate("FeatureComparator", "Energy"))
        self.cb_metric.setItemText(2, _translate("FeatureComparator", "Overlap"))
        self.cb_metric.setItemText(3, _translate("FeatureComparator", "Center Interaction"))
        self.label_8.setText(_translate("FeatureComparator", "Line Type:"))
        self.rad_raw.setText(_translate("FeatureComparator", "Raw"))
        self.rad_poly.setText(_translate("FeatureComparator", "Poly"))
        self.rad_moving_avg.setText(_translate("FeatureComparator", "Moving Average"))
        self.rad_spline.setText(_translate("FeatureComparator", "Spline"))
        self.btn_compare.setText(_translate("FeatureComparator", "Compare Group Data"))
        self.btn_save.setText(_translate("FeatureComparator", "Save Plot"))
        self.btn_save.setShortcut(_translate("FeatureComparator", "Ctrl+S"))
        self.btn_close.setText(_translate("FeatureComparator", "Close"))
        self.btn_close.setShortcut(_translate("FeatureComparator", "Ctrl+Q"))