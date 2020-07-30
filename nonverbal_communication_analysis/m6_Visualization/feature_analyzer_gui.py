# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nonverbal_communication_analysis/m6_Visualization/UI/feature_analyzer.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FeatureAnalyzer(object):
    def setupUi(self, FeatureAnalyzer):
        FeatureAnalyzer.setObjectName("FeatureAnalyzer")
        FeatureAnalyzer.resize(960, 720)
        self.verticalLayout = QtWidgets.QVBoxLayout(FeatureAnalyzer)
        self.verticalLayout.setObjectName("verticalLayout")
        self.controls_box = QtWidgets.QFrame(FeatureAnalyzer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.controls_box.sizePolicy().hasHeightForWidth())
        self.controls_box.setSizePolicy(sizePolicy)
        self.controls_box.setObjectName("controls_box")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.controls_box)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.btn_save = QtWidgets.QPushButton(self.controls_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_save.sizePolicy().hasHeightForWidth())
        self.btn_save.setSizePolicy(sizePolicy)
        self.btn_save.setText("")
        self.btn_save.setObjectName("btn_save")
        self.gridLayout.addWidget(self.btn_save, 0, 0, 1, 1)
        self.chb_poly = QtWidgets.QCheckBox(self.controls_box)
        self.chb_poly.setObjectName("chb_poly")
        self.gridLayout.addWidget(self.chb_poly, 0, 5, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 1, 1, 1)
        self.chb_raw = QtWidgets.QCheckBox(self.controls_box)
        self.chb_raw.setObjectName("chb_raw")
        self.gridLayout.addWidget(self.chb_raw, 0, 3, 1, 1)
        self.chb_spline = QtWidgets.QCheckBox(self.controls_box)
        self.chb_spline.setChecked(True)
        self.chb_spline.setObjectName("chb_spline")
        self.gridLayout.addWidget(self.chb_spline, 0, 4, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.controls_box)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)
        self.verticalLayout_10.addLayout(self.gridLayout)
        self.verticalLayout.addWidget(self.controls_box)
        self.groupBox = QtWidgets.QGroupBox(FeatureAnalyzer)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_5.addWidget(self.label_7)
        self.cvs_intragroup_dist = QtWidgets.QWidget(self.groupBox)
        self.cvs_intragroup_dist.setObjectName("cvs_intragroup_dist")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.cvs_intragroup_dist)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_11.addLayout(self.verticalLayout_6)
        self.verticalLayout_5.addWidget(self.cvs_intragroup_dist)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_5.addWidget(self.label_6)
        self.cvs_group_energy = QtWidgets.QWidget(self.groupBox)
        self.cvs_group_energy.setObjectName("cvs_group_energy")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.cvs_group_energy)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_12.addLayout(self.verticalLayout_8)
        self.verticalLayout_5.addWidget(self.cvs_group_energy)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(FeatureAnalyzer)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cvs_overlap_1 = QtWidgets.QWidget(self.groupBox_2)
        self.cvs_overlap_1.setObjectName("cvs_overlap_1")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.cvs_overlap_1)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.verticalLayout_16.addLayout(self.verticalLayout_15)
        self.horizontalLayout.addWidget(self.cvs_overlap_1)
        self.cvs_overlap_2 = QtWidgets.QWidget(self.groupBox_2)
        self.cvs_overlap_2.setObjectName("cvs_overlap_2")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.cvs_overlap_2)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.verticalLayout_18.addLayout(self.verticalLayout_17)
        self.horizontalLayout.addWidget(self.cvs_overlap_2)
        self.cvs_overlap_3 = QtWidgets.QWidget(self.groupBox_2)
        self.cvs_overlap_3.setObjectName("cvs_overlap_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.cvs_overlap_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        self.horizontalLayout.addWidget(self.cvs_overlap_3)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.label = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.cvs_env_interaction = QtWidgets.QWidget(self.groupBox_2)
        self.cvs_env_interaction.setObjectName("cvs_env_interaction")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.cvs_env_interaction)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_9.addLayout(self.verticalLayout_7)
        self.verticalLayout_3.addWidget(self.cvs_env_interaction)
        self.verticalLayout.addWidget(self.groupBox_2)

        self.retranslateUi(FeatureAnalyzer)
        QtCore.QMetaObject.connectSlotsByName(FeatureAnalyzer)

    def retranslateUi(self, FeatureAnalyzer):
        _translate = QtCore.QCoreApplication.translate
        FeatureAnalyzer.setWindowTitle(_translate("FeatureAnalyzer", "Feature Analyzer"))
        self.chb_poly.setText(_translate("FeatureAnalyzer", "Poly"))
        self.chb_raw.setText(_translate("FeatureAnalyzer", "Raw"))
        self.chb_spline.setText(_translate("FeatureAnalyzer", "Spline"))
        self.label_3.setText(_translate("FeatureAnalyzer", "Line Type:"))
        self.groupBox.setTitle(_translate("FeatureAnalyzer", "Group Features"))
        self.label_7.setText(_translate("FeatureAnalyzer", "Intragroup Distance"))
        self.label_6.setText(_translate("FeatureAnalyzer", "Group Energy"))
        self.groupBox_2.setTitle(_translate("FeatureAnalyzer", "Subject Features"))
        self.label_2.setText(_translate("FeatureAnalyzer", "Overlap"))
        self.label.setText(_translate("FeatureAnalyzer", "Environment Interaction"))
