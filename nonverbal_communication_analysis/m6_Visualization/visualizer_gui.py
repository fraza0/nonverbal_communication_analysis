# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nonverbal_communication_analysis/m6_Visualization/UI/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Visualizer(object):
    def setupUi(self, Visualizer):
        Visualizer.setObjectName("Visualizer")
        Visualizer.resize(1420, 720)
        self.centralwidget = QtWidgets.QWidget(Visualizer)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.fr1_group_select = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fr1_group_select.sizePolicy().hasHeightForWidth())
        self.fr1_group_select.setSizePolicy(sizePolicy)
        self.fr1_group_select.setObjectName("fr1_group_select")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.fr1_group_select)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.btn_confirm = QtWidgets.QPushButton(self.fr1_group_select)
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setObjectName("btn_confirm")
        self.gridLayout.addWidget(self.btn_confirm, 0, 6, 1, 1)
        self.lbl_groupId = QtWidgets.QLabel(self.fr1_group_select)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_groupId.sizePolicy().hasHeightForWidth())
        self.lbl_groupId.setSizePolicy(sizePolicy)
        self.lbl_groupId.setObjectName("lbl_groupId")
        self.gridLayout.addWidget(self.lbl_groupId, 0, 0, 1, 1)
        self.cb_task = QtWidgets.QComboBox(self.fr1_group_select)
        self.cb_task.setEnabled(False)
        self.cb_task.setObjectName("cb_task")
        self.gridLayout.addWidget(self.cb_task, 0, 4, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 2, 1, 1)
        self.cb_groupId = QtWidgets.QComboBox(self.fr1_group_select)
        self.cb_groupId.setObjectName("cb_groupId")
        self.cb_groupId.addItem("")
        self.gridLayout.addWidget(self.cb_groupId, 0, 1, 1, 1)
        self.lbl_task = QtWidgets.QLabel(self.fr1_group_select)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_task.sizePolicy().hasHeightForWidth())
        self.lbl_task.setSizePolicy(sizePolicy)
        self.lbl_task.setObjectName("lbl_task")
        self.gridLayout.addWidget(self.lbl_task, 0, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 5, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addWidget(self.fr1_group_select)
        self.fr2_video_player = QtWidgets.QFrame(self.centralwidget)
        self.fr2_video_player.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr2_video_player.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr2_video_player.setObjectName("fr2_video_player")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.fr2_video_player)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.video_1 = QtWidgets.QLabel(self.fr2_video_player)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_1.sizePolicy().hasHeightForWidth())
        self.video_1.setSizePolicy(sizePolicy)
        self.video_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.video_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.video_1.setText("")
        self.video_1.setObjectName("video_1")
        self.horizontalLayout_3.addWidget(self.video_1)
        spacerItem2 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.video_2 = QtWidgets.QLabel(self.fr2_video_player)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_2.sizePolicy().hasHeightForWidth())
        self.video_2.setSizePolicy(sizePolicy)
        self.video_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.video_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.video_2.setText("")
        self.video_2.setObjectName("video_2")
        self.horizontalLayout_3.addWidget(self.video_2)
        spacerItem3 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.video_3 = QtWidgets.QLabel(self.fr2_video_player)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_3.sizePolicy().hasHeightForWidth())
        self.video_3.setSizePolicy(sizePolicy)
        self.video_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.video_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.video_3.setText("")
        self.video_3.setObjectName("video_3")
        self.horizontalLayout_3.addWidget(self.video_3)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.fr4_controls = QtWidgets.QFrame(self.fr2_video_player)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fr4_controls.sizePolicy().hasHeightForWidth())
        self.fr4_controls.setSizePolicy(sizePolicy)
        self.fr4_controls.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr4_controls.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr4_controls.setObjectName("fr4_controls")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.fr4_controls)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.btn_back = QtWidgets.QPushButton(self.fr4_controls)
        self.btn_back.setEnabled(False)
        self.btn_back.setText("")
        self.btn_back.setObjectName("btn_back")
        self.horizontalLayout_4.addWidget(self.btn_back)
        self.btn_play = QtWidgets.QPushButton(self.fr4_controls)
        self.btn_play.setEnabled(False)
        self.btn_play.setText("")
        self.btn_play.setObjectName("btn_play")
        self.horizontalLayout_4.addWidget(self.btn_play)
        self.btn_skip = QtWidgets.QPushButton(self.fr4_controls)
        self.btn_skip.setEnabled(False)
        self.btn_skip.setText("")
        self.btn_skip.setObjectName("btn_skip")
        self.horizontalLayout_4.addWidget(self.btn_skip)
        self.time_slider = QtWidgets.QSlider(self.fr4_controls)
        self.time_slider.setEnabled(False)
        self.time_slider.setOrientation(QtCore.Qt.Horizontal)
        self.time_slider.setObjectName("time_slider")
        self.horizontalLayout_4.addWidget(self.time_slider)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.verticalLayout_3.addWidget(self.fr4_controls)
        self.verticalLayout_2.addWidget(self.fr2_video_player)
        self.fr3_overlays = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fr3_overlays.sizePolicy().hasHeightForWidth())
        self.fr3_overlays.setSizePolicy(sizePolicy)
        self.fr3_overlays.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.fr3_overlays.setFrameShadow(QtWidgets.QFrame.Raised)
        self.fr3_overlays.setObjectName("fr3_overlays")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.fr3_overlays)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.groupBox = QtWidgets.QGroupBox(self.fr3_overlays)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_5.addWidget(self.label, 0, 0, 1, 1)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.radbtn_raw = QtWidgets.QRadioButton(self.groupBox)
        self.radbtn_raw.setEnabled(True)
        self.radbtn_raw.setObjectName("radbtn_raw")
        self.verticalLayout_10.addWidget(self.radbtn_raw)
        self.radbtn_enh = QtWidgets.QRadioButton(self.groupBox)
        self.radbtn_enh.setEnabled(True)
        self.radbtn_enh.setChecked(True)
        self.radbtn_enh.setObjectName("radbtn_enh")
        self.verticalLayout_10.addWidget(self.radbtn_enh)
        self.gridLayout_5.addLayout(self.verticalLayout_10, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_5.addWidget(self.label_2, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.spn_frame_idx = QtWidgets.QSpinBox(self.groupBox)
        self.spn_frame_idx.setEnabled(False)
        self.spn_frame_idx.setMaximum(99999)
        self.spn_frame_idx.setObjectName("spn_frame_idx")
        self.horizontalLayout_2.addWidget(self.spn_frame_idx)
        self.btn_frame_go = QtWidgets.QPushButton(self.groupBox)
        self.btn_frame_go.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_frame_go.sizePolicy().hasHeightForWidth())
        self.btn_frame_go.setSizePolicy(sizePolicy)
        self.btn_frame_go.setObjectName("btn_frame_go")
        self.horizontalLayout_2.addWidget(self.btn_frame_go)
        self.gridLayout_5.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)
        self.verticalLayout_9.addLayout(self.gridLayout_5)
        self.horizontalLayout_5.addWidget(self.groupBox)
        self.fr1_openpose = QtWidgets.QGroupBox(self.fr3_overlays)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fr1_openpose.sizePolicy().hasHeightForWidth())
        self.fr1_openpose.setSizePolicy(sizePolicy)
        self.fr1_openpose.setObjectName("fr1_openpose")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.fr1_openpose)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.chb_op_pose = QtWidgets.QCheckBox(self.fr1_openpose)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chb_op_pose.sizePolicy().hasHeightForWidth())
        self.chb_op_pose.setSizePolicy(sizePolicy)
        self.chb_op_pose.setObjectName("chb_op_pose")
        self.gridLayout_2.addWidget(self.chb_op_pose, 0, 0, 1, 1)
        self.chb_op_overlap = QtWidgets.QCheckBox(self.fr1_openpose)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chb_op_overlap.sizePolicy().hasHeightForWidth())
        self.chb_op_overlap.setSizePolicy(sizePolicy)
        self.chb_op_overlap.setObjectName("chb_op_overlap")
        self.gridLayout_2.addWidget(self.chb_op_overlap, 2, 0, 1, 1)
        self.chb_op_ig_dist = QtWidgets.QCheckBox(self.fr1_openpose)
        self.chb_op_ig_dist.setObjectName("chb_op_ig_dist")
        self.gridLayout_2.addWidget(self.chb_op_ig_dist, 2, 1, 1, 1)
        self.chb_op_cntr_int = QtWidgets.QCheckBox(self.fr1_openpose)
        self.chb_op_cntr_int.setObjectName("chb_op_cntr_int")
        self.gridLayout_2.addWidget(self.chb_op_cntr_int, 3, 0, 1, 1)
        self.chb_op_face = QtWidgets.QCheckBox(self.fr1_openpose)
        self.chb_op_face.setObjectName("chb_op_face")
        self.gridLayout_2.addWidget(self.chb_op_face, 0, 1, 1, 1)
        self.cb_framework = QtWidgets.QComboBox(self.fr1_openpose)
        self.cb_framework.setObjectName("cb_framework")
        self.cb_framework.addItem("")
        self.cb_framework.addItem("")
        self.gridLayout_2.addWidget(self.cb_framework, 3, 1, 1, 1)
        self.verticalLayout_6.addLayout(self.gridLayout_2)
        self.horizontalLayout_5.addWidget(self.fr1_openpose)
        self.fr2_openface = QtWidgets.QGroupBox(self.fr3_overlays)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fr2_openface.sizePolicy().hasHeightForWidth())
        self.fr2_openface.setSizePolicy(sizePolicy)
        self.fr2_openface.setObjectName("fr2_openface")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.fr2_openface)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBox_3 = QtWidgets.QCheckBox(self.fr2_openface)
        self.checkBox_3.setObjectName("checkBox_3")
        self.gridLayout_3.addWidget(self.checkBox_3, 0, 0, 1, 1)
        self.checkBox_4 = QtWidgets.QCheckBox(self.fr2_openface)
        self.checkBox_4.setObjectName("checkBox_4")
        self.gridLayout_3.addWidget(self.checkBox_4, 1, 0, 1, 1)
        self.verticalLayout_7.addLayout(self.gridLayout_3)
        self.horizontalLayout_5.addWidget(self.fr2_openface)
        self.fr3_video = QtWidgets.QGroupBox(self.fr3_overlays)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fr3_video.sizePolicy().hasHeightForWidth())
        self.fr3_video.setSizePolicy(sizePolicy)
        self.fr3_video.setObjectName("fr3_video")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.fr3_video)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_3 = QtWidgets.QLabel(self.fr3_video)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 1, 0, 1, 1)
        self.chb_vid_energy_htmp = QtWidgets.QCheckBox(self.fr3_video)
        self.chb_vid_energy_htmp.setObjectName("chb_vid_energy_htmp")
        self.gridLayout_4.addWidget(self.chb_vid_energy_htmp, 0, 0, 1, 1)
        self.sld_transparency = QtWidgets.QSlider(self.fr3_video)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sld_transparency.sizePolicy().hasHeightForWidth())
        self.sld_transparency.setSizePolicy(sizePolicy)
        self.sld_transparency.setOrientation(QtCore.Qt.Horizontal)
        self.sld_transparency.setObjectName("sld_transparency")
        self.gridLayout_4.addWidget(self.sld_transparency, 1, 1, 1, 1)
        self.verticalLayout_8.addLayout(self.gridLayout_4)
        self.horizontalLayout_5.addWidget(self.fr3_video)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2.addWidget(self.fr3_overlays)
        Visualizer.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Visualizer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1420, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        Visualizer.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Visualizer)
        self.statusbar.setObjectName("statusbar")
        Visualizer.setStatusBar(self.statusbar)
        self.action_feature_analyzer = QtWidgets.QAction(Visualizer)
        self.action_feature_analyzer.setObjectName("action_feature_analyzer")
        self.actionExit = QtWidgets.QAction(Visualizer)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.action_feature_analyzer)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(Visualizer)
        QtCore.QMetaObject.connectSlotsByName(Visualizer)

    def retranslateUi(self, Visualizer):
        _translate = QtCore.QCoreApplication.translate
        Visualizer.setWindowTitle(_translate("Visualizer", "Visualizer"))
        self.fr1_group_select.setWindowTitle(_translate("Visualizer", "Frame"))
        self.btn_confirm.setText(_translate("Visualizer", "Confirm"))
        self.lbl_groupId.setText(_translate("Visualizer", "Group ID:"))
        self.cb_groupId.setItemText(0, _translate("Visualizer", "Select Group ID"))
        self.lbl_task.setText(_translate("Visualizer", "Task:"))
        self.groupBox.setTitle(_translate("Visualizer", "Video Details"))
        self.label.setText(_translate("Visualizer", "Current Frame:"))
        self.radbtn_raw.setText(_translate("Visualizer", "Raw"))
        self.radbtn_enh.setText(_translate("Visualizer", "Enhanced"))
        self.label_2.setText(_translate("Visualizer", "Data Type:"))
        self.btn_frame_go.setText(_translate("Visualizer", "Go"))
        self.fr1_openpose.setTitle(_translate("Visualizer", "Openpose / Densepose"))
        self.chb_op_pose.setText(_translate("Visualizer", "Pose Lmks"))
        self.chb_op_overlap.setText(_translate("Visualizer", "Overlap"))
        self.chb_op_ig_dist.setText(_translate("Visualizer", "Intragroup Distance"))
        self.chb_op_cntr_int.setText(_translate("Visualizer", "Center Interaction"))
        self.chb_op_face.setText(_translate("Visualizer", "Face Lmks"))
        self.cb_framework.setItemText(0, _translate("Visualizer", "Openpose"))
        self.cb_framework.setItemText(1, _translate("Visualizer", "Densepose"))
        self.fr2_openface.setTitle(_translate("Visualizer", "Openface"))
        self.checkBox_3.setText(_translate("Visualizer", "Face Lmks"))
        self.checkBox_4.setText(_translate("Visualizer", "AUs"))
        self.fr3_video.setTitle(_translate("Visualizer", "Video"))
        self.label_3.setText(_translate("Visualizer", "Overlay Transparency"))
        self.chb_vid_energy_htmp.setText(_translate("Visualizer", "Energy Heatmap Overlay"))
        self.menuFile.setTitle(_translate("Visualizer", "File"))
        self.action_feature_analyzer.setText(_translate("Visualizer", "Feature Analyzer"))
        self.action_feature_analyzer.setShortcut(_translate("Visualizer", "Ctrl+A"))
        self.actionExit.setText(_translate("Visualizer", "Exit"))
        self.actionExit.setShortcut(_translate("Visualizer", "Ctrl+Q"))
