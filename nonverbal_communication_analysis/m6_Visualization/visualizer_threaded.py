import json
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import order_points
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, QThreadPool
import time

from nonverbal_communication_analysis.environment import (
    DATASET_SYNC, FEATURE_AGGREGATE_DIR, OPENPOSE_KEYPOINT_LINKS,
    OPENPOSE_KEYPOINT_MAP, VALID_OUTPUT_FILE_TYPES, VALID_OUTPUT_IMG_TYPES,
    VIDEO_RESOLUTION)
from nonverbal_communication_analysis.m0_Classes.Subject import COLOR_MAP
from nonverbal_communication_analysis.m6_Visualization.visualizer_gui import \
    Ui_Visualizer


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress 

    '''
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)


class VideoPlayerThread(QtWidgets.QWidget):
    def __init__(self, tid, filename, video_frame, timer, group_data_path, group_feature_data: dict):
        super(QtWidgets.QWidget, self).__init__()
        self.thread_id = tid
        self.camera = 'pc%s' % tid

        self.cap = cv2.VideoCapture(str(filename))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0

        self.video_frame = video_frame

        self.group_data_path = group_data_path
        self.group_feature_data = group_feature_data

        self.is_playing = False
        self.heatmap_overlay_transparency = 0.5

        self.signals = WorkerSignals()
        self.progress_callback = self.signals.progress

    def play(self, skip: bool = False):
        print("Player thread", self.thread_id)
        self.frame_count += 1
        self.progress_callback.emit(self.frame_count - self.length == 0)


class VideoPlayerMonitor(object):
    def __init__(self, visualizer: QtWidgets.QMainWindow, ui: Ui_Visualizer, groups_directories: dict, q_components: dict):
        print("Starting Monitor")
        self.ui = ui
        self.groups_directories = groups_directories
        self.q_components = q_components

        group_select_disable = [self.ui.cb_groupId,
                                self.ui.cb_task,
                                self.ui.btn_confirm]
        group_select_enable = [self.ui.spn_frame_idx,
                               self.ui.btn_frame_go]
        self.ui.cb_groupId.currentIndexChanged.connect(self.combo_on_select)
        self.ui.btn_confirm.clicked.connect(lambda: self.group_select_confirm(group_select_disable,
                                                                              group_select_enable))

        self.selected_group = None
        self.selected_task = None

        self.current_frame = 0
        self.is_playing = False

        self.timer = QtCore.QTimer()
        self.timer.start(1000.0/30)

        visualizer.show()

    def play(self):
        print("Monitor loop", self.is_playing)
        if self.is_playing:
            time.sleep(1)
            self.current_frame += 1

    def combo_on_select(self):
        cb_group = self.ui.cb_groupId
        cb_task = self.ui.cb_task
        btn_confirm = self.ui.btn_confirm

        if int(cb_group.currentIndex()) != 0:
            cb_task.setEnabled(True)
            cb_task.clear()
            cb_task.addItems([x.name for x in self.groups_directories[cb_group.currentText()].iterdir()
                              if x.is_dir() and 'task' in x.name])
            btn_confirm.setEnabled(True)
        else:
            cb_task.setEnabled(False)
            cb_task.clear()
            btn_confirm.setEnabled(False)

    def group_select_confirm(self, disable_widgets, enable_widgets):
        for widget in disable_widgets:
            widget.setEnabled(False)

        for widget in enable_widgets:
            widget.setEnabled(True)

        self.initialize_player_threads()

    def initialize_player_threads(self):
        self.selected_group = self.ui.cb_groupId.currentText()
        selected_group_dir = self.groups_directories[self.selected_group]

        self.selected_task = self.ui.cb_task.currentText()
        selected_group_task = selected_group_dir / self.selected_task
        selected_group_videos = [x for x in selected_group_task.iterdir()
                                 if not x.is_dir()]

        feature_data_path = DATASET_SYNC / self.selected_group / \
            FEATURE_AGGREGATE_DIR / self.selected_task

        group_feature_data = {int(re.search(r'(\d{12})', x.name).group(0)): x
                              for x in feature_data_path.iterdir()
                              if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES}

        for video in selected_group_videos:

            if 'pc1' in video.name:
                self.ui.video_1 = VideoPlayerThread(1, video, self.ui.video_1, self.timer,
                                                    feature_data_path, group_feature_data)
                self.player_1 = self.ui.video_1
                self.worker_player_1 = QThread()
                self.worker_player_1.started.connect(self.player_1.play)

            elif 'pc2' in video.name:
                self.ui.video_2 = VideoPlayerThread(2, video, self.ui.video_2, self.timer,
                                                    feature_data_path, group_feature_data)
                self.player_2 = self.ui.video_2
                self.worker_player_2 = QThread()
                self.worker_player_2.started.connect(self.player_2.play)

            elif 'pc3' in video.name:
                self.ui.video_3 = VideoPlayerThread(3, video, self.ui.video_3, self.timer,
                                                    feature_data_path, group_feature_data)
                self.player_3 = self.ui.video_3
                self.worker_player_3 = QThread()
                self.worker_player_3.started.connect(self.player_3.play)

        self.ui.btn_play.setEnabled(True)
        self.ui.btn_back.setEnabled(True)
        self.ui.btn_skip.setEnabled(True)
        self.ui.time_slider.setEnabled(True)

        self.worker_player_1.start()
        self.worker_player_2.start()
        self.worker_player_3.start()

        self.timer.timeout.connect(self.play)


class Visualizer(object):
    """
    Visualizer is the Main class of the Visualization Tool. This class initializes the GUI 
     and its required initial variables.
    Visualizer instanciates the Monitor which is responsible to manipulate the VideoPlayer 
     threads and the feature analyzer tab states.
    """

    def __init__(self):
        self.playing_status = False
        self.group_dirs = {x.name: x for x in DATASET_SYNC.iterdir()
                           if x.is_dir()}
        self.group_id = None
        self.q_components = dict()

    def main(self):
        app = QtWidgets.QApplication(sys.argv)
        visualizer = QtWidgets.QMainWindow()
        self.ui = Ui_Visualizer()
        self.ui.setupUi(visualizer)

        # Window Components
        # self.ui.action_feature_analyzer.triggered.connect()
        self.ui.actionExit.triggered.connect(self.close_application)

        # Group Frame Components
        self.ui.cb_groupId.addItems(list(self.group_dirs.keys()))

        # Controls Frame Components
        btn_play = self.ui.btn_play
        btn_back = self.ui.btn_back
        btn_skip = self.ui.btn_skip

        btn_play.setIcon(visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay))
        btn_back.setIcon(visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_ArrowBack))
        btn_skip.setIcon(visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_ArrowForward))

        self.ui.sld_transparency.setMinimum(0)
        self.ui.sld_transparency.setMaximum(10)
        self.ui.sld_transparency.setValue(5)
        self.ui.sld_transparency.setTickInterval(1)
        self.ui.sld_transparency.setTickPosition(5)

        self.q_components['play_btn'] = self.ui.btn_play
        self.q_components['back_btn'] = self.ui.btn_back
        self.q_components['skip_btn'] = self.ui.btn_skip
        self.q_components['slider'] = self.ui.time_slider
        self.q_components['chb_op_pose'] = self.ui.chb_op_pose
        self.q_components['chb_op_face'] = self.ui.chb_op_face
        self.q_components['chb_op_exp'] = self.ui.chb_op_exp
        self.q_components['chb_op_cntr_int'] = self.ui.chb_op_cntr_int
        self.q_components['chb_op_ig_dist'] = self.ui.chb_op_ig_dist
        self.q_components['chb_vid_energy_htmp'] = self.ui.chb_vid_energy_htmp
        self.q_components['spn_frame_idx'] = self.ui.spn_frame_idx
        self.q_components['frame_goto_btn'] = self.ui.btn_frame_go
        self.q_components['radbtn_raw'] = self.ui.radbtn_raw
        self.q_components['radbtn_enhanced'] = self.ui.radbtn_enh
        self.q_components['sld_overlay_transp'] = self.ui.sld_transparency

        monitor_GUI = VideoPlayerMonitor(
            visualizer, self.ui, self.group_dirs, self.q_components)

        sys.exit(app.exec_())

    def close_application(self):
        sys.exit()


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.main()
