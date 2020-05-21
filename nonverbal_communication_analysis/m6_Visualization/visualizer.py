import sys

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from nonverbal_communication_analysis.environment import DATASET_SYNC
from nonverbal_communication_analysis.m6_Visualization.visualizer_gui import Ui_Visualizer


class Visualizer:

    def __init__(self):
        self.playing_status = False
        self.group_dirs = {
            x.name: x for x in DATASET_SYNC.iterdir() if x.is_dir()}

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

    # GROUP FRAME METHODS
    def group_tasks(self, group_id):
        return [x.name for x in self.group_dirs[group_id].iterdir()
                if x.is_dir() and 'task' in x.name]

    def combo_on_select(self):
        cb_group = self.ui.cb_groupId
        cb_task = self.ui.cb_task
        btn_confirm = self.ui.btn_confirm
        group_cb_idx = int(cb_group.currentIndex())
        # task_cb_idx = int(cb_task.currentIndex())

        if group_cb_idx != 0:
            cb_task.setEnabled(True)
            cb_task.clear()
            cb_task.addItems(self.group_tasks(cb_group.currentText()))
            btn_confirm.setEnabled(True)

        else:
            cb_task.clear()
            cb_task.setEnabled(False)
            btn_confirm.setEnabled(False)

    def group_select_confirm(self, widgets):
        for widget in widgets:
            widget.setEnabled(False)

        self.load_group_videos()

    # PLAYER FRAMES METHODS
    def load_group_videos(self):
        selected_group = self.ui.cb_groupId.currentText()
        selected_group_dir = self.group_dirs[selected_group]
        selected_task = self.ui.cb_task.currentText()
        selected_group_task = selected_group_dir / selected_task
        selected_group_videos = [x for x in selected_group_task.iterdir()
                                 if not x.is_dir()]
        for video in selected_group_videos:
            video_name = video.name
            if 'pc1' in video_name:
                self.player_1.setMedia(QMediaContent(
                    QtCore.QUrl.fromLocalFile(str(video))))
                self.player_1.setVideoOutput(self.ui.video_1)
            elif 'pc2' in video_name:
                self.player_2.setMedia(QMediaContent(
                    QtCore.QUrl.fromLocalFile(str(video))))
                self.player_2.setVideoOutput(self.ui.video_2)
            elif 'pc3' in video_name:
                self.player_3.setMedia(QMediaContent(
                    QtCore.QUrl.fromLocalFile(str(video))))
                self.player_3.setVideoOutput(self.ui.video_3)

        self.ui.time_slider.setEnabled(True)
        self.ui.btn_play.setEnabled(True)
        self.ui.btn_back.setEnabled(True)
        self.ui.btn_skip.setEnabled(True)

    # OVERLAYS FRAMES METHODS

    # CONTROLS FRAMES METHODS
    def videos_state(self, visualizer, btn, playing_status):
        if not playing_status:
            self.player_1.play()
            self.player_2.play()
            self.player_3.play()
            btn.setIcon(visualizer.style().standardIcon(
                QtWidgets.QStyle.SP_MediaPause))
            self.playing_status = True
        else:
            self.player_1.pause()
            self.player_2.pause()
            self.player_3.pause()
            btn.setIcon(visualizer.style().standardIcon(
                QtWidgets.QStyle.SP_MediaPlay))
            self.playing_status = False

    def position_changed(self, position):
        self.ui.time_slider.setValue(position)

    def duration_changed(self, duration):
        self.ui.time_slider.setRange(0, duration)

    def set_position(self, position):
        self.player_1.setPosition(position)
        self.player_2.setPosition(position)
        self.player_3.setPosition(position)

    def main(self):
        app = QtWidgets.QApplication(sys.argv)
        Visualizer = QtWidgets.QMainWindow()
        self.ui = Ui_Visualizer()
        self.ui.setupUi(Visualizer)

        # Group Frame
        cb_group_id = self.ui.cb_groupId
        cb_task = self.ui.cb_task
        btn_confirm = self.ui.btn_confirm
        group_select_frame = [cb_group_id, cb_task, btn_confirm]

        cb_group_id.addItems(
            ['Select Group ID']+list(self.group_dirs.keys()))
        cb_group_id.currentIndexChanged.connect(self.combo_on_select)
        btn_confirm.clicked.connect(
            lambda: self.group_select_confirm(group_select_frame))

        # Player Frames
        self.player_1 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player_2 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player_3 = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.player_1.positionChanged.connect(self.position_changed)
        self.player_1.durationChanged.connect(self.duration_changed)

        self.player_2.positionChanged.connect(self.position_changed)
        self.player_2.durationChanged.connect(self.duration_changed)

        self.player_3.positionChanged.connect(self.position_changed)
        self.player_3.durationChanged.connect(self.duration_changed)

        # Controls Frame
        btn_play = self.ui.btn_play
        btn_back = self.ui.btn_back
        btn_skip = self.ui.btn_skip
        time_slider = self.ui.time_slider
        btn_play.setIcon(Visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay))
        btn_play.clicked.connect(
            lambda: self.videos_state(Visualizer, btn_play, self.playing_status))
        btn_back.setIcon(Visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_ArrowBack))
        btn_skip.setIcon(Visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_ArrowForward))

        time_slider.sliderMoved.connect(self.set_position)

        Visualizer.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.main()
