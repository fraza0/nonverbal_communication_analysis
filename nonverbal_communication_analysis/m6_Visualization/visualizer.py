import sys

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from nonverbal_communication_analysis.environment import DATASET_SYNC
from nonverbal_communication_analysis.m6_Visualization.visualizer_gui import Ui_Visualizer


class VideoCapture(QtWidgets.QWidget):

    # TODO: fix bug: Play button not changing style after video finish automatically.
    # Need to press 2 times to replay
    def __init__(self, filename, video_frame, slider):
        super(QtWidgets.QWidget, self).__init__()
        self.cap = cv2.VideoCapture(str(filename))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        self.video_frame = video_frame
        self.slider = slider
        self.is_playing = False

    def nextFrameSlot(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
        ret, frame = self.cap.read()

        if ret and self.is_playing:
            self.is_playing = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = self.frame_transform(frame)
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0],
                               QtGui.QImage.Format_RGBA8888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setScaledContents(True)
            self.video_frame.setPixmap(pix)
            self.frame_count += 1

            slider_position = round(self.frame_count/self.length * 100)
        else:
            self.is_playing = False
            self.frame_count = 0
            slider_position = 0
        self.slider.setValue(slider_position)

    def frame_transform(self, frame, openpose: bool = False, densepose: bool = False, openface: bool = False, video: bool = False):

        if openpose:
            self.openpose_overlay()

        if densepose:
            self.densepose_overlay()

        if openface:
            self.openface_overlay()

        if video:
            self.video_overlay()

        return frame

    def openpose_overlay(self):
        print("Openpose Overlay")

    def densepose_overlay(self):
        print("Densepose Overlay")

    def openface_overlay(self):
        print("Openface Overlay")

    def video_overlay(self):
        print("Video Overlay")

    def start(self):
        self.is_playing = True
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000.0/30)

    def pause(self):
        self.is_playing = False
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QtWidgets.QWidget, self).deleteLater()


class Visualizer(object):

    def __init__(self):
        self.playing_status = False
        self.group_dirs = {x.name: x for x in DATASET_SYNC.iterdir()
                           if x.is_dir()}

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
                self.ui.video_1 = VideoCapture(
                    str(video), self.ui.video_1, self.ui.time_slider)
                self.player_1 = self.ui.video_1
            elif 'pc2' in video_name:
                self.ui.video_2 = VideoCapture(
                    str(video), self.ui.video_2, self.ui.time_slider)
                self.player_2 = self.ui.video_2

            elif 'pc3' in video_name:
                self.ui.video_3 = VideoCapture(
                    str(video), self.ui.video_3, self.ui.time_slider)
                self.player_3 = self.ui.video_3

        self.ui.time_slider.setEnabled(True)
        self.ui.btn_play.setEnabled(True)
        self.ui.btn_back.setEnabled(True)
        self.ui.btn_skip.setEnabled(True)

    # OVERLAYS FRAMES METHODS

    # CONTROLS FRAMES METHODS
    def video_play(self):
        print(self.ui.video_1.frame_count)
        self.ui.video_1.start()
        self.ui.video_2.start()
        self.ui.video_3.start()
        self.ui.btn_play.setIcon(self.visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPause))
        self.playing_status = True

    def video_pause(self):
        self.ui.video_1.pause()
        self.ui.video_2.pause()
        self.ui.video_3.pause()
        self.ui.btn_play.setIcon(self.visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay))
        self.playing_status = False

    def videos_state(self):
        if not self.playing_status:
            self.video_play()
        else:
            self.video_pause()

    def set_position(self, position):
        total_frames = self.player_1.length
        frame_position = round(position*total_frames/100)

        self.player_1.frame_count = frame_position
        self.player_2.frame_count = frame_position
        self.player_3.frame_count = frame_position

    def main(self):
        app = QtWidgets.QApplication(sys.argv)
        self.visualizer = QtWidgets.QMainWindow()
        self.ui = Ui_Visualizer()
        self.ui.setupUi(self.visualizer)

        self.ui.actionExit.triggered.connect(self.exit_application)

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
        self.player_1 = None
        self.player_2 = None
        self.player_3 = None

        # Controls Frame
        btn_play = self.ui.btn_play
        btn_back = self.ui.btn_back
        btn_skip = self.ui.btn_skip
        time_slider = self.ui.time_slider
        btn_play.setIcon(self.visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay))
        btn_play.clicked.connect(self.videos_state)
        btn_back.setIcon(self.visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_ArrowBack))
        btn_skip.setIcon(self.visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_ArrowForward))

        time_slider.sliderMoved.connect(self.set_position)

        self.visualizer.show()
        sys.exit(app.exec_())

    def exit_application(self):
        sys.exit()


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.main()
