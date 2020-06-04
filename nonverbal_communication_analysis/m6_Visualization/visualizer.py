import json
import re
import sys
import numpy as np

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer

from nonverbal_communication_analysis.environment import (
    DATASET_SYNC, FEATURE_AGGREGATE_DIR, VALID_OUTPUT_FILE_TYPES,
    VIDEO_RESOLUTION, OPENPOSE_KEYPOINT_LINKS, OPENPOSE_KEYPOINT_MAP)
from nonverbal_communication_analysis.m6_Visualization.visualizer_gui import \
    Ui_Visualizer
from nonverbal_communication_analysis.m0_Classes.Subject import COLOR_MAP


class VideoPlayerManager(object):
    def __init__(self):
        pass


class VideoCapture(QtWidgets.QWidget):

    # TODO: fix bug: Play button not changing style after video finish automatically.
    # Need to press 2 times to replay
    def __init__(self, tid, filename, video_frame, group_feature_data: dict, q_components: dict):
        super(QtWidgets.QWidget, self).__init__()
        self.thread_id = tid,
        self.camera = 'pc%s' % tid
        self.group_feature_data = group_feature_data
        self.cap = cv2.VideoCapture(str(filename))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_frame = video_frame
        self.frame_count = 0
        self.is_playing = False
        self.visualizer = visualizer
        self.slider = q_components['slider']
        self.play_btn = q_components['play_btn']
        self.back_btn = q_components['back_btn']
        self.skip_btn = q_components['skip_btn']
        self.slider = q_components['slider']
        self.chb_op_pose = q_components['chb_op_pose']
        self.chb_op_face = q_components['chb_op_face']
        self.chb_op_exp = q_components['chb_op_exp']
        self.chb_op_cntr_int = q_components['chb_op_cntr_int']
        self.chb_op_ig_dist = q_components['chb_op_ig_dist']
        self.lbl_frame_idx = q_components['lbl_frame_idx']
        self.radbtn_raw = q_components['radbtn_raw']
        self.radbtn_enhanced = q_components['radbtn_enhanced']

    def nextFrameSlot(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
        ret, frame = self.cap.read()
        self.lbl_frame_idx.setText(str(self.frame_count))
        self.frame_count += 1
        print("FRAME: ", self.frame_count)

        if ret and self.is_playing:
            self.is_playing = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = self.frame_transform(frame)
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0],
                               QtGui.QImage.Format_RGBA8888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setScaledContents(True)
            self.video_frame.setPixmap(pix)

            slider_position = round(self.frame_count/self.length * 100)
        else:
            self.is_playing = False
            self.frame_count = 0
            slider_position = 0
        self.slider.setValue(slider_position)

    def openpose_overlay_video(self, openpose_data: dict, img_frame: np.ndarray, camera: str):
        subject_id = openpose_data['id'] if 'id' in openpose_data else 0
        del openpose_data['id']

        for key, data in openpose_data.items():
            for keypoint_idx, keypoint_values in data.items():
                keypoint_x = round(
                    keypoint_values[0] * VIDEO_RESOLUTION[camera]['x'])
                keypoint_y = round(
                    keypoint_values[1] * VIDEO_RESOLUTION[camera]['y'])
                keypoint_c = round(keypoint_values[2] * 5)

                cv2.circle(img_frame, (keypoint_x, keypoint_y),
                           keypoint_c, COLOR_MAP[subject_id], -1)

                # Connected joints
                if key == 'pose':
                    keypoint_idx = int(keypoint_idx)
                    if keypoint_idx in OPENPOSE_KEYPOINT_LINKS:
                        for keypoint_link_idx in OPENPOSE_KEYPOINT_LINKS[keypoint_idx]:
                            pose_keypoints = openpose_data['pose']
                            if str(keypoint_link_idx) in pose_keypoints:
                                keypoint = pose_keypoints[str(
                                    keypoint_link_idx)]
                                keypoint_link_x = round(
                                    keypoint[0] * VIDEO_RESOLUTION[camera]['x'])
                                keypoint_link_y = round(
                                    keypoint[1] * VIDEO_RESOLUTION[camera]['y'])

                                if keypoint_x == 0 or keypoint_y == 0 or keypoint_link_x == 0 or keypoint_link_y == 0:
                                    break

                                cv2.line(img_frame, (keypoint_x, keypoint_y),
                                         (keypoint_link_x, keypoint_link_y), COLOR_MAP[subject_id], 2)

        return img_frame

    def frame_transform(self, frame):

        # Read FEATURE_DATA
        current_frame = self.frame_count
        if current_frame not in self.group_feature_data:
            return frame

        frame_data = json.load(
            open(self.group_feature_data[current_frame], 'r'))
        frame_id = frame_data['frame']
        is_raw_data_valid = frame_data['is_raw_data_valid']
        is_enhanced_data_valid = frame_data['is_enhanced_data_valid']
        group_data = frame_data['group']
        subjects_data = frame_data['subjects']

        openpose_data = dict()
        densepose_data = dict()
        openface_data = dict()
        video_metrics_data = dict()

        data_type = None

        if self.radbtn_raw.isChecked():
            data_type = 'raw'
        elif self.radbtn_enhanced.isChecked():
            data_type = 'enhanced'
        else:
            print(
                "Error", "Invalid type of feature data. must be {raw, enhanced}")

        for subject in subjects_data:
            # Openpose
            # Pose
            if self.chb_op_pose.isChecked():
                openpose_data['id'] = subject['id']
                subject_type_data = subject[data_type]
                subject_type_data_pose = subject_type_data['pose'] \
                    if 'pose' in subject_type_data else dict()
                subject_type_data_pose_openpose = subject_type_data_pose['openpose'] \
                    if 'openpose' in subject_type_data_pose else dict()
                openpose_data['pose'] = subject_type_data_pose_openpose[self.camera] \
                    if self.camera in subject_type_data_pose_openpose else dict()

                frame = self.openpose_overlay_video(
                    openpose_data, frame, self.camera)

            # # Face
            # if self.chb_op_face.isChecked():
            #     openpose_data['face'] = subject['face']['openpose'] if 'face' in subject else dict(
            #     )

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
        self.group_id = group_id
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

    def group_select_confirm(self, disable_widgets, enable_widgets):
        for widget in disable_widgets:
            widget.setEnabled(False)

        for widget in enable_widgets:
            widget.setEnabled(True)

        self.load_group_videos()

    # PLAYER FRAMES METHODS
    def load_group_videos(self):
        selected_group = self.ui.cb_groupId.currentText()
        selected_group_dir = self.group_dirs[selected_group]
        selected_task = self.ui.cb_task.currentText()
        self.selected_task = selected_task
        selected_group_task = selected_group_dir / selected_task
        selected_group_videos = [x for x in selected_group_task.iterdir()
                                 if not x.is_dir()]

        q_components = dict()
        q_components['play_btn'] = self.ui.btn_play
        q_components['back_btn'] = self.ui.btn_back
        q_components['skip_btn'] = self.ui.btn_skip
        q_components['slider'] = self.ui.time_slider
        q_components['chb_op_pose'] = self.ui.chb_op_pose
        q_components['chb_op_face'] = self.ui.chb_op_face
        q_components['chb_op_exp'] = self.ui.chb_op_exp
        q_components['chb_op_cntr_int'] = self.ui.chb_op_cntr_int
        q_components['chb_op_ig_dist'] = self.ui.chb_op_ig_dist
        q_components['lbl_frame_idx'] = self.ui.lbl_frame_idx
        q_components['radbtn_raw'] = self.ui.radbtn_raw
        q_components['radbtn_enhanced'] = self.ui.radbtn_enh

        for video in selected_group_videos:
            video_name = video.name

            feature_data_path = DATASET_SYNC / self.group_id / \
                FEATURE_AGGREGATE_DIR / self.selected_task

            group_feature_data = {int(re.search(r'(\d{12})', x.name).group(0)): x
                                  for x in feature_data_path.iterdir()
                                  if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES}

            if 'pc1' in video_name:
                self.ui.video_1 = VideoCapture(1, str(video),
                                               self.ui.video_1, group_feature_data,
                                               q_components)
                self.player_1 = self.ui.video_1
            elif 'pc2' in video_name:
                self.ui.video_2 = VideoCapture(2, str(video),
                                               self.ui.video_2, group_feature_data,
                                               q_components)
                self.player_2 = self.ui.video_2

            elif 'pc3' in video_name:
                self.ui.video_3 = VideoCapture(3, str(video),
                                               self.ui.video_3, group_feature_data,
                                               q_components)
                self.player_3 = self.ui.video_3

        self.ui.time_slider.setEnabled(True)
        self.ui.btn_play.setEnabled(True)
        self.ui.btn_back.setEnabled(True)
        self.ui.btn_skip.setEnabled(True)

    # OVERLAYS FRAMES METHODS

    # CONTROLS FRAMES METHODS
    def video_play(self):
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

    def set_positions(self, position):
        print(position)

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
        spinner_frame_idx = self.ui.lbl_frame_idx
        group_select_disable = [cb_group_id, cb_task, btn_confirm]
        group_select_enable = []

        cb_group_id.addItems(
            ['Select Group ID']+list(self.group_dirs.keys()))
        cb_group_id.currentIndexChanged.connect(self.combo_on_select)
        btn_confirm.clicked.connect(
            lambda: self.group_select_confirm(group_select_disable, group_select_enable))

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
