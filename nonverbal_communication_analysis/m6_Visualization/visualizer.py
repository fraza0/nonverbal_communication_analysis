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


class VideoPlayerMonitor(object):
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
        self.play_btn = q_components['play_btn']
        self.skip_btn = q_components['skip_btn']
        self.skip_btn.clicked.connect(lambda: self.jump_frames(10))
        self.back_btn = q_components['back_btn']
        self.back_btn.clicked.connect(lambda: self.jump_frames(-10))
        self.slider = q_components['slider']
        self.slider.sliderMoved.connect(self.set_frame_by_slider)
        self.chb_op_pose = q_components['chb_op_pose']
        self.chb_op_face = q_components['chb_op_face']
        self.chb_op_exp = q_components['chb_op_exp']
        self.chb_op_cntr_int = q_components['chb_op_cntr_int']
        self.chb_op_ig_dist = q_components['chb_op_ig_dist']
        self.spn_frame_idx = q_components['spn_frame_idx']
        self.btn_frame_goto = q_components['frame_goto_btn']
        self.btn_frame_goto.clicked.connect(self.set_frame_by_spinner)
        self.radbtn_raw = q_components['radbtn_raw']
        self.radbtn_enhanced = q_components['radbtn_enhanced']

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.play)

    def play(self, skip: bool = False):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
        ret, frame = self.cap.read()
        self.spn_frame_idx.setValue(self.frame_count)

        if ret and (self.is_playing or skip):
            self.is_playing = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = self.frame_transform(frame)
            self.show_frame(frame)
            slider_position = round(self.frame_count/self.length * 100)
        else:
            self.is_playing = False
            self.frame_count = 0
            slider_position = 0

        self.slider.setValue(slider_position)
        self.frame_count += 1

    def show_frame(self, frame):
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0],
                           QtGui.QImage.Format_RGBA8888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setScaledContents(True)
        self.video_frame.setPixmap(pix)

    def set_frame_by_slider(self, position):
        total_frames = self.length
        frame_position = round(position*total_frames/100)
        self.frame_count = frame_position
        self.update_frame()

    def set_frame_by_spinner(self):
        self.frame_count = int(self.spn_frame_idx.value())
        self.update_frame()

    def jump_frames(self, increment):
        self.frame_count += increment-1
        if self.frame_count < 0:
            self.frame_count = 0
        self.update_frame()

    def update_frame(self):
        self.play(skip=True)
        self.pause()

    def openpose_overlay(self, subject_id: int, subject_data: dict, img_frame: np.ndarray, camera: str):
        for key, data in subject_data.items():
            if key == 'pose' or key == 'face':
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
                                pose_keypoints = subject_data['pose']
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
                                             (keypoint_link_x, keypoint_link_y), COLOR_MAP[subject_id], 1)
            elif key == 'expansiveness':
                if camera in data:
                    expansiveness_data = data[camera]
                    xmin = round(
                        expansiveness_data['x'][0] * VIDEO_RESOLUTION[camera]['x'])
                    xmax = round(
                        expansiveness_data['x'][1] * VIDEO_RESOLUTION[camera]['x'])
                    ymin = round(
                        expansiveness_data['y'][0] * VIDEO_RESOLUTION[camera]['y'])
                    ymax = round(
                        expansiveness_data['y'][1] * VIDEO_RESOLUTION[camera]['y'])

                    cv2.rectangle(img_frame, (xmin, ymax), (xmax, ymin),
                                  COLOR_MAP[subject_id])

            elif key == 'intragroup_distance':
                color = (163, 32, 219, 255)
                if camera in data:
                    intragroup_distance_data = data[camera]
                    intragroup_distance_center = intragroup_distance_data['center']
                    intragroup_distance_polygon = intragroup_distance_data['polygon']
                    center_x = round(intragroup_distance_center[0] *
                                     VIDEO_RESOLUTION[camera]['x'])
                    center_y = round(intragroup_distance_center[1] *
                                     VIDEO_RESOLUTION[camera]['y'])

                    cv2.drawMarker(img_frame, (center_x, center_y), color,
                                   markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

                    vertices = list()
                    for point in intragroup_distance_polygon:
                        point_x, point_y = point[0], point[1]
                        point_x = round(
                            point_x * VIDEO_RESOLUTION[camera]['x'])
                        point_y = round(
                            point_y * VIDEO_RESOLUTION[camera]['y'])

                        point_int = [point_x, point_y]
                        if point_int not in vertices:
                            vertices.append(point_int)

                    vertices = np.array(vertices, np.int32)
                    vertices = np.array(order_points(vertices), np.int32)
                    cv2.polylines(img_frame, [vertices],
                                  True, color)

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

        frame_subject_data = {
            'openpose': dict(),
            'densepose': dict(),
            'openface': dict(),
            'video': dict()
        }

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
            subject_id = subject['id']
            frame_subject_data['id'] = subject_id
            # Pose
            if self.chb_op_pose.isChecked():
                subject_type_data = subject[data_type]
                subject_type_data_pose = subject_type_data['pose'] \
                    if 'pose' in subject_type_data else dict()
                subject_type_data_pose_openpose = subject_type_data_pose['openpose'] \
                    if 'openpose' in subject_type_data_pose else dict()
                openpose_data['pose'] = subject_type_data_pose_openpose[self.camera] \
                    if self.camera in subject_type_data_pose_openpose else dict()

            # Face
            if self.chb_op_face.isChecked():
                subject_type_data = subject[data_type]
                subject_type_data_face = subject_type_data['face'] \
                    if 'face' in subject_type_data else dict()
                subject_type_data_face_openpose = subject_type_data_face['openpose'] \
                    if 'openpose' in subject_type_data_face else dict()
                openpose_data['face'] = subject_type_data_face_openpose[self.camera] \
                    if self.camera in subject_type_data_face_openpose else dict()

            # Expansiveness
            if self.chb_op_exp.isChecked():
                openpose_data['expansiveness'] = subject['metrics']['expansiveness']

                if self.frame_count == 178:
                    print(subject['id'])

            # Intragroup Distance
            if self.chb_op_ig_dist.isChecked():
                openpose_data['intragroup_distance'] = frame_data['group']['intragroup_distance']

            frame_subject_data['openpose'] = openpose_data
            frame = self.openpose_overlay(subject_id, frame_subject_data['openpose'],
                                          frame, self.camera)

        return frame

    def densepose_overlay(self):
        print("Densepose Overlay")

    def openface_overlay(self):
        print("Openface Overlay")

    def video_overlay(self):
        print("Video Overlay")

    def start(self):
        self.is_playing = True
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
        q_components['spn_frame_idx'] = self.ui.spn_frame_idx
        q_components['frame_goto_btn'] = self.ui.btn_frame_go
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
                self.player_1.update_frame()

            elif 'pc2' in video_name:
                self.ui.video_2 = VideoCapture(2, str(video),
                                               self.ui.video_2, group_feature_data,
                                               q_components)
                self.player_2 = self.ui.video_2
                self.player_2.update_frame()

            elif 'pc3' in video_name:
                self.ui.video_3 = VideoCapture(3, str(video),
                                               self.ui.video_3, group_feature_data,
                                               q_components)
                self.player_3 = self.ui.video_3
                self.player_3.update_frame()

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
        spn_frame_idx = self.ui.spn_frame_idx
        btn_frame_goto = self.ui.btn_frame_go
        group_select_disable = [cb_group_id, cb_task, btn_confirm]
        group_select_enable = [spn_frame_idx, btn_frame_goto]

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

        # time_slider.sliderMoved.connect(self.set_position)

        self.visualizer.show()
        sys.exit(app.exec_())

    def exit_application(self):
        sys.exit()


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.main()
