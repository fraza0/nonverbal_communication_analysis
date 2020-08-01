import json
import re
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import order_points
from PyQt5 import QtCore, QtGui, QtWidgets

from nonverbal_communication_analysis.environment import (
    DATASET_SYNC, FEATURE_AGGREGATE_DIR, OPENPOSE_KEYPOINT_LINKS,
    OPENPOSE_KEYPOINT_MAP, VALID_OUTPUT_FILE_TYPES, VALID_OUTPUT_IMG_TYPES,
    VIDEO_RESOLUTION)
from nonverbal_communication_analysis.utils import vertices_from_polygon
from nonverbal_communication_analysis.m0_Classes.Subject import COLOR_MAP
from nonverbal_communication_analysis.m6_Visualization.visualizer_gui import \
    Ui_Visualizer
from nonverbal_communication_analysis.m6_Visualization.feature_analyzer import FeatureAnalyzer


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, tid, filename, video_frame, timer, group_data_path, group_feature_data: dict):
        super(QtWidgets.QWidget, self).__init__()
        self.thread_id = tid
        self.camera = 'pc%s' % tid

        self.cap = cv2.VideoCapture(str(filename))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_idx = 0
        self.progress = 0.0

        self.video_frame = video_frame

        self.group_data_path = group_data_path
        self.group_feature_data = group_feature_data

        self.overlay_heatmap_transparency = 0.5

        self.gui_state = None

    def play_step(self, frame_idx: int = 0, gui_state: dict = None):
        self.frame_idx = frame_idx
        self.gui_state = gui_state
        self.overlay_heatmap_transparency = gui_state['overlay_heatmap_transparency']
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = self.frame_transform(frame)
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0],
                               QtGui.QImage.Format_RGBA8888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setScaledContents(True)
            self.video_frame.setPixmap(pix)

    def openpose_overlay(self, subject_id: int, subject_data: dict, img_frame: np.ndarray, camera: str):
        # print(subject_data)
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
                    exp_xmin = round(
                        expansiveness_data['x'][0] * VIDEO_RESOLUTION[camera]['x'])
                    exp_xmax = round(
                        expansiveness_data['x'][1] * VIDEO_RESOLUTION[camera]['x'])
                    exp_ymin = round(
                        expansiveness_data['y'][0] * VIDEO_RESOLUTION[camera]['y'])
                    exp_ymax = round(
                        expansiveness_data['y'][1] * VIDEO_RESOLUTION[camera]['y'])

                    cv2.rectangle(img_frame, (exp_xmin, exp_ymax), (exp_xmax, exp_ymin),
                                  COLOR_MAP[subject_id])

            elif key == 'overlap':
                color = (252, 239, 93, 255)
                if camera in data:
                    overlap_data = data[camera]
                    vertices = vertices_from_polygon(
                        overlap_data['polygon'])

                    ovl_xmin = round(
                        vertices['x']['min'] * VIDEO_RESOLUTION[camera]['x'])
                    ovl_xmax = round(
                        vertices['x']['max'] * VIDEO_RESOLUTION[camera]['x'])
                    ovl_ymin = round(
                        vertices['y']['min'] * VIDEO_RESOLUTION[camera]['y'])
                    ovl_ymax = round(
                        vertices['y']['max'] * VIDEO_RESOLUTION[camera]['y'])

                    cv2.rectangle(img_frame, (ovl_xmin, ovl_ymax), (ovl_xmax, ovl_ymin),
                                color, 2)

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
                    try:
                        vertices = np.array(order_points(vertices), np.int32)
                    except ValueError:
                        print(self.frame_idx, camera, vertices)
                    cv2.polylines(img_frame, [vertices],
                                  True, color)

            elif key == 'center_interaction':
                color = (163, 32, 219, 255)

        return img_frame

    def frame_transform(self, frame):

        # Read FEATURE_DATA
        current_frame = self.frame_idx
        if current_frame not in self.group_feature_data:
            return frame

        frame_data = json.load(
            open(self.group_feature_data[current_frame], 'r'))
        # is_raw_data_valid = frame_data['is_raw_data_valid']
        # is_enhanced_data_valid = frame_data['is_enhanced_data_valid']
        subjects_data = frame_data['subjects']

        openpose_data = dict()
        # openface_data = dict()
        # densepose_data = dict()

        frame_subject_data = {
            'openpose': dict(),
            'densepose': dict(),
            'openface': dict(),
            'video': dict()
        }

        data_type = None

        if self.gui_state['overlay_data_raw']:
            data_type = 'raw'
        elif self.gui_state['overlay_data_enhanced']:
            data_type = 'enhanced'
        else:
            print("Error",
                  "Invalid type of feature data. must be {raw, enhanced}")
            exit()

        for subject in subjects_data:
            # Openpose
            subject_id = subject['id']
            frame_subject_data['id'] = subject_id
            # print(self.camera)
            # Pose
            if self.gui_state['overlay_framework_pose']:
                subject_type_data = subject[data_type]
                subject_type_data_pose = subject_type_data['pose'] \
                    if 'pose' in subject_type_data else dict()
                subject_type_data_pose_openpose = subject_type_data_pose['openpose'] \
                    if 'openpose' in subject_type_data_pose else dict()
                openpose_data['pose'] = subject_type_data_pose_openpose[self.camera] \
                    if self.camera in subject_type_data_pose_openpose else dict()

                # print(subject_type_data)
                # print(subject_type_data_pose_openpose)

            # Face
            if self.gui_state['overlay_framework_face']:
                subject_type_data = subject[data_type]
                subject_type_data_face = subject_type_data['face'] \
                    if 'face' in subject_type_data else dict()
                subject_type_data_face_openpose = subject_type_data_face['openpose'] \
                    if 'openpose' in subject_type_data_face else dict()
                openpose_data['face'] = subject_type_data_face_openpose[self.camera] \
                    if self.camera in subject_type_data_face_openpose else dict()

            # Expansiveness + Overlap
            if self.gui_state['overlay_framework_overlap']:
                openpose_data['expansiveness'] = subject['metrics']['expansiveness']
                if 'overlap' in subject['metrics']:
                    openpose_data['overlap'] = subject['metrics']['overlap']

            # Intragroup Distance
            if self.gui_state['overlay_framework_intragroup_distance']:
                openpose_data['intragroup_distance'] = frame_data['group']['intragroup_distance']

            frame_subject_data['openpose'] = openpose_data
            if frame_subject_data['openpose']:
                frame = self.openpose_overlay(subject_id, frame_subject_data['openpose'],
                                              frame, self.camera)

            # Center Interaction
            if self.gui_state['overlay_framework_center_interaction']:
                print(subject['metrics'])
                # openpose_data['center_interaction'] = subject['metrics']['center_interaction']

            # Densepose

            # Openface

        # Video
        if self.gui_state['overlay_video_energy_heatmap']:
            frame = self.video_overlay(frame)

        return frame

    def densepose_overlay(self):
        print("Densepose Overlay")

    def openface_overlay(self):
        print("Openface Overlay")

    def video_overlay(self, frame):
        heatmap_file = [str(x) for x in self.group_data_path.iterdir()
                        if self.camera in x.name
                        and x.suffix in VALID_OUTPUT_IMG_TYPES][0]

        heatmap = cv2.imread(heatmap_file, cv2.IMREAD_UNCHANGED)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGRA2RGBA)

        frame = cv2.addWeighted(frame, 1, heatmap,
                                self.overlay_heatmap_transparency, 0)

        return frame


class VideoPlayerMonitor(object):
    def __init__(self, visualizer: QtWidgets.QMainWindow, ui: Ui_Visualizer, groups_directories: dict, q_components: dict, feature_analyzer: FeatureAnalyzer):
        self.visualizer = visualizer
        self.ui = ui
        self.groups_directories = groups_directories
        self.q_components = q_components
        self.feature_analyzer = feature_analyzer

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
        self.video_length = 0
        self.is_playing = False

        self.timer = QtCore.QTimer()

        self.btn_play = q_components['btn_play']
        self.btn_play.clicked.connect(self.play)
        self.btn_back = q_components['btn_back']
        self.btn_forward = q_components['btn_forward']
        self.sld_time = q_components['sld_time']
        self.sld_time.sliderMoved.connect(self.set_frame_by_slider)
        self.spn_frame_idx = q_components['spn_frame_idx']

        self.overlay_heatmap_transparency = 0.5
        self.gui_state = self.check_gui_state()

        self.timer.timeout.connect(self.playing_loop)
        self.visualizer.show()

    def playing_loop(self, update: bool = False):
        if self.is_playing or update:

            self.gui_state = self.check_gui_state()

            if self.current_frame <= self.video_length:
                self.ui.video_1.play_step(self.current_frame, self.gui_state)
                self.ui.video_2.play_step(self.current_frame, self.gui_state)
                self.ui.video_3.play_step(self.current_frame, self.gui_state)

                self.spn_frame_idx.setValue(self.current_frame)
                self.sld_time.setValue(self.current_frame)
                self.current_frame += 1
            else:
                self.current_frame = 0
                self.pause(replay=True)

    def play(self):
        if not self.is_playing:
            self.start()
        else:
            self.pause()

    def initialize_player_threads(self):
        self.selected_group = self.ui.cb_groupId.currentText()
        selected_group_dir = self.groups_directories[self.selected_group]

        self.selected_task = self.ui.cb_task.currentText()
        selected_group_task = selected_group_dir / self.selected_task
        selected_group_videos = [x for x in selected_group_task.iterdir()
                                 if not x.is_dir()]

        feature_data_path = DATASET_SYNC / self.selected_group / \
            FEATURE_AGGREGATE_DIR / self.selected_task
        self.feature_data_path = feature_data_path

        self.group_feature_data = {int(re.search(r'(\d{12})', x.name).group(0)): x
                                   for x in feature_data_path.iterdir()
                                   if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES}

        for video in selected_group_videos:

            if 'pc1' in video.name:
                self.ui.video_1 = VideoPlayer(1, video, self.ui.video_1, self.timer,
                                              feature_data_path, self.group_feature_data)

            elif 'pc2' in video.name:
                self.ui.video_2 = VideoPlayer(2, video, self.ui.video_2, self.timer,
                                              feature_data_path, self.group_feature_data)

            elif 'pc3' in video.name:
                self.ui.video_3 = VideoPlayer(3, video, self.ui.video_3, self.timer,
                                              feature_data_path, self.group_feature_data)

        self.ui.btn_play.setEnabled(True)
        self.ui.btn_back.setEnabled(True)
        self.ui.btn_skip.setEnabled(True)
        self.ui.time_slider.setEnabled(True)

        self.ui.video_1.play_step(gui_state=self.gui_state)
        self.ui.video_2.play_step(gui_state=self.gui_state)
        self.ui.video_3.play_step(gui_state=self.gui_state)

        self.video_length = set({self.ui.video_1.length,
                                 self.ui.video_2.length,
                                 self.ui.video_3.length})

        self.video_framerate = set({self.ui.video_1.fps,
                                    self.ui.video_2.fps,
                                    self.ui.video_3.fps})

        if len(self.video_length) > 1:
            print("DIFFERENT SIZED VIDEOS", self.video_length)
            # exit()
        self.video_length = list(self.video_length)[0]

        self.players = [self.ui.video_1,
                        self.ui.video_2,
                        self.ui.video_3]

        if len(self.video_framerate) > 1:
            print("VIDEOS WITH DIFFERENT FRAMERATE")
            exit()
        self.video_framerate = list(self.video_framerate)[0]

        self.sld_time.setMinimum(0)
        self.sld_time.setMaximum(self.video_length)
        self.sld_time.setTickInterval(1)
        self.timer.timeout.connect(self.playing_loop)

    def check_gui_state(self):
        components = self.q_components

        state = {
            'overlay_data_raw': components['radbtn_raw'].isChecked(),
            'overlay_data_enhanced': components['radbtn_enhanced'].isChecked(),
            'overlay_framework': components['cb_framework'],
            'overlay_framework_pose': components['chb_op_pose'].isChecked(),
            'overlay_framework_face': components['chb_op_face'].isChecked(),
            'overlay_framework_overlap': components['chb_op_overlap'].isChecked(),
            'overlay_framework_intragroup_distance': components['chb_op_ig_dist'].isChecked(),
            'overlay_framework_center_interaction': components['chb_op_cntr_int'].isChecked(),
            'overlay_openface_face':  components['chb_of_face'].isChecked(),
            # 'overlay_openface_aus':  components['chb_of_aus'].isChecked(),
            'overlay_video_energy_heatmap': components['chb_vid_energy_htmp'].isChecked(),
            'overlay_heatmap_transparency': self.overlay_heatmap_transparency,
        }

        return state

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

        data = {
            'group_id': self.selected_group,
            'task': self.selected_task,
            'path': self.feature_data_path
        }

        self.feature_analyzer.set_data(data)

    def jump_frames(self, increment):
        self.current_frame += increment-1
        if self.current_frame < 0:
            self.current_frame = 0
        self.playing_loop(True)

    def set_frame_by_slider(self, position):
        self.current_frame = position
        self.playing_loop(True)

    def set_frame_by_spinner(self):
        self.current_frame = int(self.spn_frame_idx.value())
        self.playing_loop(True)

    def update_heatmap_transparency(self, position):
        self.overlay_heatmap_transparency = position / 10

    def start(self):
        self.is_playing = True
        self.timer.start(1000.0/self.video_framerate)
        self.btn_play.setIcon(self.visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPause))

    def pause(self, replay: bool = False):
        self.is_playing = False
        self.timer.stop()
        self.btn_play.setIcon(self.visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay))
        if replay:
            self.btn_play.setIcon(self.visualizer.style().standardIcon(
                QtWidgets.QStyle.SP_BrowserReload))


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
        self.feature_analyzer = FeatureAnalyzer()
        self.ui.action_feature_analyzer.triggered.connect(
            self.feature_analyzer.open)
        self.ui.actionExit.triggered.connect(QtWidgets.QApplication.quit)

        # Group Frame Components
        group_names = list(self.group_dirs.keys())
        group_names.sort()
        self.ui.cb_groupId.addItems(group_names)

        # Controls Frame Components
        btn_play = self.ui.btn_play
        btn_back = self.ui.btn_back
        btn_skip = self.ui.btn_skip

        btn_play.setIcon(visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay))
        btn_back.setIcon(visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaSeekBackward))
        btn_skip.setIcon(visualizer.style().standardIcon(
            QtWidgets.QStyle.SP_MediaSeekForward))

        sld_transparency = self.ui.sld_transparency

        sld_transparency.setMinimum(0)
        sld_transparency.setMaximum(10)
        sld_transparency.setValue(5)
        sld_transparency.setTickInterval(1)
        sld_transparency.setTickPosition(5)

        btn_frame_goto = self.ui.btn_frame_go

        self.q_components['btn_play'] = self.ui.btn_play
        self.q_components['btn_back'] = self.ui.btn_back
        self.q_components['btn_forward'] = self.ui.btn_skip
        self.q_components['sld_time'] = self.ui.time_slider
        self.q_components['chb_op_pose'] = self.ui.chb_op_pose
        self.q_components['chb_op_face'] = self.ui.chb_op_face
        self.q_components['chb_op_overlap'] = self.ui.chb_op_overlap
        self.q_components['chb_op_cntr_int'] = self.ui.chb_op_cntr_int
        self.q_components['chb_op_ig_dist'] = self.ui.chb_op_ig_dist
        self.q_components['chb_vid_energy_htmp'] = self.ui.chb_vid_energy_htmp
        self.q_components['spn_frame_idx'] = self.ui.spn_frame_idx
        self.q_components['frame_goto_btn'] = self.ui.btn_frame_go
        self.q_components['cb_framework'] = self.ui.cb_framework
        self.q_components['radbtn_raw'] = self.ui.radbtn_raw
        self.q_components['radbtn_enhanced'] = self.ui.radbtn_enh
        self.q_components['chb_of_face'] = self.ui.chb_op_face
        self.q_components['sld_overlay_transp'] = self.ui.sld_transparency

        monitor_GUI = VideoPlayerMonitor(
            visualizer, self.ui, self.group_dirs, self.q_components, self.feature_analyzer)

        btn_play.clicked.connect(monitor_GUI.playing_loop)
        btn_back.clicked.connect(lambda: monitor_GUI.jump_frames(-10))
        btn_skip.clicked.connect(lambda: monitor_GUI.jump_frames(10))
        btn_frame_goto.clicked.connect(monitor_GUI.set_frame_by_spinner)
        sld_transparency.sliderMoved.connect(
            monitor_GUI.update_heatmap_transparency)

        sys.exit(app.exec_())

    def closeEvent(self):
        print("Closing ...")


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.main()
